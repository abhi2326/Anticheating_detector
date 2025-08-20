import os
import time
import csv
import math
import pathlib
from collections import defaultdict

import cv2
import numpy as np

# YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO not installed. Run: pip install ultralytics") from e

# MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ===================== CONFIG =====================
CAM_INDEX = 0
FPS_LIMIT = 30
MODEL_PATH = "yolov8n.pt"           # small & fast; change to yolov8s.pt for more accuracy
CONF_THRESH = 0.4                    # YOLO confidence threshold
IOU_THRESH = 0.45

# Proctoring thresholds
ALERT_HOLD = 2.0                     # seconds required to latch an alert
COOLDOWN = 1.0                       # seconds false before clearing
FACE_ABSENT_HOLD = 2.0               # how long without face -> alert
YAW_DEG_THRESHOLD = 28               # abs(yaw) beyond this => looking away
PITCH_DEG_THRESHOLD = 22             # abs(pitch) beyond this => looking away
LOOKING_AWAY_HOLD = 2.0
SMOOTHING_ALPHA = 0.25               # EMA for yaw/pitch

# What to watch
WATCH_PHONE = True
WATCH_BOOK = True
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = "cheat_snaps"
LOG_PATH = "cheating_logs.csv"
DRAW_LANDMARKS = False               # toggle face mesh drawing

# ===================== HELPERS =====================

def ema(prev, new, alpha=SMOOTHING_ALPHA):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new


def estimate_head_pose_simple(landmarks, w, h):
    """Heuristic 2D yaw/pitch using FaceMesh keypoints."""
    idx_left_eye, idx_right_eye, idx_nose, idx_chin = 33, 263, 1, 152
    try:
        lx, ly = landmarks[idx_left_eye].x * w,  landmarks[idx_left_eye].y * h
        rx, ry = landmarks[idx_right_eye].x * w, landmarks[idx_right_eye].y * h
        nx, ny = landmarks[idx_nose].x * w,     landmarks[idx_nose].y * h
        cx, cy = landmarks[idx_chin].x * w,     landmarks[idx_chin].y * h
    except Exception:
        return 0.0, 0.0
    midx, midy = (lx + rx) / 2.0, (ly + ry) / 2.0
    yaw = (nx - midx) / (abs(rx - lx) + 1e-6)
    yaw_deg = float(np.degrees(np.arctan(yaw)))
    eye_to_chin = abs(cy - midy) + 1e-6
    pitch = (midy - ny) / eye_to_chin
    pitch_deg = float(np.degrees(np.arctan(pitch)))
    return yaw_deg, pitch_deg


class DebouncedFlag:
    def __init__(self, hold=ALERT_HOLD, cooldown=COOLDOWN):
        self.hold = hold
        self.cooldown = cooldown
        self.state = False
        self.since = None
        self.clear_since = None

    def update(self, condition, now):
        if condition:
            if not self.state:
                if self.since is None:
                    self.since = now
                if now - self.since >= self.hold:
                    self.state = True
                    self.clear_since = None
            else:
                self.clear_since = None
        else:
            self.since = None
            if self.state:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= self.cooldown:
                    self.state = False


class Logger:
    def __init__(self, path):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        try:
            with open(self.path, "x", newline="") as f:
                csv.writer(f).writerow(["timestamp", "alert_type", "details"]) 
        except FileExistsError:
            pass

    def write(self, ts, alert_type, details):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([ts, alert_type, details])


# ===================== MAIN =====================

def main():
    # Prepare snapshot dir
    if SAVE_SNAPSHOTS:
        pathlib.Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Map names for convenience
    names = model.model.names if hasattr(model.model, 'names') else model.names
    # Try to resolve target class IDs by name
    person_id = None
    phone_id = None
    book_id = None
    for k, v in names.items():
        name = str(v).lower()
        if name == 'person':
            person_id = k
        if WATCH_PHONE and name in ('cell phone', 'mobile phone', 'phone'):
            phone_id = k
        if WATCH_BOOK and name == 'book':
            book_id = k

    if person_id is None:
        print("[WARN] YOLO model does not have 'person' class; person counting will be disabled.")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Change CAM_INDEX or check permissions.")

    logger = Logger(LOG_PATH)

    # MediaPipe FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    yaw_s, pitch_s = None, None
    last_face_time = time.time()

    # Alerts
    f_multi_person = DebouncedFlag(hold=ALERT_HOLD, cooldown=COOLDOWN)
    f_phone = DebouncedFlag(hold=ALERT_HOLD, cooldown=COOLDOWN)
    f_book = DebouncedFlag(hold=ALERT_HOLD, cooldown=COOLDOWN)
    f_face_absent = DebouncedFlag(hold=FACE_ABSENT_HOLD, cooldown=COOLDOWN)
    f_away = DebouncedFlag(hold=LOOKING_AWAY_HOLD, cooldown=COOLDOWN)

    latched_msgs_last_ts = {}

    frame_interval = 1.0 / FPS_LIMIT
    try:
        while True:
            loop_t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------- YOLO inference ----------------
            yolo_res = model.predict(source=rgb, imgsz=max(320, min(w, h)), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
            boxes = yolo_res.boxes

            person_count = 0
            phone_found = False
            book_found = False

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    c = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                    conf = float(b.conf.item()) if hasattr(b.conf, 'item') else float(b.conf)
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    label = names.get(c, str(c))

                    # draw boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

                    if c == person_id:
                        person_count += 1
                    if phone_id is not None and c == phone_id:
                        phone_found = True
                    if book_id is not None and c == book_id:
                        book_found = True

            # ---------------- Face Mesh (head pose) ----------------
            face_res = face_mesh.process(rgb)
            yaw_deg, pitch_deg = 0.0, 0.0
            face_present = False
            if face_res.multi_face_landmarks:
                face_present = True
                last_face_time = time.time()
                lm = face_res.multi_face_landmarks[0]
                yaw_raw, pitch_raw = estimate_head_pose_simple(lm.landmark, w, h)
                yaw_s = ema(yaw_s, yaw_raw)
                pitch_s = ema(pitch_s, pitch_raw)
                yaw_deg = yaw_s if yaw_s is not None else yaw_raw
                pitch_deg = pitch_s if pitch_s is not None else pitch_raw
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        frame, lm,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )

            # ---------------- Rules update ----------------
            now = time.time()
            f_multi_person.update(person_count > 1, now)
            f_phone.update(phone_found, now)
            f_book.update(book_found, now)
            f_face_absent.update((now - last_face_time) > FACE_ABSENT_HOLD, now)
            f_away.update((abs(yaw_deg) > YAW_DEG_THRESHOLD) or (abs(pitch_deg) > PITCH_DEG_THRESHOLD), now)

            # ---------------- Logging & snapshots when latch ----------------
            def latch_log(flag: DebouncedFlag, key: str, details: str):
                prev = latched_msgs_last_ts.get(key)
                if flag.state and (prev is None):
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                    logger.write(ts, key, details)
                    if SAVE_SNAPSHOTS:
                        fn = f"{key}_{int(now)}.jpg"
                        cv2.imwrite(os.path.join(SNAPSHOT_DIR, fn), frame)
                    latched_msgs_last_ts[key] = now
                if not flag.state and prev is not None:
                    latched_msgs_last_ts.pop(key, None)

            latch_log(f_multi_person, "MULTIPLE_PERSONS", f"count={person_count}")
            latch_log(f_phone, "PHONE_DETECTED", "cell phone visible")
            latch_log(f_book, "BOOK_DETECTED", "book visible")
            latch_log(f_face_absent, "FACE_ABSENT", f"no face for > {FACE_ABSENT_HOLD}s")
            latch_log(f_away, "LOOKING_AWAY", f"yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}")

            # ---------------- HUD ----------------
            y = 26
            cv2.putText(frame, "\U0001F575\uFE0F Proctoring (YOLOv8 + FaceMesh)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2); y += 26
            cv2.putText(frame, f"Persons: {person_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1); y += 20
            cv2.putText(frame, f"Yaw: {yaw_deg:.1f}  Pitch: {pitch_deg:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1); y += 24

            if f_multi_person.state:
                cv2.putText(frame, "\u26A0\uFE0F Multiple persons detected", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); y += 24
            if f_phone.state:
                cv2.putText(frame, "\u26A0\uFE0F Phone detected", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); y += 24
            if f_book.state:
                cv2.putText(frame, "\u26A0\uFE0F Book detected", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); y += 24
            if f_face_absent.state:
                cv2.putText(frame, "\u26A0\uFE0F Face absent", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); y += 24
            if f_away.state:
                cv2.putText(frame, "\u26A0\uFE0F Looking away", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); y += 24

            # Show
            cv2.imshow("Proctoring: YOLOv8 + MediaPipe", frame)

            # FPS limit
            elapsed = time.time() - loop_t0
            wait = max(0.0, frame_interval - elapsed)
            if wait > 0:
                time.sleep(wait)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
