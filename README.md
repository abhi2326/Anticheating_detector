# 🎥 AI Proctoring System – Cheating Detection

An **AI-powered proctoring system** that detects cheating behaviors in real-time using **YOLOv8** for object detection and **MediaPipe FaceMesh** for head pose estimation.  
It is designed for **online exams, assessments, and remote proctoring**, providing alerts and logs when suspicious activities are detected.

---

## 🚀 Features
- ✅ **Multi-person detection** (flags if more than one person is present)  
- 📱 **Phone detection** (detects use of mobile devices)  
- 📖 **Book detection** (optional toggle for open-book exams)  
- 👀 **Face absence tracking** (alerts if candidate leaves frame)  
- 🔄 **Head pose estimation** (flags looking away for too long)  
- 📊 **Cheating log system** (all alerts saved in CSV file)  
- 🖼 **Optional snapshot capture** (saves frames where cheating was detected)  
- ⏳ **Debouncing mechanism** (avoids false positives due to brief movements)  

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-proctoring-system.git
cd ai-proctoring-system
2. Install dependencies
pip install ultralytics opencv-python mediapipe numpy
pip uninstall -y protobuf
pip install protobuf==3.20.3

▶️ Usage

Run the main script:

python Proctoring_AI_CheatingDetector_YOLO_MediaPipe.py

⚙️ Configuration

At the top of the script, you can adjust thresholds:

LOOKING_AWAY_THRESH_YAW = 28 → Max horizontal head angle (°)

LOOKING_AWAY_THRESH_PITCH = 22 → Max vertical head angle (°)

LOOKING_AWAY_HOLD = 1.2 → Seconds before “looking away” is flagged

FACE_ABSENT_HOLD = 2.0 → Seconds face can be missing before alert

WATCH_BOOK = True → Enable/disable book detection

📊 Logs & Output

All cheating events are saved to cheating_log.csv with:

Timestamp

Detected behavior (phone, book, face absent, etc.)

Optionally saves snapshots of cheating moments (toggle in script).

🧩 Tech Stack

YOLOv8 (Ultralytics) – Person, phone, and book detection

MediaPipe FaceMesh – Facial landmark & head pose estimation

OpenCV – Video stream processing and visualization

Python – Core logic and alert system

📌 Future Improvements

🎙 Audio analysis (detect multiple voices or background noise)

🌐 Live proctor dashboard (web-based monitoring)

🖥 Cross-platform packaging (Windows/Linux/macOS executables)
