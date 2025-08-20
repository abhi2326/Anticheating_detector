🎥 Proctoring AI – Cheating Detection System

This project is an AI-powered online exam proctoring tool that uses YOLOv8 object detection and MediaPipe FaceMesh to detect suspicious activities such as:

✅ Multiple people in the frame
✅ Use of mobile phones or books
✅ Looking away from the screen
✅ Face absence (student leaves camera view)

It provides real-time alerts, an on-screen HUD, CSV logs, and optional snapshot evidence.

🚀 Features

YOLOv8 Detection – Detects person, phone, and books in real time

MediaPipe FaceMesh – Estimates head orientation (yaw/pitch)

Cheating Alerts – Raised for multiple people, phone usage, book usage, or looking away

Debouncing – Reduces false positives by requiring sustained violations

Logging – Saves cheating events with timestamps to a CSV file

Snapshot Evidence – (Optional) Stores screenshots when alerts trigger

Configurable Thresholds – Adjust sensitivity for yaw, pitch, and absence detection

📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/proctoring-ai.git
cd proctoring-ai
pip install ultralytics opencv-python mediapipe numpy
pip uninstall -y protobuf
pip install protobuf==3.20.3

▶️ Usage

Run the script:

python Proctoring_AI_CheatingDetector_YOLO_MediaPipe.py


Press q to quit the application.

⚙️ Configuration

At the top of the script, you can tweak thresholds:

LOOKING_AWAY_YAW = 28      # Degrees left/right
LOOKING_AWAY_PITCH = 22    # Degrees up/down
LOOKING_AWAY_HOLD = 1.5    # Seconds before alert
FACE_ABSENT_HOLD = 2.0     # Seconds before alert
WATCH_BOOK = True          # Enable/Disable book detection

📊 Output

On-screen HUD: Displays real-time detection results and alert messages

CSV Logs: Saved in cheating_log.csv with timestamp and type of cheating

Snapshots: (Optional) Saved in snapshots/ folder whenever cheating is detected

📸 Example HUD
[ALERT] Multiple persons detected!
[ALERT] Phone usage detected!
[ALERT] Looking away from screen!

🛠 Tech Stack

Python 3.8+

YOLOv8 (Ultralytics) – Object detection

OpenCV – Video processing & visualization

MediaPipe – Face & head orientation tracking

NumPy – Math utilities

📌 Future Improvements

🔊 Audio checks for multiple voices

🌐 Remote logging / cloud storage of alerts

🎯 Advanced student behavior analysis (eye tracking, stress levels, etc.)

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork and submit PRs.
