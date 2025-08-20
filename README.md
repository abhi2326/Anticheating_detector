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

🖥 Cross-platform packaging (Windows/Linux/macOS executables)
