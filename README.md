# 🎓 AI-Powered Smart Exam Surveillance System

An intelligent exam proctoring solution that uses **computer vision and AI-based behavior analysis** to detect cheating in real time. The system monitors candidates during exams, identifies suspicious activities, generates explainable alerts, and stores verifiable evidence for invigilators.

---

## 🚀 Project Overview

The **AI-Powered Smart Exam Surveillance System** is designed to improve exam integrity in both **online and offline examination environments**. Unlike traditional rule-based systems, this project uses **multi-signal analysis** to reduce false positives and ensure fairness.

It detects cheating attempts such as:
- Presence of multiple people
- Talking during the exam
- Excessive head or gaze movement
- Identity mismatch

All alerts are backed by **screenshots, timestamps, and clear reasons**, making the system transparent and reliable.

---

## 🧠 Key Features

- 👤 **Multi-Face Detection** – Detects more than one person in the camera frame  
- 🗣 **Talking Detection** – Identifies mouth movement indicating speech  
- 👀 **Head & Gaze Movement Tracking** – Flags suspicious viewing behavior  
- 🆔 **Face Verification** – Confirms the candidate’s identity  
- 📊 **Cheating Confidence Score** – Explainable suspicion scoring system  
- 📸 **Evidence Logging** – Saves screenshots with timestamps and reasons  
- ⚖️ **Fairness-Aware Alerts** – Warning-based escalation to avoid false accusations  
- 🔔 **Real-Time Monitoring** – Instant alerts for invigilators  

---

## 🏗 System Architecture

Camera Feed
↓
Face & Behavior Detection (YOLO + OpenCV)
↓
Multi-Signal Risk Scoring Engine
↓
Evidence Logger
↓
Live Alerts / Dashboard

yaml
Copy code

---

## 🛠 Tech Stack

- **Language:** Python  
- **Computer Vision:** OpenCV  
- **Object Detection:** YOLOv8  
- **Face Recognition:** InsightFace  
- **Backend:** Flask  
- **Database:** MongoDB / SQLite  
- **Numerical Processing:** NumPy  

---

## 🎥 Demo Workflow

1. Candidate behaves normally → 🟢 Safe  
2. Head or gaze deviation → ⚠ Warning  
3. Talking detected → Suspicion score increases  
4. Multiple faces detected → 🚨 Cheating flagged  
5. Screenshot and evidence saved automatically  
6. Invigilator reviews alert and evidence  

---

## 📁 Project Structure

├── models/
├── logs/
│ ├── screenshots/
│ └── alerts/
├── static/
├── templates/
├── main.py
├── config.py
└── requirements.txt

yaml
Copy code

---

## 🎯 Why This Project Stands Out

- Uses **multi-modal AI**, not single-rule detection  
- Provides **explainable decisions**, not black-box alerts  
- Generates **court-proof evidence**  
- Designed with **fairness and transparency**  
- Highly suitable for **hackathons, universities, and real deployment**

---

## 🔮 features

- Mobile phone detection  
- Face spoofing prevention  
- Audio noise classification  
- Cloud-based alert notifications  
- Encrypted evidence storage  

---

## 🏆 Hackathon Readiness

This project demonstrates strong **technical depth**, **real-world applicability**, and **ethical AI design**, making it ideal for hackathons, academic projects, and smart campus solutions.

---

## 📜 License

This project is licensed under the **MIT License**.
