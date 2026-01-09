import cv2
import os
import time
import csv
import numpy as np
import smtplib
from collections import deque
from email.message import EmailMessage
from datetime import datetime
from pymongo import MongoClient
from insightface.app import FaceAnalysis

# ===============================
# BASE PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# MONGODB
# ===============================
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_surveillance"]
violation_col = db["violations"]

# ===============================
# EMAIL CONFIG
# ===============================
SENDER_EMAIL = "cresisconnectora@gmail.com"
APP_PASSWORD = "tnjb yglp qnpq wedq"
PRINCIPAL_EMAIL = "amdarshan557@gmail.com"

# ===============================
# LOAD STUDENTS
# ===============================
usn_to_name = {}
with open(os.path.join(BASE_DIR, "students.csv"), newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        usn_to_name[row["usn"].strip().upper()] = row["name"].strip()

print("📄 Students loaded:", usn_to_name)

# ===============================
# FACE MODEL
# ===============================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ===============================
# LOAD REGISTERED FACES
# ===============================
known_embeddings = []
known_usns = []
dataset_path = os.path.join(BASE_DIR, "dataset")

for usn in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, usn)
    if not os.path.isdir(folder):
        continue
    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)
        if faces:
            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)
            known_embeddings.append(emb)
            known_usns.append(usn.upper())

print(f"✅ Loaded {len(known_embeddings)} registered faces")

# ===============================
# CONFIG
# ===============================
FACE_THRESHOLD = 1.0
YAW_THRESHOLD = 0.20
EXTREME_YAW = 0.30
PITCH_DOWN_THRESHOLD = 0.25
LOOK_DURATION = 1.5
COOLDOWN = 6
SMOOTH_WINDOW = 7

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence", "looking_sideways")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

yaw_buffers = {}
pitch_buffers = {}
look_start_times = {}
last_event_time = 0

# ===============================
# EMAIL FUNCTION
# ===============================
def send_email(subject, body, img_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = PRINCIPAL_EMAIL
        msg.set_content(body)
        with open(img_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image",
                               subtype="jpeg", filename=os.path.basename(img_path))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("📧 Email sent")
    except Exception as e:
        print("❌ Email failed:", e)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
print("👀 Looking sideways detection started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)

    unknown_detected = False

    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        distances = np.linalg.norm(np.array(known_embeddings) - emb, axis=1)
        idx = np.argmin(distances)

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

        # ---------------- UNKNOWN PERSON ----------------
        if distances[idx] > FACE_THRESHOLD:
            unknown_detected = True
            cv2.putText(frame, "UNKNOWN", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        # ---------------- REGISTERED STUDENT ----------------
        usn = known_usns[idx]
        name = usn_to_name.get(usn, "Unknown")

        lm = face.landmark_2d_106
        nose_x, nose_y = lm[49]
        left_eye_x, left_eye_y = lm[36]
        right_eye_x, right_eye_y = lm[45]
        chin_x, chin_y = lm[6]

        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2
        eye_width = max(abs(right_eye_x - left_eye_x), 1)

        raw_yaw = (nose_x - eye_center_x) / eye_width
        raw_pitch = (chin_y - nose_y) / eye_width

        raw_yaw = -raw_yaw  # mirror correction

        yaw_buffers.setdefault(usn, deque(maxlen=SMOOTH_WINDOW)).append(raw_yaw)
        pitch_buffers.setdefault(usn, deque(maxlen=SMOOTH_WINDOW)).append(raw_pitch)

        smooth_yaw = float(np.mean(yaw_buffers[usn]))
        smooth_pitch = float(np.mean(pitch_buffers[usn]))

        direction = "STRAIGHT"
        if smooth_yaw > YAW_THRESHOLD:
            direction = "LEFT"
        elif smooth_yaw < -YAW_THRESHOLD:
            direction = "RIGHT"
        elif smooth_pitch > PITCH_DOWN_THRESHOLD:
            direction = "DOWN"

        extreme = abs(smooth_yaw) > EXTREME_YAW

        now = time.time()
        if extreme:
            look_start_times.setdefault(usn, now)
            if now - look_start_times[usn] > LOOK_DURATION:
                if now - last_event_time > COOLDOWN:
                    filename = f"{usn}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    path = os.path.join(EVIDENCE_DIR, filename)
                    cv2.imwrite(path, frame)

                    violation_col.insert_one({
                        "usn": usn,
                        "name": name,
                        "violation": f"Extreme Looking {direction}",
                        "yaw": round(float(smooth_yaw),3),
                        "pitch": round(float(smooth_pitch),3),
                        "evidence_path": path,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M:%S")
                    })

                    # ⚠️ Email NOT sent for registered students
                    last_event_time = now
                    look_start_times[usn] = now
        else:
            look_start_times.pop(usn, None)

        cv2.putText(frame, f"{usn} {name}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"{direction}", (x1, y2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,0,255) if extreme else (0,255,0), 2)

    # ---------------- UNKNOWN ALERT ----------------
    if unknown_detected:
        filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(BASE_DIR, "evidence", "unknown_faces", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frame)

        violation_col.insert_one({
            "usn": "UNKNOWN",
            "name": "Unknown Person",
            "violation": "Unrecognized face detected",
            "evidence_path": path,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        })

        subject = "🚨 Unknown Person Detected During Exam"
        body = f"""Violation: Unrecognized face detected
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        send_email(subject, body, path)

    cv2.imshow("Looking Sideways Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Looking sideways detection stopped")
