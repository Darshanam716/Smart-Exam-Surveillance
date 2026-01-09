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
MOUTH_RATIO_THRESHOLD = 0.38
TALK_DURATION = 2.2
COOLDOWN = 6
SMOOTH_WINDOW = 6

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence", "talking")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ===============================
# STATE (PER STUDENT)
# ===============================
mouth_buffers = {}        # usn -> deque
talk_start_times = {}     # usn -> time
last_event_times = {}     # usn -> time

# ===============================
# EMAIL FUNCTION
# ===============================
def send_email(usn, name, img_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Exam Malpractice Alert – Talking Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = PRINCIPAL_EMAIL

    msg.set_content(
        f"""Student Name: {name}
USN: {usn}
Violation: Talking Detected
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    )

    with open(img_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(img_path)
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print(f"📧 Email sent for {usn} {name}")

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
print("🗣 Talking detection started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)

    current_talkers = []

    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        distances = np.linalg.norm(np.array(known_embeddings) - emb, axis=1)
        idx = np.argmin(distances)

        if distances[idx] > FACE_THRESHOLD:
            continue  # skip unknowns

        usn = known_usns[idx]
        name = usn_to_name.get(usn, "Unknown")

        lm = face.landmark_2d_106
        upper_lip = lm[62]
        lower_lip = lm[66]
        left_mouth = lm[60]
        right_mouth = lm[64]

        mouth_open = np.linalg.norm(upper_lip - lower_lip)
        mouth_width = max(np.linalg.norm(left_mouth - right_mouth), 1.0)
        mouth_ratio = mouth_open / mouth_width

        mouth_buffers.setdefault(usn, deque(maxlen=SMOOTH_WINDOW))
        mouth_buffers[usn].append(mouth_ratio)
        smooth_ratio = np.mean(mouth_buffers[usn])

        now = time.time()
        if smooth_ratio > MOUTH_RATIO_THRESHOLD:
            if usn not in talk_start_times:
                talk_start_times[usn] = now
            elif now - talk_start_times[usn] > TALK_DURATION:
                last_evt = last_event_times.get(usn, 0)
                if now - last_evt > COOLDOWN:
                    current_talkers.append((usn, name, smooth_ratio))
                    last_event_times[usn] = now
                    talk_start_times[usn] = now
        else:
            talk_start_times.pop(usn, None)

        # UI overlay
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{usn} {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MOUTH:{smooth_ratio:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # === After processing all faces ===
    if current_talkers:
        ts = datetime.now()
        filename = f"TALK_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(EVIDENCE_DIR, filename)
        cv2.imwrite(path, frame.copy())

        for usn, name, ratio in current_talkers:
            violation_col.insert_one({
                "usn": usn,
                "name": name,
                "violation": "Talking Detected",
                "mouth_ratio": float(round(ratio, 2)),
                "evidence_path": path,
                "date": ts.strftime("%Y-%m-%d"),
                "time": ts.strftime("%H:%M:%S")
            })
            send_email(usn, name, path)

    cv2.imshow("Talking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Talking detection stopped")
