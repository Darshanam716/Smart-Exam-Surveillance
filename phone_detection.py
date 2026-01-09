import cv2
import os
import time
import csv
import smtplib
import numpy as np
from email.message import EmailMessage
from datetime import datetime
from pymongo import MongoClient
from ultralytics import YOLO
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
# YOLO MODEL
# ===============================
phone_model = YOLO("yolov8n.pt")

# ===============================
# CONFIG
# ===============================
FACE_THRESHOLD = 1.0
PHONE_THRESHOLD = 0.25
COOLDOWN = 6

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence", "phone")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

last_event_time = 0

# ===============================
# EMAIL FUNCTION
# ===============================
def send_email(usn, name, conf, img_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Exam Malpractice Alert – Mobile Phone Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = PRINCIPAL_EMAIL

        msg.set_content(
            f"""Student Name: {name}
USN: {usn}
Violation: Mobile Phone Detected
Confidence: {conf}
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

        print("📧 Email sent")

    except Exception as e:
        print("❌ Email error:", e)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open camera")

print("📱 Phone detection running")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)

    current_faces = []  # store all faces present

    # ================= FACE IDENTITY =================
    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        distances = np.linalg.norm(np.array(known_embeddings) - emb, axis=1)
        idx = np.argmin(distances)

        if len(distances) > 0 and distances[idx] < FACE_THRESHOLD:
            usn = known_usns[idx]
            name = usn_to_name.get(usn, "Unknown")
        else:
            usn = "UNKNOWN"
            name = "Unknown Person"

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({usn})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        current_faces.append((usn, name))

    status = f"Faces detected: {len(current_faces)}"
    cv2.putText(frame, status, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # ================= PHONE DETECTION =================
    results = phone_model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            label = phone_model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if label == "cell phone" and conf > PHONE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,"PHONE",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                if time.time() - last_event_time < COOLDOWN:
                    continue

                now = datetime.now()
                filename = f"phone_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                path = os.path.join(EVIDENCE_DIR, filename)
                cv2.imwrite(path, frame)

                # log/email for each face present
                for usn, name in current_faces:
                    violation_col.insert_one({
                        "usn": usn,
                        "name": name,
                        "violation": "Mobile Phone Detected",
                        "confidence": round(conf,2),
                        "evidence_path": path,
                        "date": now.strftime("%Y-%m-%d"),
                        "time": now.strftime("%H:%M:%S")
                    })
                    send_email(usn, name, round(conf,2), path)

                last_event_time = time.time()
                print("🚨 PHONE CHEATING LOGGED")

    cv2.imshow("Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Phone detection stopped")
