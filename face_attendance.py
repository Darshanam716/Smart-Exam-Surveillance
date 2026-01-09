import cv2
import os
import time
import csv
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from insightface.app import FaceAnalysis

# =====================================
# MongoDB SETUP
# =====================================
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_surveillance"]
attendance_col = db["attendance"]

# =====================================
# LOAD STUDENT CSV (USN → NAME)
# =====================================
usn_to_name = {}

with open("students.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        usn_clean = row["usn"].strip().upper()
        name_clean = row["name"].strip()
        usn_to_name[usn_clean] = name_clean

print("📄 Student mapping loaded:", usn_to_name)

# =====================================
# LOAD INSIGHTFACE MODEL
# =====================================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# =====================================
# LOAD REGISTERED FACE EMBEDDINGS
# =====================================
known_embeddings = []
known_usns = []

dataset_path = "dataset"

for folder_usn in os.listdir(dataset_path):
    folder_usn_clean = folder_usn.strip().upper()
    folder_path = os.path.join(dataset_path, folder_usn)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if faces:
            emb = faces[0].embedding
            emb_norm = emb / np.linalg.norm(emb)  # normalize
            known_embeddings.append(emb_norm)
            known_usns.append(folder_usn_clean)

print(f"✅ Loaded {len(known_embeddings)} face samples")

if len(known_embeddings) == 0:
    print("❌ No registered faces found. Run face_register.py first.")
    exit()

# =====================================
# START ATTENDANCE CAMERA
# =====================================
cap = cv2.VideoCapture(0)
start_time = time.time()
attendance_marked = set()

ATTENDANCE_DURATION = 60  # seconds
MATCH_THRESHOLD = 1.0     # normalized threshold

print("🟢 Attendance started (1 minute)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)

    print(f"Faces detected: {len(faces)}")

    for face in faces:
        emb = face.embedding
        emb_norm = emb / np.linalg.norm(emb)  # normalize live embedding

        distances = np.linalg.norm(np.array(known_embeddings) - emb_norm, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        usn = known_usns[min_index]
        name = usn_to_name.get(usn, "Unknown")

        # -----------------------------
        # DRAW FACE BOX ALWAYS
        # -----------------------------
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_y = y1 - 10 if y1 - 40 > 0 else y2 + 25
        cv2.putText(frame, f"Name: {name}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"USN: {usn}", (x1, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -----------------------------
        # MARK ATTENDANCE (CONFIDENT)
        # -----------------------------
        print(f"Closest match: {usn}, distance={min_distance:.2f}")

        if min_distance < MATCH_THRESHOLD and usn not in attendance_marked:
            now = datetime.now()
            record = {
                "usn": usn,
                "name": name,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "status": "Present"
            }
            attendance_col.insert_one(record)
            attendance_marked.add(usn)

            print(f"✅ Attendance marked for {usn} ({name})")
            print(f"📏 Match distance: {round(min_distance, 3)}")

    cv2.imshow("Attendance System", frame)

    if time.time() - start_time > ATTENDANCE_DURATION:
        print("⏹ Attendance window closed")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Attendance session ended.")

# =====================================
# FINALIZE ATTENDANCE (MARK ABSENTEES)
# =====================================
for usn, name in usn_to_name.items():
    if usn in attendance_marked:
        continue
    now = datetime.now()
    record = {
        "usn": usn,
        "name": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": "Absent"
    }
    attendance_col.insert_one(record)
    print(f"❌ Absent marked for {usn} ({name})")
