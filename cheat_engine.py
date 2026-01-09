# cheat_engine.py (robust multi-face + hardened talking detection)
import cv2
import os
import time
import csv
import smtplib
import numpy as np
from collections import deque, defaultdict
from email.message import EmailMessage
from datetime import datetime
from pymongo import MongoClient
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ===============================
# BASE PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_BASE = os.path.join(BASE_DIR, "evidence")
EVIDENCE_PHONE = os.path.join(EVIDENCE_BASE, "phone")
EVIDENCE_TALK = os.path.join(EVIDENCE_BASE, "talking")
EVIDENCE_SIDE = os.path.join(EVIDENCE_BASE, "looking_sideways")
EVIDENCE_UNKNOWN = os.path.join(EVIDENCE_BASE, "unknown_faces")
for p in [EVIDENCE_PHONE, EVIDENCE_TALK, EVIDENCE_SIDE, EVIDENCE_UNKNOWN]:
    os.makedirs(p, exist_ok=True)

# ===============================
# DATABASE
# ===============================
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_surveillance"]
violation_col = db["violations"]
attendance_col = db.get_collection("attendance")

# ===============================
# EMAIL
# ===============================
SENDER_EMAIL = "cresisconnectora@gmail.com"
APP_PASSWORD = "tnjb yglp qnpq wedq"
PRINCIPAL_EMAIL = "amdarshan557@gmail.com"

def send_email(usn, name, violation, img_path, extra=""):
    try:
        msg = EmailMessage()
        subj = f"🚨 Exam Malpractice – {violation}" if usn != "UNKNOWN" else f"🚨 Unknown Person – {violation}"
        msg["Subject"] = subj
        msg["From"] = SENDER_EMAIL
        msg["To"] = PRINCIPAL_EMAIL
        msg.set_content(
            f"Student: {name}\nUSN: {usn}\nViolation: {violation}\n{extra}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        with open(img_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(img_path))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, APP_PASSWORD)
            s.send_message(msg)
        print("📧 Email sent:", violation, usn, name)
    except Exception as e:
        print("❌ Email failed:", e)

# ===============================
# LOAD STUDENTS
# ===============================
usn_to_name = {}
students_csv = os.path.join(BASE_DIR, "students.csv")
with open(students_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        usn_to_name[r["usn"].strip().upper()] = r["name"].strip()
print(f"📄 Students loaded: {len(usn_to_name)}")

# ===============================
# FACE MODEL
# ===============================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ===============================
# LOAD REGISTERED FACES
# ===============================
known_embeddings, known_usns = [], []
dataset = os.path.join(BASE_DIR, "dataset")
if not os.path.isdir(dataset):
    raise SystemExit(f"Dataset folder not found: {dataset}")

for usn in os.listdir(dataset):
    folder = os.path.join(dataset, usn)
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
            known_usns.append(usn.strip().upper())

if len(known_embeddings) == 0:
    raise SystemExit("❌ No registered faces found. Populate the dataset folder first.")
print(f"✅ Loaded {len(known_embeddings)} face samples")

# ===============================
# YOLO (PHONE)
# ===============================
phone_model = YOLO("yolov8n.pt")

# ===============================
# CONFIG
# ===============================
# Identity
FACE_THRESHOLD = 0.85            # stricter to reduce mis-ID
ATTENDANCE_COOLDOWN = 10         # seconds per student

# Phone
PHONE_CONF = 0.35
PHONE_COOLDOWN = 8               # global cooldown

# Talking (hardened)
MOUTH_ENTER = 0.42               # enter threshold (mouth open)
MOUTH_EXIT = 0.36                # exit threshold (hysteresis)
TALK_CONSEC_FRAMES = 8           # consecutive frames above ENTER
TALK_COOLDOWN = 10               # per student cooldown
TALK_BUFFER = 20                 # history buffer length
TALK_MIN_CYCLES = 2              # open→close cycles required
FPS_ASSUMPTION = 20              # used to interpret durations

# Side-looking (yaw/pitch)
YAW_THRESHOLD = 0.22
EXTREME_YAW = 0.32
PITCH_DOWN_THRESHOLD = 0.27
LOOK_DURATION = 1.6
YAW_SMOOTH_WINDOW = 7
MIRROR = True

# Unknown face alert
UNKNOWN_ALERT_COOLDOWN = 12

# ===============================
# STATE (PER STUDENT)
# ===============================
attendance_last = defaultdict(lambda: 0.0)

# Talking state
mouth_history = defaultdict(lambda: deque(maxlen=TALK_BUFFER))  # usn -> deque of ratios
talk_consec = defaultdict(int)                                  # usn -> consecutive frames above ENTER
talk_last_event = defaultdict(lambda: 0.0)                       # usn -> last violation time
talk_cycles = defaultdict(int)                                   # usn -> open→close cycles
mouth_prev_open = defaultdict(lambda: False)                     # usn -> previous open state

# Side-looking state
yaw_buffers = defaultdict(lambda: deque(maxlen=YAW_SMOOTH_WINDOW))
pitch_buffers = defaultdict(lambda: deque(maxlen=YAW_SMOOTH_WINDOW))
look_start_times = defaultdict(lambda: 0.0)
look_last_event = defaultdict(lambda: 0.0)

# Unknown alert
unknown_last_event = 0.0

# Phone state
phone_last_event = 0.0

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open camera")

print("🧭 Multi-face Attendance + Cheat detection started")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to read frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)

        present_faces = []  # {usn, name, bbox, lm, dist}
        unknown_present = False

        # ---------- FACE RECOGNITION FOR ALL FACES ----------
        for f in faces:
            emb = f.embedding / np.linalg.norm(f.embedding)
            dists = np.linalg.norm(np.array(known_embeddings) - emb, axis=1)
            idx = int(np.argmin(dists)) if len(dists) else -1

            if len(dists) == 0 or dists[idx] > FACE_THRESHOLD:
                usn = "UNKNOWN"
                name = "Unknown Person"
                unknown_present = True
            else:
                usn = known_usns[idx]
                name = usn_to_name.get(usn, "Unknown")

            x1, y1, x2, y2 = map(int, f.bbox)
            color = (0, 200, 0) if usn != "UNKNOWN" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({usn})" if usn != "UNKNOWN" else f"Unknown Dist:{dists[idx]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            present_faces.append({
                "usn": usn,
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "lm": f.landmark_2d_106 if hasattr(f, "landmark_2d_106") else None,
                "dist": dists[idx] if len(dists) else None
            })

        cv2.putText(frame, f"Faces: {len(present_faces)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        now_ts = time.time()
        now_dt = datetime.now()

        # ---------- Attendance per recognized student ----------
        for pf in present_faces:
            if pf["usn"] == "UNKNOWN":
                continue
            if now_ts - attendance_last[pf["usn"]] > ATTENDANCE_COOLDOWN:
                attendance_col.insert_one({
                    "usn": pf["usn"],
                    "name": pf["name"],
                    "date": now_dt.strftime("%Y-%m-%d"),
                    "time": now_dt.strftime("%H:%M:%S")
                })
                attendance_last[pf["usn"]] = now_ts
                print(f"✅ Attendance logged: {pf['usn']} {pf['name']}")

        # ---------- Unknown face alert (optional) ----------
        if unknown_present and (now_ts - unknown_last_event > UNKNOWN_ALERT_COOLDOWN):
            img_path = os.path.join(EVIDENCE_UNKNOWN, f"unknown_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(img_path, frame)
            violation_col.insert_one({
                "usn": "UNKNOWN",
                "name": "Unknown Person",
                "violation": "Unrecognized face detected",
                "faces_count": int(len(present_faces)),
                "evidence_path": img_path,
                "date": now_dt.strftime("%Y-%m-%d"),
                "time": now_dt.strftime("%H:%M:%S")
            })
            send_email("UNKNOWN", "Unknown Person", "Unrecognized face detected", img_path)
            unknown_last_event = now_ts

        # ---------- Side-looking per student ----------
        for pf in present_faces:
            if pf["usn"] == "UNKNOWN" or pf["lm"] is None:
                continue
            lm = pf["lm"]
            nose_x, nose_y = lm[49]
            left_eye_x, left_eye_y = lm[36]
            right_eye_x, right_eye_y = lm[45]
            chin_x, chin_y = lm[6]

            eye_center_x = (left_eye_x + right_eye_x) / 2.0
            eye_width = max(abs(right_eye_x - left_eye_x), 1.0)

            raw_yaw = (nose_x - eye_center_x) / eye_width
            raw_pitch = (chin_y - nose_y) / eye_width
            if MIRROR:
                raw_yaw = -raw_yaw

            u = pf["usn"]
            yaw_buffers[u].append(raw_yaw)
            pitch_buffers[u].append(raw_pitch)
            smooth_yaw = float(np.mean(yaw_buffers[u]))
            smooth_pitch = float(np.mean(pitch_buffers[u]))

            direction = "STRAIGHT"
            if smooth_yaw > YAW_THRESHOLD:
                direction = "LEFT"
            elif smooth_yaw < -YAW_THRESHOLD:
                direction = "RIGHT"
            elif smooth_pitch > PITCH_DOWN_THRESHOLD:
                direction = "DOWN"

            x1, y1, x2, y2 = pf["bbox"]
            cv2.putText(frame, f"Yaw:{smooth_yaw:.2f} Pitch:{smooth_pitch:.2f} {direction}",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            extreme = abs(smooth_yaw) > EXTREME_YAW
            if extreme:
                if look_start_times[u] == 0.0:
                    look_start_times[u] = now_ts
                elif (now_ts - look_start_times[u] > LOOK_DURATION) and (now_ts - look_last_event[u] > TALK_COOLDOWN):
                    img_path = os.path.join(EVIDENCE_SIDE, f"{u}_{direction}_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(img_path, frame)
                    violation_col.insert_one({
                        "usn": u,
                        "name": pf["name"],
                        "violation": f"Extreme Looking {direction}",
                        "yaw": round(smooth_yaw, 3),
                        "pitch": round(smooth_pitch, 3),
                        "evidence_path": img_path,
                        "date": now_dt.strftime("%Y-%m-%d"),
                        "time": now_dt.strftime("%H:%M:%S")
                    })
                    send_email(u, pf["name"], f"Extreme Looking {direction}", img_path,
                               extra=f"Yaw={round(smooth_yaw,3)}, Pitch={round(smooth_pitch,3)}")
                    look_last_event[u] = now_ts
                    look_start_times[u] = now_ts
            else:
                look_start_times[u] = 0.0

        # ---------- Hardened talking detection per student ----------
        for pf in present_faces:
            if pf["usn"] == "UNKNOWN" or pf["lm"] is None:
                continue
            u = pf["usn"]
            lm = pf["lm"]
            upper_lip = np.array(lm[62])
            lower_lip = np.array(lm[66])
            left_mouth = np.array(lm[60])
            right_mouth = np.array(lm[64])

            mouth_open = float(np.linalg.norm(upper_lip - lower_lip))
            mouth_width = float(max(np.linalg.norm(left_mouth - right_mouth), 1e-6))
            ratio = mouth_open / mouth_width

            # Hysteresis + consecutive frames
            mouth_history[u].append(ratio)
            is_open = ratio >= MOUTH_ENTER if mouth_prev_open[u] else ratio >= MOUTH_ENTER
            # Update cycles: count open->close transitions
            if mouth_prev_open[u] and ratio <= MOUTH_EXIT:
                talk_cycles[u] += 1
                mouth_prev_open[u] = False
            elif not mouth_prev_open[u] and ratio >= MOUTH_ENTER:
                mouth_prev_open[u] = True

            if ratio >= MOUTH_ENTER:
                talk_consec[u] += 1
            elif ratio <= MOUTH_EXIT:
                talk_consec[u] = 0  # reset when clearly closed

            x1, y1, x2, y2 = pf["bbox"]
            cv2.putText(frame, f"Mouth:{ratio:.2f} C:{talk_consec[u]} Cy:{talk_cycles[u]}",
                        (x1, y2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Trigger only if:
            # - enough consecutive open frames (speech-like)
            # - enough cycles (open->close transitions)
            # - cooldown respected
            if (talk_consec[u] >= TALK_CONSEC_FRAMES) and (talk_cycles[u] >= TALK_MIN_CYCLES) and (now_ts - talk_last_event[u] > TALK_COOLDOWN):
                img_path = os.path.join(EVIDENCE_TALK, f"{u}_TALK_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(img_path, frame)
                violation_col.insert_one({
                    "usn": u,
                    "name": pf["name"],
                    "violation": "Talking Detected",
                    "mouth_ratio": round(float(np.mean(mouth_history[u])), 2),
                    "cycles": int(talk_cycles[u]),
                    "consec_frames": int(talk_consec[u]),
                    "evidence_path": img_path,
                    "date": now_dt.strftime("%Y-%m-%d"),
                    "time": now_dt.strftime("%H:%M:%S")
                })
                send_email(u, pf["name"], "Talking Detected", img_path,
                           extra=f"Avg ratio={round(float(np.mean(mouth_history[u])),2)}, cycles={talk_cycles[u]}, consec={talk_consec[u]}")
                talk_last_event[u] = now_ts
                # Reset counters to avoid immediate retrigger
                talk_consec[u] = 0
                talk_cycles[u] = 0
                mouth_history[u].clear()
                mouth_prev_open[u] = False

        # ---------- PHONE DETECTION (global, email per face present) ----------
        results = phone_model(frame, verbose=False)
        phone_found = False
        phone_conf_used = 0.0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = phone_model.names.get(cls_id, str(cls_id))
                if label == "cell phone" and conf >= PHONE_CONF:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    phone_found = True
                    phone_conf_used = max(phone_conf_used, conf)

        if phone_found and (now_ts - phone_last_event > PHONE_COOLDOWN):
            img_path = os.path.join(EVIDENCE_PHONE, f"PHONE_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(img_path, frame)
            for pf in present_faces:
                violation_col.insert_one({
                    "usn": pf["usn"],
                    "name": pf["name"],
                    "violation": "Mobile Phone Detected",
                    "confidence": round(phone_conf_used, 2),
                    "evidence_path": img_path,
                    "date": now_dt.strftime("%Y-%m-%d"),
                    "time": now_dt.strftime("%H:%M:%S")
                })
                send_email(pf["usn"], pf["name"], "Mobile Phone Detected", img_path,
                           extra=f"Confidence={round(phone_conf_used,2)}")
            phone_last_event = now_ts
            print("🚨 PHONE CHEATING LOGGED for all present faces")

        cv2.imshow("Attendance + Cheat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped")
    print("✅ Cheat detection ended safely")