import cv2
import os

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

os.makedirs("evidence_test", exist_ok=True)
path = "evidence_test/test.jpg"

saved = cv2.imwrite(path, frame)
print("Saved:", saved)
print("Path exists:", os.path.exists(path))
