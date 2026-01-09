import cv2
import os

usn = input("Enter Student USN: ")

dataset_path = "dataset"
student_path = os.path.join(dataset_path, usn)

if not os.path.exists(student_path):
    os.makedirs(student_path)

cap = cv2.VideoCapture(0)

count = 0
print("📸 Capturing face images... Look at the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    file_name = f"{student_path}/{count}.jpg"
    cv2.imwrite(file_name, frame)

    cv2.putText(
        frame,
        f"Captured: {count}/20",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(200) & 0xFF == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Face registration completed for {usn}")
