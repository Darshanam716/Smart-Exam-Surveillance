import time
import subprocess
import sys
from datetime import datetime

# ===============================
# CONFIG
# ===============================
ATTENDANCE_DURATION = 60  # seconds

print("\n🎓 SMART EXAM SURVEILLANCE SYSTEM")
print("====================================")
print("📅 Date:", datetime.now().strftime("%Y-%m-%d"))
print("⏰ Time:", datetime.now().strftime("%H:%M:%S"))
print("====================================\n")

def run_blocking(script_name):
    """
    Run script in blocking mode (waits until it exits)
    """
    print(f"\n▶ STARTING: {script_name}")
    subprocess.run([sys.executable, script_name])
    print(f"✔ COMPLETED: {script_name}")

# ===============================
# STEP 1: ATTENDANCE
# ===============================
print("🟢 STEP 1: ATTENDANCE (1 minute)")

attendance_proc = subprocess.Popen(
    [sys.executable, "face_attendance.py"]
)

time.sleep(ATTENDANCE_DURATION)

attendance_proc.terminate()
attendance_proc.wait()
print("⏹ Attendance phase completed safely")

# Small cooldown so camera is fully released
time.sleep(3)

# ===============================
# STEP 2: CHEAT DETECTION (SINGLE CAMERA OWNER)
# ===============================
print("\n🚨 STEP 2: CHEAT DETECTION MODE ACTIVATED")
print("📌 Phone + Talking + Looking + Multiple Faces")

try:
    # IMPORTANT:
    # This script must internally handle ALL detections
    run_blocking("cheat_engine.py")

except KeyboardInterrupt:
    print("\n🛑 Exam stopped by invigilator")

print("\n✅ Exam session ended safely")
