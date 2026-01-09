import os
import csv
from datetime import datetime
from pymongo import MongoClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===============================
# BASE PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ===============================
# DATABASE
# ===============================
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_surveillance"]
attendance_col = db["attendance"]
violation_col = db["violations"]

# ===============================
# FILE PATHS
# ===============================
ATT_CSV = os.path.join(REPORT_DIR, "attendance.csv")
VIO_CSV = os.path.join(REPORT_DIR, "violations.csv")
PDF_PATH = os.path.join(REPORT_DIR, "exam_report.pdf")

# ===============================
# EXPORT ATTENDANCE CSV
# ===============================
attendance = list(attendance_col.find({}, {"_id": 0}))

if attendance:
    att_keys = set()
    for row in attendance:
        att_keys.update(row.keys())
    with open(ATT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(att_keys))
        writer.writeheader()
        for row in attendance:
            writer.writerow(row)

print("✅ attendance.csv created")

# ===============================
# EXPORT VIOLATIONS CSV
# ===============================
violations = list(violation_col.find({}, {"_id": 0}))

if violations:
    vio_keys = set()
    for v in violations:
        vio_keys.update(v.keys())
    with open(VIO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(vio_keys))
        writer.writeheader()
        for v in violations:
            writer.writerow(v)

print("✅ violations.csv created")

# ===============================
# GENERATE PDF REPORT (structured)
# ===============================
pdf = canvas.Canvas(PDF_PATH, pagesize=A4)
width, height = A4
y = height - 40

pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(40, y, "SMART EXAM SURVEILLANCE – FINAL REPORT")
y -= 30

pdf.setFont("Helvetica", 11)
pdf.drawString(40, y, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
y -= 30

# ---------- ATTENDANCE ----------
pdf.setFont("Helvetica-Bold", 13)
pdf.drawString(40, y, "Attendance Summary")
y -= 20

# Table headers
pdf.setFont("Helvetica-Bold", 10)
pdf.drawString(50, y, "USN")
pdf.drawString(150, y, "Date")
pdf.drawString(250, y, "Time")
y -= 15

pdf.setFont("Helvetica", 10)
for a in attendance:
    pdf.drawString(50, y, str(a.get("usn", "")))
    pdf.drawString(150, y, str(a.get("date", "")))
    pdf.drawString(250, y, str(a.get("time", "")))
    y -= 15
    if y < 50:
        pdf.showPage()
        y = height - 40

# ---------- VIOLATIONS ----------
pdf.showPage()
y = height - 40

pdf.setFont("Helvetica-Bold", 13)
pdf.drawString(40, y, "Cheating / Violation Summary")
y -= 20

# Table headers
pdf.setFont("Helvetica-Bold", 10)
pdf.drawString(50, y, "USN")
pdf.drawString(150, y, "Violation")
pdf.drawString(350, y, "Time")
y -= 15

pdf.setFont("Helvetica", 10)
for v in violations:
    pdf.drawString(50, y, str(v.get("usn", "")))
    pdf.drawString(150, y, str(v.get("violation", "")))
    pdf.drawString(350, y, str(v.get("time", "")))
    y -= 15
    if y < 50:
        pdf.showPage()
        y = height - 40

pdf.save()
print("✅ exam_report.pdf created")

print("\n📂 REPORTS SAVED IN:", REPORT_DIR)
