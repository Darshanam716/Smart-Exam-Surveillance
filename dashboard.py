import os
import io
import csv
import cv2
import time
from datetime import datetime
from flask import (
    Flask, render_template, Response, request, redirect,
    url_for, session, send_from_directory, flash
)
from flask_pymongo import PyMongo
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- App setup ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret")

# MongoDB
app.config["MONGO_URI"] = "mongodb://localhost:27017/exam_surveillance"
mongo = PyMongo(app)

# Camera (single global capture)
camera = cv2.VideoCapture(0)

# Reports folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def is_logged_in():
    return session.get("principal") is True

def export_attendance_csv(path):
    attendance = list(mongo.db.attendance.find({}, {"_id": 0}))
    if not attendance:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["usn", "name", "date", "time"])
        return
    keys = set()
    for r in attendance:
        keys.update(r.keys())
    keys = list(keys)
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in attendance:
            writer.writerow(r)

def export_violations_csv(path):
    violations = list(mongo.db.violations.find({}, {"_id": 0}))
    if not violations:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["usn", "violation", "date", "time"])
        return
    keys = set()
    for v in violations:
        keys.update(v.keys())
    keys = list(keys)
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for v in violations:
            writer.writerow(v)

def export_report_pdf(path):
    attendance = list(mongo.db.attendance.find({}, {"_id": 0}))
    violations = list(mongo.db.violations.find({}, {"_id": 0}))
    pdf = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 40
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "SMART EXAM SURVEILLANCE – REPORT")
    y -= 30
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 25

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Attendance")
    y -= 18
    pdf.setFont("Helvetica", 9)
    for a in attendance:
        line = f"{a.get('usn','')}  |  {a.get('name','')}  |  {a.get('time','')}"
        pdf.drawString(50, y, line)
        y -= 14
        if y < 60:
            pdf.showPage()
            y = h - 40

    pdf.showPage()
    y = h - 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Violations")
    y -= 18
    pdf.setFont("Helvetica", 9)
    for v in violations:
        line = f"{v.get('usn','')}  |  {v.get('violation','')}  |  {v.get('time','')}"
        pdf.drawString(50, y, line)
        y -= 14
        if y < 60:
            pdf.showPage()
            y = h - 40
    pdf.save()

# ---------------- Routes ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    ADMIN_USER = os.environ.get("PRINCIPAL_USER", "principal")
    ADMIN_PASS = os.environ.get("PRINCIPAL_PASS", "admin123")
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["principal"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("principal", None)
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    attendance = list(mongo.db.attendance.find({}, {"_id": 0}).sort("time", -1))
    violations = list(mongo.db.violations.find({}, {"_id": 0}).sort("time", -1))
    total_attendance = len(attendance)
    total_violations = len(violations)
    recent_violations = violations[:6]
    return render_template(
        "dashboard.html",
        attendance=attendance,
        violations=violations,
        total_attendance=total_attendance,
        total_violations=total_violations,
        recent_violations=recent_violations,
        username=session.get("username", "Principal")
    )

@app.route("/video")
def video():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/reports")
def reports():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("reports.html")

@app.route("/reports/download")
def reports_download():
    if not is_logged_in():
        return redirect(url_for("login"))
    typ = request.args.get("type", "all")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = []
    if typ in ("attendance", "all"):
        att_path = os.path.join(REPORT_DIR, f"attendance_{ts}.csv")
        export_attendance_csv(att_path)
        files.append(("attendance", os.path.basename(att_path)))
    if typ in ("violations", "all"):
        vio_path = os.path.join(REPORT_DIR, f"violations_{ts}.csv")
        export_violations_csv(vio_path)
        files.append(("violations", os.path.basename(vio_path)))
    if typ in ("pdf", "all"):
        pdf_path = os.path.join(REPORT_DIR, f"report_{ts}.pdf")
        export_report_pdf(pdf_path)
        files.append(("pdf", os.path.basename(pdf_path)))
    if len(files) == 1:
        return send_from_directory(REPORT_DIR, files[0][1], as_attachment=True)
    return render_template("reports.html", files=files)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if not is_logged_in():
        return redirect(url_for("login"))
    if request.method == "POST":
        yaw = float(request.form.get("yaw_threshold", 0.30))
        mouth = float(request.form.get("mouth_threshold", 0.35))
        cooldown = int(request.form.get("cooldown", 6))
        mongo.db.settings.update_one({"_id": "global"}, {"$set": {
            "yaw_threshold": yaw,
            "mouth_threshold": mouth,
            "cooldown": cooldown
        }}, upsert=True)
        flash("Settings saved", "success")
        return redirect(url_for("settings"))
    s = mongo.db.settings.find_one({"_id": "global"}) or {}
    return render_template("settings.html", settings=s)

@app.route("/reports/files/<filename>")
def reports_file(filename):
    if not is_logged_in():
        return redirect(url_for("login"))
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

# ---------------- Shutdown camera on exit ----------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        try:
            camera.release()
        except Exception:
            pass
