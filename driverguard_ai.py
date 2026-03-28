# =============================================================================
#  DriverGuard AI — Smart Driver Safety System
#  Author  : Senior Python & Computer Vision Developer
#  Version : 2.0 PRO
#
#  UNIQUE FEATURES:
#  ✅ Named Driver Alerts ("Rajesh, WAKE UP!")
#  ✅ Live EAR Graph on Dashboard
#  ✅ Fatigue Score (0–100) with Accident Risk Level
#  ✅ Auto Screenshot on Drowsiness
#  ✅ Email Alert to Family with Screenshot Attached
#  ✅ Session Stats (total alerts, time driven, drowsy episodes)
#  ✅ HUD-style Dashboard overlay on webcam feed
#
#  SETUP:
#  pip install opencv-python dlib numpy scipy imutils face_recognition
#
#  EMAIL SETUP:
#  - Set SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL below
#  - For Gmail: enable "App Passwords" in Google Account settings
#    (Google Account → Security → 2-Step Verification → App Passwords)
#
#  DRIVER REGISTRATION:
#  - Run with --register flag to add a new driver face:
#    python driverguard_ai.py --register "YourName"
#  - Then run normally: python driverguard_ai.py
# =============================================================================

import cv2
import dlib
import numpy as np
import winsound
import time
import os
import sys
import smtplib
import pickle
import argparse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# EMAIL CONFIGURATION — Fill these in!
# ─────────────────────────────────────────────────────────────────────────────
SENDER_EMAIL    = "your_email@gmail.com"       # Gmail sender
SENDER_PASSWORD = "your_app_password_here"     # Gmail App Password (not your login password)
RECEIVER_EMAIL  = "family_email@gmail.com"     # Who gets the alert

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LANDMARK_MODEL    = "shape_predictor_68_face_landmarks.dat"
DRIVER_DB_FILE    = "drivers.pkl"            # Saved driver face encodings
SCREENSHOTS_DIR   = "drowsy_screenshots"     # Folder for auto screenshots

EAR_THRESHOLD     = 0.25                     # Below this = eyes closed
CONSEC_FRAMES     = 20                       # Frames before alarm triggers
EAR_HISTORY_LEN   = 60                       # Points on live EAR graph

# Alarm
ALARM_FREQ        = 1000
ALARM_DURATION    = 600

# Fatigue scoring weights
FATIGUE_DECAY     = 0.995    # Score slowly decays when alert
FATIGUE_INCREASE  = 3.5      # Score rises fast when drowsy

# Risk thresholds
RISK_LOW      = 30
RISK_MEDIUM   = 55
RISK_HIGH     = 75
RISK_CRITICAL = 90

# Colors (BGR)
GREEN    = (0, 230, 100)
RED      = (0, 60, 230)
ORANGE   = (0, 140, 255)
YELLOW   = (0, 220, 220)
CYAN     = (230, 200, 0)
WHITE    = (255, 255, 255)
BLACK    = (0, 0, 0)
DARK_BG  = (18, 18, 24)
GRAY     = (120, 120, 130)

# Eye landmark indices
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def load_drivers():
    """Load saved driver face encodings from disk."""
    if os.path.exists(DRIVER_DB_FILE):
        with open(DRIVER_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_drivers(drivers):
    with open(DRIVER_DB_FILE, "wb") as f:
        pickle.dump(drivers, f)


def register_driver(name):
    """Capture face from webcam and save encoding for named alerts."""
    try:
        import face_recognition
    except ImportError:
        print("[ERROR] Run: pip install face_recognition")
        return

    print(f"[INFO] Registering driver: {name}")
    print("[INFO] Look at the camera. Press SPACE to capture, Q to quit.")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, f"Registering: {name}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, GREEN, 2)
        cv2.putText(frame, "Press SPACE to capture", (20, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 1)
        cv2.imshow("Driver Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                drivers = load_drivers()
                drivers[name] = encodings[0]
                save_drivers(drivers)
                print(f"[SUCCESS] Driver '{name}' registered!")
                break
            else:
                print("[WARNING] No face detected. Try again.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()


def recognize_driver(frame, drivers):
    """Return driver name if face matches a registered driver."""
    if not drivers:
        return "Unknown Driver"
    try:
        import face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return "Unknown Driver"
        encs = face_recognition.face_encodings(rgb, locs)
        for enc in encs:
            for name, known_enc in drivers.items():
                match = face_recognition.compare_faces([known_enc], enc, tolerance=0.5)
                if match[0]:
                    return name
    except Exception:
        pass
    return "Unknown Driver"


def save_screenshot(frame, driver_name, fatigue_score):
    """Save a screenshot when drowsiness is detected."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOTS_DIR, f"drowsy_{driver_name}_{ts}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def send_email_alert(driver_name, fatigue_score, risk_level, screenshot_path):
    """Send email alert with screenshot to family."""
    if SENDER_EMAIL == "your_email@gmail.com":
        print("[INFO] Email not configured — skipping email alert.")
        return

    try:
        ts = datetime.now().strftime("%d %b %Y at %I:%M %p")
        subject = f"🚨 DROWSINESS ALERT — {driver_name} needs attention!"

        body = f"""
⚠️ DRIVER SAFETY ALERT — DriverGuard AI

Driver Name  : {driver_name}
Time         : {ts}
Fatigue Score: {fatigue_score:.0f} / 100
Risk Level   : {risk_level}

The system detected prolonged eye closure indicating drowsiness.
Please check on the driver immediately.

Auto-screenshot is attached.

— DriverGuard AI Safety System
        """

        msg = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(screenshot_path))
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"[EMAIL] Alert sent to {RECEIVER_EMAIL}")

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def get_risk_info(score):
    """Return risk label and color based on fatigue score."""
    if score < RISK_LOW:
        return "SAFE", GREEN
    elif score < RISK_MEDIUM:
        return "CAUTION", YELLOW
    elif score < RISK_HIGH:
        return "WARNING", ORANGE
    elif score < RISK_CRITICAL:
        return "HIGH RISK", RED
    else:
        return "CRITICAL", (0, 0, 255)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10, fill=False, alpha=0.6):
    """Draw a filled semi-transparent rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    if fill:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.rectangle(img, pt1, pt2, color, thickness)


def draw_ear_graph(frame, ear_history, x, y, w, h):
    """Draw a live EAR waveform graph."""
    # Background
    draw_rounded_rect(frame, (x, y), (x+w, y+h), (10, 10, 20), -1, fill=True, alpha=0.75)
    cv2.rectangle(frame, (x, y), (x+w, y+h), GRAY, 1)

    # Label
    cv2.putText(frame, "EAR Live Graph", (x+8, y+16),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, CYAN, 1)

    # Threshold line
    thresh_y = int(y + h - (EAR_THRESHOLD / 0.45) * (h - 24) - 4)
    cv2.line(frame, (x+4, thresh_y), (x+w-4, thresh_y), RED, 1)
    cv2.putText(frame, f"Threshold {EAR_THRESHOLD}", (x+w-90, thresh_y - 3),
                cv2.FONT_HERSHEY_DUPLEX, 0.3, RED, 1)

    # Plot EAR history
    if len(ear_history) > 1:
        pts = []
        for i, val in enumerate(ear_history):
            px = int(x + 4 + i * (w - 8) / EAR_HISTORY_LEN)
            py = int(y + h - (val / 0.45) * (h - 24) - 4)
            py = max(y + 4, min(y + h - 4, py))
            pts.append((px, py))
        for i in range(1, len(pts)):
            color = GREEN if ear_history[i] >= EAR_THRESHOLD else RED
            cv2.line(frame, pts[i-1], pts[i], color, 1)


def draw_fatigue_bar(frame, score, x, y, w, h):
    """Draw an animated fatigue score bar."""
    draw_rounded_rect(frame, (x, y), (x+w, y+h), (10, 10, 20), -1, fill=True, alpha=0.75)
    cv2.rectangle(frame, (x, y), (x+w, y+h), GRAY, 1)
    cv2.putText(frame, "Fatigue Score", (x+8, y+15),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, CYAN, 1)

    bar_x = x + 6
    bar_y = y + 22
    bar_w = w - 12
    bar_h = 12

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40, 40, 50), -1)

    # Filled portion
    fill_w = int(bar_w * score / 100)
    _, bar_color = get_risk_info(score)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), bar_color, -1)

    # Score text
    cv2.putText(frame, f"{score:.0f}/100", (bar_x + bar_w//2 - 20, bar_y + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, WHITE, 1)


def draw_hud(frame, data):
    """
    Draw the full HUD dashboard overlay.
    data keys: ear, fatigue, risk_label, risk_color, driver_name,
               alert_count, session_time, drowsy_count, alert_active, eye_status
    """
    h, w = frame.shape[:2]
    ear          = data["ear"]
    fatigue      = data["fatigue"]
    risk_label   = data["risk_label"]
    risk_color   = data["risk_color"]
    driver_name  = data["driver_name"]
    alert_count  = data["alert_count"]
    session_time = data["session_time"]
    drowsy_count = data["drowsy_count"]
    alert_active = data["alert_active"]
    eye_status   = data["eye_status"]
    ear_history  = data["ear_history"]

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    draw_rounded_rect(frame, (0, 0), (w, 42), (10, 10, 22), -1, fill=True, alpha=0.82)

    # Brand
    cv2.putText(frame, "DriverGuard", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, CYAN, 2)
    cv2.putText(frame, "AI", (138, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, GREEN, 2)

    # Driver name (center)
    name_text = f"Driver: {driver_name}"
    cv2.putText(frame, name_text, (w//2 - 90, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, WHITE, 1)

    # Time (right)
    clock = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, clock, (w - 100, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, GRAY, 1)

    # ── LEFT PANEL ───────────────────────────────────────────────────────────
    panel_x, panel_y = 8, 50
    panel_w, panel_h = 180, 200
    draw_rounded_rect(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h),
                      (10, 10, 22), -1, fill=True, alpha=0.78)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), GRAY, 1)

    # EAR value
    cv2.putText(frame, "EAR VALUE", (panel_x+8, panel_y+20),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, CYAN, 1)
    ear_color = GREEN if ear >= EAR_THRESHOLD else RED
    cv2.putText(frame, f"{ear:.3f}", (panel_x+8, panel_y+50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, ear_color, 2)

    # Eye status
    cv2.putText(frame, eye_status, (panel_x+8, panel_y+75),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, ear_color, 1)

    # Divider
    cv2.line(frame, (panel_x+8, panel_y+85), (panel_x+panel_w-8, panel_y+85), GRAY, 1)

    # Stats
    mins = int(session_time // 60)
    secs = int(session_time % 60)
    stats = [
        ("Session",    f"{mins:02d}:{secs:02d}"),
        ("Alerts",     str(alert_count)),
        ("Episodes",   f"{drowsy_count}/3→Email"),
        ("Risk",       risk_label),
    ]
    for i, (label, value) in enumerate(stats):
        ly = panel_y + 105 + i * 24
        cv2.putText(frame, label, (panel_x+8, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, GRAY, 1)
        val_color = risk_color if label == "Risk" else WHITE
        cv2.putText(frame, value, (panel_x+80, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, val_color, 1)

    # ── EAR GRAPH (bottom left) ───────────────────────────────────────────────
    draw_ear_graph(frame, ear_history, 8, h - 100, 220, 88)

    # ── FATIGUE BAR (bottom right) ────────────────────────────────────────────
    draw_fatigue_bar(frame, fatigue, w - 220, h - 60, 212, 48)

    # Risk badge (bottom right top)
    draw_rounded_rect(frame, (w-220, h-100), (w-8, h-68),
                      risk_color, -1, fill=True, alpha=0.75)
    cv2.putText(frame, f"ACCIDENT RISK: {risk_label}",
                (w-215, h-77),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, WHITE, 1)

    # ── ALERT BANNER ─────────────────────────────────────────────────────────
    if alert_active:
        # Full-width flashing red banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        alert_text = f"  {driver_name.upper()}, WAKE UP!  "
        cv2.putText(frame, alert_text,
                    (w//2 - len(alert_text)*9, h//2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, "!!! DROWSINESS DETECTED !!!",
                    (w//2 - 175, h//2 - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, YELLOW, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DriverGuard AI")
    parser.add_argument("--register", type=str, help="Register a new driver by name")
    args = parser.parse_args()

    # Registration mode
    if args.register:
        register_driver(args.register)
        return

    print("=" * 60)
    print("       DriverGuard AI — Smart Driver Safety System")
    print("=" * 60)
    print("[INFO] Loading models...")

    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(LANDMARK_MODEL)
    except RuntimeError:
        print(f"\n[ERROR] '{LANDMARK_MODEL}' not found!")
        print("Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    drivers = load_drivers()
    if drivers:
        print(f"[INFO] Registered drivers: {', '.join(drivers.keys())}")
    else:
        print("[INFO] No registered drivers. Run with --register 'YourName' to add one.")

    print("[INFO] Starting webcam...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # State
    closed_frame_count = 0
    alert_active       = False
    fatigue_score      = 0.0
    alert_count        = 0
    drowsy_count       = 0      # counts distinct drowsy episodes
    session_start      = time.time()
    last_email_time    = 0
    ear_history        = deque([0.3] * EAR_HISTORY_LEN, maxlen=EAR_HISTORY_LEN)
    driver_name        = "Unknown Driver"
    last_recognition   = 0
    recognition_interval = 3.0  # Recognize face every 3 seconds (performance)

    # ── Alert episode logic ───────────────────────────────────────────────────
    # Episode 1 → beep only
    # Episode 2 → beep only
    # Episode 3+ → beep + screenshot + email
    episode_in_progress = False   # True while eyes are currently closed (1 episode)

    print("[INFO] System running. Press Q to quit.\n")

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (800, 560))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face recognition (throttled)
        now = time.time()
        if drivers and (now - last_recognition) > recognition_interval:
            driver_name = recognize_driver(frame, drivers)
            last_recognition = now

        # Dlib face detection
        rects = detector(gray, 0)

        ear        = 0.0
        eye_status = "NO FACE"

        for rect in rects:
            shape    = predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)

            left_eye  = shape_np[L_START:L_END]
            right_eye = shape_np[R_START:R_END]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear       = (left_ear + right_ear) / 2.0

            # Eye contours
            for eye in (left_eye, right_eye):
                hull = cv2.convexHull(eye)
                cv2.drawContours(frame, [hull], -1, GREEN, 1)

            # Drowsiness logic
            if ear < EAR_THRESHOLD:
                closed_frame_count += 1
                eye_status    = "EYES CLOSED"
                fatigue_score = min(100, fatigue_score + FATIGUE_INCREASE)

                # Threshold crossed → new episode starts
                if closed_frame_count == CONSEC_FRAMES:
                    drowsy_count       += 1
                    alert_active        = True
                    alert_count        += 1
                    episode_in_progress = True

                    risk_label, _ = get_risk_info(fatigue_score)

                    if drowsy_count == 1:
                        # ── Episode 1: beep only ──────────────────────────
                        winsound.Beep(ALARM_FREQ, ALARM_DURATION)
                        print(f"[ALERT 1] Episode 1 — Beep only. Stay alert, {driver_name}!")

                    elif drowsy_count == 2:
                        # ── Episode 2: double beep, louder warning ────────
                        winsound.Beep(ALARM_FREQ, ALARM_DURATION)
                        time.sleep(0.15)
                        winsound.Beep(ALARM_FREQ + 200, ALARM_DURATION)
                        print(f"[ALERT 2] Episode 2 — Double beep. Warning, {driver_name}!")

                    else:
                        # ── Episode 3+: triple beep + screenshot + email ──
                        for _ in range(3):
                            winsound.Beep(ALARM_FREQ + 400, ALARM_DURATION)
                            time.sleep(0.1)

                        screenshot_path = save_screenshot(frame, driver_name, fatigue_score)
                        print(f"[ALERT 3] Episode {drowsy_count} — Screenshot saved: {screenshot_path}")

                        if (now - last_email_time) > 60:
                            send_email_alert(driver_name, fatigue_score,
                                             risk_label, screenshot_path)
                            last_email_time = now

                elif closed_frame_count > CONSEC_FRAMES:
                    # Keep beeping every ~30 frames while still drowsy
                    if closed_frame_count % 30 == 0:
                        winsound.Beep(ALARM_FREQ, 300)

            else:
                closed_frame_count  = 0
                alert_active        = False
                episode_in_progress = False
                eye_status          = "EYES OPEN"
                fatigue_score       = max(0, fatigue_score * FATIGUE_DECAY)

        ear_history.append(ear)
        risk_label, risk_color = get_risk_info(fatigue_score)
        session_time = time.time() - session_start

        # Draw HUD
        draw_hud(frame, {
            "ear":          ear,
            "fatigue":      fatigue_score,
            "risk_label":   risk_label,
            "risk_color":   risk_color,
            "driver_name":  driver_name,
            "alert_count":  alert_count,
            "session_time": session_time,
            "drowsy_count": drowsy_count,
            "alert_active": alert_active,
            "eye_status":   eye_status,
            "ear_history":  ear_history,
        })

        cv2.imshow("DriverGuard AI — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    # Session summary
    print("\n" + "="*50)
    print("         SESSION SUMMARY")
    print("="*50)
    print(f"  Driver       : {driver_name}")
    print(f"  Session Time : {int(session_time//60):02d}:{int(session_time%60):02d}")
    print(f"  Total Alerts : {alert_count}")
    print(f"  Drowsy Episodes: {drowsy_count}")
    print(f"  Final Fatigue: {fatigue_score:.1f}/100")
    print("="*50)

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
