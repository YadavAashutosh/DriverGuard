# =============================================================================
#  DROWSINESS DETECTION SYSTEM
#  Author  : Senior Python & Computer Vision Developer
#  Stack   : OpenCV · Dlib · SciPy · imutils · winsound (Windows)
#  Purpose : Real-time driver/user drowsiness alert via webcam
# =============================================================================
#
#  SETUP CHECKLIST
#  ---------------
#  1. pip install opencv-python dlib numpy scipy imutils cmake
#     (if dlib fails, grab a pre-built wheel from:
#      https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
#
#  2. Download the 68-landmark model:
#     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#     Extract it and place  shape_predictor_68_face_landmarks.dat
#     in the SAME folder as this script.
#
#  3. Run:  python drowsiness_detection.py
#           Press  Q  to quit.
# =============================================================================

import cv2
import dlib
import numpy as np
import winsound                         # Windows-only; swap for playsound/beepy on Linux/macOS
import time

from scipy.spatial   import distance as dist
from imutils         import face_utils
from imutils.video   import VideoStream

# ---------------------------------------------------------------------------
# CONFIGURATION  –  tweak these constants to suit your lighting / camera
# ---------------------------------------------------------------------------

# Path to Dlib's pre-trained 68-point facial landmark predictor
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

# Eye Aspect Ratio threshold:
#   EAR drops below this value when eyes are closed.
#   Typical open-eye EAR ≈ 0.25–0.30; adjust if getting false positives.
EAR_THRESHOLD = 0.25

# How many *consecutive* frames the EAR must stay below EAR_THRESHOLD
# before the alarm fires.  At 30 fps → 20 frames ≈ 0.67 s, 30 frames ≈ 1 s.
CONSEC_FRAMES = 25

# Alarm beep parameters (winsound.Beep)
ALARM_FREQ_HZ  = 1000   # frequency in Hz  (440 = concert A, 1000 = sharp alert)
ALARM_DURATION = 500    # duration in ms per beep burst

# Dlib face-index ranges for each eye (standard 68-point model)
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    Formula (Soukupová & Čech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)

    Parameters
    ----------
    eye : np.ndarray, shape (6, 2)
        Six (x, y) landmark coordinates for one eye.

    Returns
    -------
    float
        EAR value.  ~0.25-0.30 → open eye;  <0.20 → closed eye.
    """
    # Vertical distances (two pairs)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


def draw_eye_contours(frame, left_eye, right_eye) -> None:
    """
    Draw green convex-hull contours around both eyes on *frame* (in-place).
    """
    for eye in (left_eye, right_eye):
        hull = cv2.convexHull(eye)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)


def overlay_status(frame, ear: float, alert: bool, frame_count: int) -> None:
    """
    Burn the EAR value, frame counter, and alert banner onto *frame* (in-place).
    """
    h, w = frame.shape[:2]

    # ── EAR readout (top-left) ──────────────────────────────────────────────
    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.75,
        (0, 255, 255),           # cyan
        2, cv2.LINE_AA
    )

    # ── Consecutive closed-frame counter ───────────────────────────────────
    cv2.putText(
        frame,
        f"Closed Frames: {frame_count}/{CONSEC_FRAMES}",
        (10, 60),
        cv2.FONT_HERSHEY_DUPLEX, 0.6,
        (200, 200, 200),
        1, cv2.LINE_AA
    )

    # ── Drowsiness alert banner ─────────────────────────────────────────────
    if alert:
        # Filled semi-transparent red rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame,
            "!  DROWSINESS ALERT  !",
            (w // 2 - 185, h - 28),
            cv2.FONT_HERSHEY_DUPLEX, 0.95,
            (255, 255, 255),
            2, cv2.LINE_AA
        )


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading Dlib face detector and landmark predictor …")

    # Dlib HOG-based frontal face detector (CPU-friendly)
    detector  = dlib.get_frontal_face_detector()

    # 68-point shape predictor – needs the .dat file in the same folder
    try:
        predictor = dlib.shape_predictor(LANDMARK_MODEL)
    except RuntimeError:
        print(
            f"\n[ERROR] Cannot load '{LANDMARK_MODEL}'.\n"
            "  → Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "  → Extract and place the .dat file in the same folder as this script.\n"
        )
        return

    # ── Open webcam ─────────────────────────────────────────────────────────
    print("[INFO] Starting webcam stream …")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)   # let the sensor warm up

    # State variables
    closed_frame_count = 0     # consecutive frames where EAR < threshold
    alarm_on           = False # whether the alarm is currently sounding

    print("[INFO] Monitoring started.  Press  Q  in the video window to quit.\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        frame = vs.read()
        if frame is None:
            print("[WARNING] No frame received from camera.")
            break

        # Resize for consistent processing speed (width=640 keeps quality fine)
        frame = cv2.resize(frame, (640, 480))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces in the grayscale frame
        rects = detector(gray, 0)

        ear    = 0.0
        alert  = False

        for rect in rects:
            # Predict 68 landmarks for this face
            shape      = predictor(gray, rect)
            shape_np   = face_utils.shape_to_np(shape)

            # Extract left & right eye coordinate arrays
            left_eye  = shape_np[L_START:L_END]
            right_eye = shape_np[R_START:R_END]

            # Compute EAR for each eye, then average them
            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear       = (left_ear + right_ear) / 2.0

            # Visualise eye contours on the frame
            draw_eye_contours(frame, left_eye, right_eye)

            # ── Drowsiness logic ────────────────────────────────────────────
            if ear < EAR_THRESHOLD:
                closed_frame_count += 1

                if closed_frame_count >= CONSEC_FRAMES:
                    alert = True
                    # Play a non-blocking beep on Windows
                    # (winsound.Beep is synchronous, so we keep it short)
                    if not alarm_on:
                        alarm_on = True
                    # Fire the beep every frame once triggered
                    # Using MessageBeep is near-instant; Beep() blocks briefly
                    winsound.Beep(ALARM_FREQ_HZ, ALARM_DURATION)

            else:
                # Eyes are open → reset counter and silence alarm
                closed_frame_count = 0
                alarm_on           = False

        # ── HUD overlay ─────────────────────────────────────────────────────
        overlay_status(frame, ear, alert, closed_frame_count)

        # ── Show frame ──────────────────────────────────────────────────────
        cv2.imshow("Drowsiness Detection  |  Press Q to quit", frame)

        # Quit on 'q' or 'Q'
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("[INFO] Quit signal received.")
            break

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print("[INFO] Releasing resources …")
    cv2.destroyAllWindows()
    vs.stop()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()