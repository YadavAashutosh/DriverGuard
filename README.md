# 🚗 DriverGuard-AI
### Real-Time Driver Drowsiness Detection & Alert System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dlib-20.0-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey?style=for-the-badge&logo=windows"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>An AI-powered driver safety system that detects drowsiness in real-time using computer vision, triggers progressive alerts, and automatically notifies emergency contacts via email — before an accident happens.</b>
</p>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Alert Logic](#-alert-logic)
- [Dashboard Preview](#-dashboard-preview)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 About the Project

Every year, thousands of road accidents occur due to driver fatigue and drowsiness. **DriverGuard-AI** is a real-time computer vision system designed to prevent such accidents by continuously monitoring the driver's eye activity and triggering intelligent, progressive alerts.

Unlike generic drowsiness detectors, DriverGuard-AI features:
- **Named alerts** — calls the driver by their registered name
- **Progressive warning system** — escalates from a gentle beep to an emergency email
- **Live HUD dashboard** — shows EAR graph, fatigue score, and accident risk level
- **Face recognition** — identifies registered drivers automatically

> Built as a college final-year project demonstrating real-world application of Computer Vision, Machine Learning, and IoT-style alerting.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👁️ **Eye Aspect Ratio (EAR)** | Detects eye closure using 68-point facial landmarks |
| 🧑‍💼 **Face Recognition** | Identifies driver and uses their name in alerts |
| 🔔 **Progressive Alerts** | 3-stage alert system (beep → double beep → emergency) |
| 📸 **Auto Screenshot** | Captures and saves frame on 3rd drowsy episode |
| 📧 **Email Alert** | Sends emergency email with screenshot to family |
| 📊 **Live HUD Dashboard** | Real-time EAR graph, fatigue score, risk level |
| ⚠️ **Accident Risk Score** | Calculates risk: SAFE → CAUTION → WARNING → CRITICAL |
| 🕐 **Session Stats** | Tracks session time, alert count, drowsy episodes |

---

## ⚙️ How It Works

```
Webcam Feed
    │
    ▼
Grayscale Conversion
    │
    ▼
Dlib Face Detection (HOG + SVM)
    │
    ▼
68-Point Facial Landmark Detection
    │
    ▼
Eye Aspect Ratio (EAR) Calculation
  EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
    │
    ├── EAR ≥ 0.25 → Eyes Open → Fatigue decays
    │
    └── EAR < 0.25 for 20+ frames → Drowsy Episode
              │
              ├── Episode 1 → 🔔 Single Beep
              ├── Episode 2 → 🔔🔔 Double Beep
              └── Episode 3 → 🔔🔔🔔 Triple Beep + 📸 Screenshot + 📧 Email
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **OpenCV** | Video capture, frame processing, HUD rendering |
| **Dlib** | Face detection, 68-point landmark prediction |
| **face_recognition** | Driver identity recognition |
| **SciPy** | Euclidean distance for EAR calculation |
| **imutils** | Video stream utilities, landmark helpers |
| **smtplib** | Email alert delivery |
| **NumPy** | Array operations |

---

## 📁 Project Structure

```
DriverGuard-AI/
│
├── driverguard_ai.py               # Main application
├── shape_predictor_68_face_landmarks.dat  # Dlib landmark model (download separately)
├── drivers.pkl                     # Registered driver face encodings (auto-generated)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── drowsy_screenshots/             # Auto-created on first alert
    └── drowsy_DriverName_YYYYMMDD_HHMMSS.jpg
```

---

## 🚀 Installation

### Prerequisites
- Windows 10/11
- Python 3.9 – 3.12 (recommended: 3.11)
- Webcam
- Visual Studio Build Tools with **Desktop development with C++**
  → Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/DriverGuard-AI.git
cd DriverGuard-AI
```

### Step 2 — Install Dependencies
```bash
pip install cmake
pip install dlib
pip install opencv-python numpy scipy imutils face_recognition
```

> ⚠️ **dlib requires Visual Studio C++ Build Tools.** If build fails, see [Troubleshooting](#troubleshooting).

### Step 3 — Download Landmark Model
Download the 68-point facial landmark model:

**Link:** http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place `shape_predictor_68_face_landmarks.dat` in the project root folder.

### Step 4 — Install Requirements
```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Register Your Face (First Time)
```bash
python driverguard_ai.py --register "YourName"
```
- Camera will open
- Look straight at the camera
- Press **SPACE** to capture your face
- Press **Q** to quit

### Run the System
```bash
python driverguard_ai.py
```

### Controls
| Key | Action |
|---|---|
| `Q` | Quit and show session summary |

---

## 🔧 Configuration

Open `driverguard_ai.py` and edit the configuration section at the top:

```python
# ── Email Configuration ──────────────────────────────────────
SENDER_EMAIL    = "your_gmail@gmail.com"
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"   # Gmail App Password
RECEIVER_EMAIL  = "family@gmail.com"

# ── Detection Tuning ─────────────────────────────────────────
EAR_THRESHOLD = 0.25    # Lower = less sensitive | Higher = more sensitive
CONSEC_FRAMES = 20      # Frames before episode triggers (~0.67s at 30fps)
```

### Gmail App Password Setup
1. Go to **Google Account → Security**
2. Enable **2-Step Verification**
3. Go to **App Passwords**
4. Select **Mail** → Generate
5. Copy the 16-character password into `SENDER_PASSWORD`

---

## 🔔 Alert Logic

DriverGuard-AI uses a **3-stage progressive alert system** to avoid unnecessary panic:

```
Episode 1 (First drowsy detection)
  └── 🔔 Single beep — "Stay alert"

Episode 2 (Second drowsy detection)
  └── 🔔🔔 Double beep — "Final warning"

Episode 3+ (Third drowsy detection)
  └── 🔔🔔🔔 Triple beep
  └── 📸 Screenshot saved to /drowsy_screenshots/
  └── 📧 Emergency email sent to family with screenshot attached
```

> **Why 3 stages?** A single blink or brief eye closure should not trigger an emergency. The 3-stage system ensures alerts are meaningful and not false positives.

---

## 📊 Dashboard Preview

The live HUD displays the following in real-time:

```
┌─────────────────────────────────────────────────────────┐
│  DriverGuard AI          Driver: Rajesh        14:32:05  │
├──────────────┬──────────────────────────────────────────┤
│ EAR VALUE    │                                          │
│  0.284       │          [ LIVE WEBCAM FEED ]            │
│  EYES OPEN   │                                          │
│──────────────│                                          │
│ Session 04:21│                                          │
│ Alerts     2 │                                          │
│ Episodes 1/3→│                                          │
│ Risk    SAFE │                                          │
├──────────────┴──────────────────────────────────────────┤
│ [EAR Live Graph ~~~~~~~~~~~~~~~~]  ACCIDENT RISK: SAFE  │
│                                    [Fatigue ██░░░ 24/100]│
└─────────────────────────────────────────────────────────┘
```

**Risk Levels:**

| Score | Level | Color |
|---|---|---|
| 0 – 29 | ✅ SAFE | Green |
| 30 – 54 | ⚠️ CAUTION | Yellow |
| 55 – 74 | 🟠 WARNING | Orange |
| 75 – 89 | 🔴 HIGH RISK | Red |
| 90 – 100 | 🚨 CRITICAL | Bright Red |

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `Cannot load shape_predictor` | `.dat` file missing from project folder |
| `dlib build failed` | Install Visual Studio C++ Build Tools |
| `No module named face_recognition` | Run `pip install face_recognition` |
| `No face detected` | Ensure good lighting and face camera directly |
| Black screen | Change `VideoStream(src=0)` to `src=1` |
| Email not sending | Check App Password, not your login password |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an issue for bugs or feature requests
- Fork the repo and submit a pull request
- Star ⭐ the repo if you found it useful

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Aashutosh**
- GitHub: ( https://github.com/YadavAashutosh )
- Project: ( https://github.com/YadavAashutosh/DriverGuard )

---

<p align="center">
  Made with ❤️ for Road Safety
  <br/>
  <i>"Because every driver deserves to reach home safely."</i>
</p>
