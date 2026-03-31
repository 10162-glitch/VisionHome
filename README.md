# 🏠 VisionHome — CV-Powered Smart Home Control System

> Control your home appliances using only hand gestures and facial expressions — no touch, no voice, no special hardware.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Course](https://img.shields.io/badge/Course-Computer%20Vision%20BYOP-blueviolet)]()

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Gesture Reference](#gesture-reference)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Limitations](#limitations)
- [Future Scope](#future-scope)

---

## About the Project

**VisionHome** is a real-time, computer vision-based smart home simulation system. It uses a standard webcam to recognize hand gestures and facial events, mapping them to commands for home appliances — lights, fans, doors, and a security alarm.

The system runs **entirely offline on your local machine** — no cloud, no voice assistant, no IoT hardware required. It was built as a BYOP (Bring Your Own Project) for a Computer Vision course, applying core CV concepts including landmark detection, geometric feature extraction, Eye Aspect Ratio (EAR), and real-time video processing.

**Why?** Most smart home systems are expensive, proprietary, or require internet. VisionHome proves that a `₹0 webcam` you already own can be a universal home controller.

---

## Features

| Feature | Description |
|--------|-------------|
| 💡 **Light Control** | Toggle room lights with an open palm gesture |
| 🌀 **Fan Speed Control** | Set fan speed 1–3 using finger count |
| 🚪 **Door Lock/Unlock** | Tilt your head to lock or unlock the door |
| 🚨 **Security Alarm** | Arm/disarm alarm with a double blink |
| ✊ **Emergency All-OFF** | Hold a fist for 2 seconds to cut all appliances |
| 📊 **Live Dashboard** | Real-time on-screen appliance state overlay |
| ⚡ **No Hardware Needed** | Fully simulated — runs on any laptop with a webcam |
| 🔒 **Completely Offline** | No internet, no cloud, no microphone |

---

## Gesture Reference

| Appliance | Gesture / Trigger | Action |
|-----------|-------------------|--------|
| 💡 Lights | Open palm (all 5 fingers up) | Toggle ON / OFF |
| 🌀 Fan | 1 finger raised (index only) | Fan Speed 1 (Low) |
| 🌀 Fan | 2 fingers raised (index + middle) | Fan Speed 2 (Medium) |
| 🌀 Fan | 3 fingers raised (index + middle + ring) | Fan Speed 3 (High) |
| 🚪 Door | Head tilt LEFT | Unlock Door |
| 🚪 Door | Head tilt RIGHT | Lock Door |
| 🚨 Alarm | Double blink (2x within 1 second) | Toggle Alarm Armed/Disarmed |
| ✊ All OFF | Closed fist, held 2 seconds | All Appliances OFF |

> 💡 **Tip:** Each gesture has a 500ms cooldown to prevent accidental re-triggering. Gestures are only registered when your hand is clearly in frame.

---

## Tech Stack

- **Python 3.9+** — Core language
- **OpenCV 4.x** (`cv2`) — Webcam capture, frame processing, UI overlay
- **MediaPipe 0.10+** — Hand landmark detection (21 points) + Face Mesh (468 points)
- **NumPy** — Geometric calculations on landmark coordinates

No deep learning training required. No GPU required. No cloud required.

---

## Project Structure

```
VisionHome/
│
├── vision_home.py          # Main application — run this file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
│
├── docs/
│   ├── gesture_guide.md    # Detailed gesture documentation
│   └── architecture.md     # System design notes
│
└── screenshots/
    ├── dashboard.png        # Live dashboard screenshot
    ├── light_gesture.png    # Light toggle gesture
    └── fan_gesture.png      # Fan speed gesture
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher ([download](https://python.org/downloads))
- A webcam (built-in or USB)
- Good lighting (natural or lamp light on your face and hands)

> ⚠️ Works on **Windows, macOS, and Linux**. On Linux, you may need `v4l2` webcam drivers installed.

---

### Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/10162-glitch/VisionHome.git
cd VisionHome
```

**Step 2 — (Recommended) Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Running the Application

```bash
python vision_home.py
```

**Optional flags:**

```bash
python vision_home.py --camera 1          # Use external USB camera (default: 0)
python vision_home.py --quality high      # High accuracy mode (slower on old hardware)
python vision_home.py --no-calibrate      # Skip the 3-second head calibration phase
```

**To quit:** Press `Q` while the application window is active.

---

### Requirements File

The `requirements.txt` contains:

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

## How It Works

### Processing Pipeline

Every frame from the webcam goes through this pipeline:

```
Webcam Frame
    ↓
BGR → RGB Conversion (OpenCV → MediaPipe format)
    ↓
Hand Landmark Detection (MediaPipe Hands — 21 points/hand)
    ↓
Face Mesh Detection (MediaPipe Face Mesh — 468 points)
    ↓
Feature Extraction:
  • Finger states (up/down from y-coordinates)
  • Eye Aspect Ratio (EAR) for blink detection
  • Nose horizontal offset for head tilt
    ↓
Gesture Matching + Cooldown Check
    ↓
Appliance State Update
    ↓
Dashboard Overlay Rendered on Frame
    ↓
Display Window
```

### Blink Detection (EAR)

Eye blink is detected using the **Eye Aspect Ratio** formula:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

Where p1–p6 are specific facial landmark points around each eye. When EAR drops below a threshold (~0.2), a blink is registered. Two blinks within 1 second triggers the alarm gesture.

### Head Tilt Detection

The system measures the horizontal offset of the nose tip landmark relative to the midpoint of the two cheekbone landmarks. During startup, it captures a 3-second baseline of the user's neutral position. A tilt is detected when the offset exceeds ±15% of the face width from the personal neutral baseline.

---

## Screenshots

> *(Add screenshots to the `screenshots/` folder and they will appear here)*

| Dashboard View | Gesture Detection |
|----------------|-------------------|
| ![dashboard](screenshots/dashboard.png) | ![gesture](screenshots/light_gesture.png) |

**Dashboard Legend:**

```
[💡 LIGHT  : ON ]   [🌀 FAN    : SPD 2]
[🚪 DOOR   : LOCKED] [🚨 ALARM  : ARMED]
```

---

## Limitations

- Requires **good, consistent lighting** — performance degrades in low light or high contrast
- **Single person** only — designed for one user in frame at a time
- Gestures are **calibrated for right-hand dominance** by default (configurable)
- Head tilt detection is less reliable for users wearing glasses with thick frames
- No real hardware is controlled — this is a **simulation**

---

## Future Scope

- [ ] Raspberry Pi + GPIO integration for real relay/appliance control
- [ ] Multiple user profiles with face recognition
- [ ] Web dashboard for remote monitoring (Flask/FastAPI)
- [ ] Custom gesture training with TensorFlow Lite
- [ ] Voice command fusion
- [ ] Mobile app companion

---

> *Built with a webcam, Python, and the belief that your home should respond to you — not the other way around.*
