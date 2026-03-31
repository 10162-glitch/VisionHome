"""
VisionHome: CV-Powered Smart Home Control System
=================================================
Computer Vision Course — BYOP Project

Controls simulated home appliances using hand gestures and facial events
detected in real-time from a webcam feed.

Appliances Controlled:
  - Light        : Toggle with open palm
  - Fan (3 speeds): Finger count (1, 2, or 3 fingers)
  - Door         : Head tilt left (unlock) / right (lock)
  - Security Alarm: Double blink arms/disarms

Requirements:
  pip install opencv-python mediapipe numpy

Usage:
  python vision_home.py
  python vision_home.py --camera 1         # external USB webcam
  python vision_home.py --quality high     # higher accuracy, slower
  python vision_home.py --no-calibrate     # skip head calibration
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import sys

# ─────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="VisionHome Smart Home CV Controller")
parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
parser.add_argument("--quality", choices=["fast", "high"], default="fast",
                    help="Model complexity: fast=0 (default), high=1")
parser.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="Skip head tilt calibration phase")
args = parser.parse_args()

MODEL_COMPLEXITY = 0 if args.quality == "fast" else 1

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=MODEL_COMPLEXITY
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ─────────────────────────────────────────────
#  APPLIANCE STATE
# ─────────────────────────────────────────────
state = {
    "light": False,       # True = ON
    "fan":   0,           # 0 = OFF, 1/2/3 = speed
    "door":  "LOCKED",    # "LOCKED" or "UNLOCKED"
    "alarm": "DISARMED"   # "ARMED" or "DISARMED"
}

# Cooldown tracker: gesture_name -> last_trigger_time
COOLDOWN_MS = 600
last_trigger = {k: 0 for k in ["light", "fan", "door", "alarm", "all_off"]}

# ─────────────────────────────────────────────
#  BLINK TRACKING
# ─────────────────────────────────────────────
blink_times = []           # timestamps of recent blinks
DOUBLE_BLINK_WINDOW = 1.0  # seconds

# EAR threshold — calibrated adaptively, initial value
EAR_THRESHOLD = 0.22
eye_open_history = []      # rolling average of open-eye EAR

# MediaPipe face mesh landmark indices for eyes
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
#  HEAD TILT CALIBRATION
# ─────────────────────────────────────────────
neutral_nose_offset = 0.0   # calibrated nose horizontal offset
calibrated = False
calibration_samples = []
CALIBRATION_DURATION = 3.0  # seconds

# Nose and cheekbone indices for head tilt
NOSE_TIP   = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
HEAD_TILT_THRESHOLD = 0.08  # fraction of face width

# ─────────────────────────────────────────────
#  HELPER: Eye Aspect Ratio
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """Compute EAR for given eye landmark indices."""
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in eye_indices
    ])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

# ─────────────────────────────────────────────
#  HELPER: Cooldown check
# ─────────────────────────────────────────────
def can_trigger(gesture_name):
    now = time.time() * 1000
    if now - last_trigger[gesture_name] > COOLDOWN_MS:
        last_trigger[gesture_name] = now
        return True
    return False

# ─────────────────────────────────────────────
#  HELPER: Finger state (is finger up?)
# ─────────────────────────────────────────────
def get_finger_states(hand_landmarks):
    """
    Returns [thumb, index, middle, ring, pinky] as True/False (up/down).
    Uses y-coordinate comparison: tip vs. PIP joint.
    Landmark IDs: tip=4,8,12,16,20 | pip=3,6,10,14,18
    """
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = []
    for i, (tip, pip) in enumerate(zip(tips, pips)):
        if i == 0:  # thumb: use x-axis
            states.append(lm[tip].x < lm[pip].x)
        else:
            states.append(lm[tip].y < lm[pip].y)  # lower y = higher on screen = finger up
    return states

# ─────────────────────────────────────────────
#  HELPER: Gesture recognition from finger states
# ─────────────────────────────────────────────
def recognize_gesture(finger_states):
    """Map finger states to a named gesture string."""
    thumb, index, middle, ring, pinky = finger_states
    fingers_up = sum([index, middle, ring, pinky])  # exclude thumb

    if all([index, middle, ring, pinky]):
        return "OPEN_PALM"       # all 4 fingers up + possibly thumb -> light toggle
    elif index and not middle and not ring and not pinky:
        return "FAN_1"
    elif index and middle and not ring and not pinky:
        return "FAN_2"
    elif index and middle and ring and not pinky:
        return "FAN_3"
    elif not any([index, middle, ring, pinky]):
        return "FIST"            # all fingers down
    return "UNKNOWN"

# ─────────────────────────────────────────────
#  APPLY GESTURE TO STATE
# ─────────────────────────────────────────────
fist_start_time = None
FIST_HOLD_DURATION = 2.0  # seconds to hold fist for all-off

def apply_gesture(gesture):
    global fist_start_time

    if gesture == "OPEN_PALM":
        if can_trigger("light"):
            state["light"] = not state["light"]
            fist_start_time = None

    elif gesture == "FAN_1":
        if can_trigger("fan"):
            state["fan"] = 1
            fist_start_time = None

    elif gesture == "FAN_2":
        if can_trigger("fan"):
            state["fan"] = 2
            fist_start_time = None

    elif gesture == "FAN_3":
        if can_trigger("fan"):
            state["fan"] = 3
            fist_start_time = None

    elif gesture == "FIST":
        if fist_start_time is None:
            fist_start_time = time.time()
        elif time.time() - fist_start_time >= FIST_HOLD_DURATION:
            if can_trigger("all_off"):
                state["light"] = False
                state["fan"]   = 0
                state["door"]  = "LOCKED"
                state["alarm"] = "DISARMED"
                fist_start_time = None
    else:
        fist_start_time = None  # reset fist timer if gesture breaks

# ─────────────────────────────────────────────
#  APPLY BLINK EVENT
# ─────────────────────────────────────────────
def apply_blink():
    global blink_times
    now = time.time()
    blink_times.append(now)
    # Keep only blinks in the last DOUBLE_BLINK_WINDOW seconds
    blink_times = [t for t in blink_times if now - t <= DOUBLE_BLINK_WINDOW]
    if len(blink_times) >= 2:
        if can_trigger("alarm"):
            state["alarm"] = "ARMED" if state["alarm"] == "DISARMED" else "DISARMED"
            blink_times = []

# ─────────────────────────────────────────────
#  APPLY HEAD TILT
# ─────────────────────────────────────────────
def apply_head_tilt(offset):
    if offset < -(HEAD_TILT_THRESHOLD):
        if can_trigger("door"):
            state["door"] = "UNLOCKED"
    elif offset > HEAD_TILT_THRESHOLD:
        if can_trigger("door"):
            state["door"] = "LOCKED"

# ─────────────────────────────────────────────
#  DASHBOARD OVERLAY
# ─────────────────────────────────────────────
def draw_dashboard(frame, fist_progress=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panel
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 320, 160
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def status_color(on):
        return (0, 220, 80) if on else (80, 80, 80)

    def row(label, value_text, is_on, y_pos):
        cv2.putText(frame, label, (panel_x + 14, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(frame, value_text, (panel_x + 160, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color(is_on), 2)

    cv2.putText(frame, "VisionHome Dashboard", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 0), 2)
    cv2.line(frame, (panel_x + 10, panel_y + 30), (panel_x + panel_w - 10, panel_y + 30),
             (80, 80, 80), 1)

    row("LIGHT",  "ON" if state["light"] else "OFF",       state["light"],             panel_y + 58)
    fan_label = f"SPEED {state['fan']}" if state["fan"] > 0 else "OFF"
    row("FAN",    fan_label,                                state["fan"] > 0,           panel_y + 86)
    row("DOOR",   state["door"],                            state["door"] == "UNLOCKED", panel_y + 114)
    row("ALARM",  state["alarm"],                           state["alarm"] == "ARMED",  panel_y + 142)

    # Fist hold progress bar
    if fist_progress > 0.01:
        bar_y = panel_y + panel_h + 6
        bar_w = int(panel_w * fist_progress)
        cv2.rectangle(frame, (panel_x, bar_y), (panel_x + panel_w, bar_y + 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, bar_y), (panel_x + bar_w, bar_y + 10), (0, 100, 255), -1)
        cv2.putText(frame, "Hold fist to cut all power...", (panel_x, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1)

    # Hint bar at bottom
    cv2.putText(frame, "Q = Quit  |  OpenPalm=Light  |  Fingers=Fan  |  HeadTilt=Door  |  Blink2x=Alarm",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return frame

# ─────────────────────────────────────────────
#  CALIBRATION OVERLAY
# ─────────────────────────────────────────────
def draw_calibration(frame, elapsed, duration):
    h, w = frame.shape[:2]
    pct = min(elapsed / duration, 1.0)
    bar_w = int(w * 0.6)
    bx = (w - bar_w) // 2
    by = h // 2 - 20
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 30), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * pct), by + 30), (0, 200, 100), -1)
    cv2.putText(frame, "Calibrating head neutral position — look straight ahead",
                (bx, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 1)
    cv2.putText(frame, f"{int(pct * 100)}%",
                (bx + bar_w // 2 - 20, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    global neutral_nose_offset, calibrated, calibration_samples, EAR_THRESHOLD

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[VisionHome] Starting... Press Q to quit.")
    if args.calibrate:
        print("[VisionHome] Look straight ahead for 3 seconds to calibrate head tilt.")

    calibration_start = time.time() if args.calibrate else None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        frame_count += 1

        # ── Hand detection (every frame)
        hand_results = hands.process(rgb)

        # ── Face detection (every 2nd frame for performance)
        face_results = None
        if frame_count % 2 == 0:
            face_results = face_mesh.process(rgb)

        rgb.flags.writeable = True

        # ── Process hands
        gesture = "UNKNOWN"
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 130, 60), thickness=2))
                finger_states = get_finger_states(hand_lm)
                gesture = recognize_gesture(finger_states)
                apply_gesture(gesture)

        # ── Process face
        in_blink = False
        if face_results and face_results.multi_face_landmarks:
            for face_lm in face_results.multi_face_landmarks:
                lms = face_lm.landmark

                # EAR calculation
                ear_l = eye_aspect_ratio(lms, LEFT_EYE, w, h)
                ear_r = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                ear = (ear_l + ear_r) / 2.0

                # Adaptive threshold calibration (track open-eye EAR)
                if ear > 0.25:
                    eye_open_history.append(ear)
                    if len(eye_open_history) > 150:
                        eye_open_history.pop(0)
                    if len(eye_open_history) > 30:
                        avg_ear = np.mean(eye_open_history)
                        EAR_THRESHOLD = avg_ear * 0.72  # 72% of average open-eye EAR

                if ear < EAR_THRESHOLD:
                    in_blink = True
                    apply_blink()

                # Head tilt
                nose_x  = lms[NOSE_TIP].x
                lcheek_x = lms[LEFT_CHEEK].x
                rcheek_x = lms[RIGHT_CHEEK].x
                face_center_x = (lcheek_x + rcheek_x) / 2.0
                face_width = abs(rcheek_x - lcheek_x)
                raw_offset = (nose_x - face_center_x) / face_width if face_width > 0 else 0

                # Calibration phase
                if args.calibrate and not calibrated:
                    elapsed = time.time() - calibration_start
                    calibration_samples.append(raw_offset)
                    frame = draw_calibration(frame, elapsed, CALIBRATION_DURATION)
                    if elapsed >= CALIBRATION_DURATION:
                        neutral_nose_offset = np.mean(calibration_samples)
                        calibrated = True
                        print(f"[VisionHome] Calibration complete. Neutral offset: {neutral_nose_offset:.4f}")
                else:
                    normalized_offset = raw_offset - neutral_nose_offset
                    apply_head_tilt(normalized_offset)

        # If not calibrating, skip calibration screen
        if not args.calibrate:
            calibrated = True

        # ── Fist progress bar
        fist_progress = 0.0
        if fist_start_time is not None:
            fist_progress = min((time.time() - fist_start_time) / FIST_HOLD_DURATION, 1.0)

        # ── Draw dashboard
        if calibrated:
            frame = draw_dashboard(frame, fist_progress)

            # Show current gesture in top-right
            gesture_color = (0, 220, 80) if gesture != "UNKNOWN" else (80, 80, 80)
            cv2.putText(frame, f"Gesture: {gesture}", (w - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, gesture_color, 2)

            # Blink indicator
            if in_blink:
                cv2.putText(frame, "BLINK DETECTED", (w - 260, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2)

        cv2.imshow("VisionHome — Smart Home Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[VisionHome] Session ended.")


if __name__ == "__main__":
    main()
