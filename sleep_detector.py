"""
Sleep Detector for Drivers
============================
Detects eye closure using MediaPipe Face Mesh and triggers an audio alarm
when the driver appears to be falling asleep.

Usage:
    python sleep_detector.py

Controls:
    Q  - Quit
    R  - Reset alarm / acknowledge alert

Place your alarm audio file (alarm.wav or alarm.mp3) in the same folder
as this script before running.
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
import pygame
from scipy.spatial import distance as dist

# ──────────────────────────────────────────────
#  CONFIGURATION  (tweak these values as needed)
# ──────────────────────────────────────────────
CAMERA_INDEX          = 0       # 0 = default webcam
EAR_THRESHOLD         = 0.22    # EAR below this = eyes closed (lower = less sensitive)
CONSEC_FRAMES         = 60      # frames eyes must stay closed to trigger alarm
                                # (at ~30 fps that is ~2 seconds)
BLINK_GRACE_FRAMES    = 8       # if eyes reopen for FEWER than this many frames
                                # the closed-frame counter is NOT reset (blink ignored)
HEAD_DOWN_PITCH       = 28     # degrees; if head pitch exceeds this, eye check is paused
                                # (head tilted down ≠ eyes closed)
ALARM_FILES           = ["alarm.wav", "alarm.mp3"]   # fallback names; any .wav/.mp3 in dir is auto-detected
LOOP_ALARM            = True    # play alarm on repeat until eyes open / R pressed
SHOW_LANDMARKS        = False   # draw all face-mesh dots (noisy; set True to debug)
OVERLAY_ALPHA         = 0.45    # transparency of the red warning overlay
# ──────────────────────────────────────────────


# ── MediaPipe eye-landmark indices (6 pts per eye for EAR) ──────────────────
# Order: [P1(outer), P2(top-outer), P3(top-inner), P4(inner), P5(bot-inner), P6(bot-outer)]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [ 33, 160, 158, 133, 153, 144]

# ── Head pose landmarks (6 pts matched to a generic 3-D face model) ──────────
# Indices:  nose tip, chin, left-eye outer, right-eye outer, left-mouth, right-mouth
HEAD_POSE_IDX = [1, 152, 263, 33, 287, 57]
HEAD_POSE_3D  = np.array([
    [ 0.0,    0.0,    0.0 ],   # nose tip
    [ 0.0, -330.0,  -65.0],   # chin
    [-225.0, 170.0,-135.0],   # left eye outer corner
    [ 225.0, 170.0,-135.0],   # right eye outer corner
    [-150.0,-150.0,-125.0],   # left mouth corner
    [ 150.0,-150.0,-125.0],   # right mouth corner
], dtype=np.float64)


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """Return the Eye Aspect Ratio for one eye given its 6 landmark indices."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    p1, p2, p3, p4, p5, p6 = pts
    # vertical distances
    A = dist.euclidean(p2, p6)
    B = dist.euclidean(p3, p5)
    # horizontal distance
    C = dist.euclidean(p1, p4)
    ear = (A + B) / (2.0 * C)
    return ear


def get_head_pitch(landmarks, img_w, img_h):
    """Return head pitch in degrees using solvePnP.
    Positive = nose tilting downward (head bowed). Returns 0.0 on failure."""
    img_pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in HEAD_POSE_IDX],
        dtype=np.float64
    )
    focal      = img_w
    cam_matrix = np.array([[focal, 0, img_w / 2],
                            [0, focal, img_h / 2],
                            [0,     0,          1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_POSE_3D, img_pts, cam_matrix,
                                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    rot, _ = cv2.Rodrigues(rvec)
    pitch   = np.degrees(np.arcsin(-rot[2][1]))   # elevation angle
    return pitch


def find_alarm_file():
    """Search the script directory for any .wav or .mp3 file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # First try the explicit fallback names
    for name in ALARM_FILES:
        path = os.path.join(script_dir, name)
        if os.path.isfile(path):
            return path
    # Then scan for any audio file in the directory
    for f in os.listdir(script_dir):
        if f.lower().endswith((".wav", ".mp3")):
            return os.path.join(script_dir, f)
    return None


def draw_eye_box(frame, landmarks, eye_indices, img_w, img_h, color):
    """Draw a bounding rectangle around an eye."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    pts = np.array(pts)
    x, y, w, h = cv2.boundingRect(pts)
    pad = 6
    cv2.rectangle(frame, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 2)


def draw_hud(frame, ear, frame_counter, alarming, fps, head_down=False, pitch=0.0):
    """Render the HUD (EAR value, status bar, frame counter)."""
    h, w = frame.shape[:2]

    # semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    if head_down:
        status_text  = "HEAD DOWN"
        status_color = (0, 200, 255)   # orange – paused, not an alarm
    elif alarming:
        status_text  = "SLEEPING!"
        status_color = (30, 30, 255)
    else:
        status_text  = "AWAKE"
        status_color = (50, 220, 50)

    cv2.putText(frame, f"EAR: {ear:.3f}",
                (10, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, status_text,
                (int(w / 2) - 80, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, status_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 110, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (180, 180, 180), 1, cv2.LINE_AA)

    # pitch angle (bottom-right)
    cv2.putText(frame, f"Pitch: {pitch:+.1f}deg",
                (w - 175, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # closed-eye frame progress bar (hidden when head down)
    if not head_down:
        bar_max_w = 220
        fill = int((frame_counter / CONSEC_FRAMES) * bar_max_w)
        fill = min(fill, bar_max_w)
        bar_color = (0, 165, 255) if frame_counter < CONSEC_FRAMES else (0, 0, 220)
        cv2.rectangle(frame, (10, h - 22), (10 + bar_max_w, h - 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, h - 22), (10 + fill, h - 8), bar_color, -1)
        cv2.putText(frame, "Eye-closed duration",
                    (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Detection paused (head tilted down)",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)


def draw_alarm_overlay(frame):
    """Pulsing red overlay shown while the alarm is active."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    h, w = frame.shape[:2]
    text = "!  WAKE UP  !"
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 2.2, 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx, ty = (w - tw) // 2, (h + th) // 2
    # shadow
    cv2.putText(frame, text, (tx + 3, ty + 3), font, scale, (0, 0, 0),   thick + 2, cv2.LINE_AA)
    # main
    cv2.putText(frame, text, (tx, ty),          font, scale, (255, 255, 255), thick,     cv2.LINE_AA)

    cv2.putText(frame, "Press R to reset",
                (w // 2 - 95, ty + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (230, 230, 230), 1, cv2.LINE_AA)


def main():
    # ── Alarm audio setup ────────────────────────────────────────────────────
    pygame.mixer.init()
    alarm_path = find_alarm_file()
    if alarm_path is None:
        print(
            "[WARNING] No alarm audio file found in the script directory.\n"
            f"          Please add one of: {ALARM_FILES}\n"
            "          The detector will still run but without audio."
        )
    else:
        print(f"[INFO] Alarm file loaded: {alarm_path}")
        pygame.mixer.music.load(alarm_path)

    def play_alarm():
        if alarm_path and not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1 if LOOP_ALARM else 0)

    def stop_alarm():
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── MediaPipe Face Mesh ───────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # ── State ─────────────────────────────────────────────────────────────────
    closed_frames = 0
    open_frames   = 0   # consecutive frames eyes have been open
    alarming      = False
    prev_time     = time.time()

    print("[INFO] Sleep detector running.  Press Q to quit, R to reset alarm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # FPS
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear = 0.0
        pitch = 0.0
        head_down = False

        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0].landmark

            # EAR for both eyes
            left_ear  = eye_aspect_ratio(face_lms, LEFT_EYE_IDX,  w, h)
            right_ear = eye_aspect_ratio(face_lms, RIGHT_EYE_IDX, w, h)
            ear       = (left_ear + right_ear) / 2.0

            # Head pose – detect if driver is looking down
            pitch     = get_head_pitch(face_lms, w, h)
            head_down = pitch > HEAD_DOWN_PITCH

            # Eye bounding boxes
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
            draw_eye_box(frame, face_lms, LEFT_EYE_IDX,  w, h, eye_color)
            draw_eye_box(frame, face_lms, RIGHT_EYE_IDX, w, h, eye_color)

            if SHOW_LANDMARKS:
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style(),
                )

            # ── Sleep detection logic ─────────────────────────────────────────
            if head_down:
                # Head is tilted down — eyes appear compressed; skip counting
                open_frames   = 0
                closed_frames = max(0, closed_frames - 2)  # slowly decay counter
            elif ear < EAR_THRESHOLD:
                closed_frames += 1
                open_frames    = 0      # eyes are closed; reset open streak
            else:
                open_frames += 1
                if open_frames >= BLINK_GRACE_FRAMES:
                    # Eyes have been open long enough — genuine opening, reset counter
                    closed_frames = 0
                # else: brief re-opening (blink) — keep closed_frames accumulating

            if closed_frames >= CONSEC_FRAMES:
                if not alarming:
                    alarming = True
                    print("[ALERT] Driver may be sleeping!")
                play_alarm()
            elif alarming and open_frames >= BLINK_GRACE_FRAMES:
                # eyes genuinely opened → stop alarm
                alarming      = False
                closed_frames = 0
                open_frames   = 0
                stop_alarm()
                print("[INFO] Eyes opened – alarm stopped.")
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        # ── Overlays ──────────────────────────────────────────────────────────
        if alarming:
            draw_alarm_overlay(frame)

        draw_hud(frame, ear, closed_frames, alarming, fps, head_down, pitch)

        cv2.imshow("Sleep Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            alarming      = False
            closed_frames = 0
            open_frames   = 0
            stop_alarm()
            print("[INFO] Alarm reset by user.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[INFO] Detector stopped.")


if __name__ == "__main__":
    main()
