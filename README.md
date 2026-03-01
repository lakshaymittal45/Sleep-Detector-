# Sleep Detector for Drivers

Detects driver drowsiness by monitoring eye closure in real-time using your webcam.  
When the driver's eyes stay closed for **~2 seconds**, a **red warning overlay** and **audio alarm** are triggered.  
Normal blinks are ignored — only sustained eye closure fires the alert.

---

## How It Works

1. Captures live video from your webcam.
2. Uses **MediaPipe Face Mesh** to locate 468 facial landmarks every frame.
3. Calculates the **Eye Aspect Ratio (EAR)** for both eyes — a ratio that drops sharply when eyes close.
4. If EAR stays below `0.22` for **≥ 60 consecutive frames (~2 s at 30 fps)**, the alarm fires.
5. Normal blinks (~3–5 frames) are ignored via a grace period — only genuine sustained closure triggers the alarm.
6. The alarm stops automatically once eyes open, or manually with **R**.

---

## Project Structure

```
Sleep detector for drivers/
├── sleep_detector.py      ← main script
├── requirements.txt       ← Python dependencies
├── README.md
├── <your_alarm_audio>.mp3 ← any .wav or .mp3 file (auto-detected)
└── .venv/                 ← Python virtual environment
```

---

## First-Time Setup

> Do this once before running for the first time.

### 1. Create a virtual environment (if not already done)

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` appear at the start of your terminal prompt.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add your alarm audio

Drop any `.wav` or `.mp3` file into this folder.  
It will be auto-detected — no need to rename it.

> Free alarm sounds: https://freesound.org

---

## Running the App

Every time you want to run the detector:

**Step 1 — Activate the environment:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Step 2 — Start the detector:**
```powershell
python sleep_detector.py
```

A webcam window will open. Sit in front of your camera and the detector starts immediately.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the app |
| `R` | Silence / reset the alarm manually |

---

## On-Screen Display

| Element | Description |
|---------|-------------|
| **EAR value** (top-left) | Current Eye Aspect Ratio — lower means more closed |
| **Status** (top-center) | `AWAKE` (green) or `SLEEPING!` (red) |
| **FPS** (top-right) | Frames per second |
| **Progress bar** (bottom) | How close the closed-eye duration is to triggering the alarm |
| **Eye boxes** | Green = open, Red = closed |
| **Red overlay** | Full-screen warning when alarm is active |

---

## Configuration

Edit the constants at the top of `sleep_detector.py` to tune sensitivity:

| Constant            | Default | Description |
|---------------------|---------|-------------|
| `CAMERA_INDEX`      | `0`     | Webcam index — try `1` for an external camera |
| `EAR_THRESHOLD`     | `0.22`  | EAR below this = eyes considered closed (lower = less sensitive) |
| `CONSEC_FRAMES`     | `60`    | Closed-eye frames to trigger alarm (~2 s at 30 fps) |
| `BLINK_GRACE_FRAMES`| `8`     | Frames eyes must stay open to count as a genuine opening (~0.27 s) |
| `LOOP_ALARM`        | `True`  | Repeat alarm until eyes open / R is pressed |
| `SHOW_LANDMARKS`    | `False` | Draw all 468 face-mesh points (debug mode) |
| `OVERLAY_ALPHA`     | `0.45`  | Red overlay transparency (0 = invisible, 1 = solid) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Black / no camera window | Change `CAMERA_INDEX` to `1` or `2` |
| Alarm triggers on normal blinks | Lower `EAR_THRESHOLD` to `0.18`–`0.20` |
| Alarm doesn't trigger fast enough | Reduce `CONSEC_FRAMES` (e.g. `45` for ~1.5 s) |
| No audio playing | Make sure a `.wav` or `.mp3` file is in the project folder |
| `Import cv2` errors in VS Code | Select the `.venv` interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv` |

---

## Requirements

- Python 3.8+
- Webcam
- Packages: `opencv-python`, `mediapipe==0.10.9`, `pygame`, `numpy`, `scipy`
