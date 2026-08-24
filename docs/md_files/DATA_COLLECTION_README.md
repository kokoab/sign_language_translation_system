# SLT Data Collection Tool

Webcam-based video recording tool for building sign language datasets. Shows a live MediaPipe hand skeleton overlay while recording raw (no overlay) video clips, organized by label.

## Repository Structure

```
SLT-data-collection/
├── src/
│   └── collect_data.py        # Main collection script
├── data/
│   └── raw_videos/            # Recorded clips go here (auto-created)
│       ├── A/
│       │   ├── A_1.mp4
│       │   ├── A_2.mp4
│       │   └── ...
│       ├── B/
│       │   ├── B_1.mp4
│       │   └── ...
│       └── <label>/
│           └── <label>_N.mp4
├── requirements_collect.txt   # Python dependencies
├── setup_collect.bat          # One-click setup for Windows
├── setup_collect.sh           # One-click setup for macOS / Linux
└── README.md
```

## Requirements

- **Python 3.10+**
- A working webcam (built-in or USB)
- Internet connection (first run only, to install packages)

## Quick Start

### Windows

**Double-click `setup_collect.bat`** — that's it.

The script will automatically:

1. Detect whether Python 3.10+ is installed
2. Download and install Python 3.10.11 if it's missing (per-user, no admin needed)
3. Create a virtual environment (`venv_collect`)
4. Install all dependencies
5. Launch the data collection tool

If you prefer to do it manually:

```bat
python -m venv venv_collect
venv_collect\Scripts\activate
pip install -r requirements_collect.txt
python src\collect_data.py
```

### macOS / Linux

```bash
chmod +x setup_collect.sh
./setup_collect.sh
```

Or manually:

```bash
python3 -m venv venv_collect
source venv_collect/bin/activate
pip install -r requirements_collect.txt
python src/collect_data.py
```

> **macOS note:** The first launch may trigger a camera permission prompt — click **Allow**.

## How It Works

The tool runs as a state machine displayed in a single OpenCV window:

```
InputLabel → InputCount → Idle → Recording → Review
     ↑                                          │
     └──────── (after all clips done) ──────────┘
```

1. **InputLabel** — Type a label name (e.g. `A`, `hello`, `thank_you`) and press Enter
2. **InputCount** — Type how many clips you want to record for that label, press Enter
3. **Idle** — Live preview with hand skeleton. Press Space to start recording
4. **Recording** — Frames are captured in memory. Press Space to stop
5. **Review** — See the last frame. Press O to save, or Space to discard and re-record

After saving all clips for a label, the tool loops back to step 1 for the next label.

## Controls

| Key         | Action                                    |
|-------------|-------------------------------------------|
| `Enter`     | Confirm label name or video count         |
| `Space`     | Start recording / Stop recording / Discard clip |
| `O`         | Save the recorded clip                    |
| `U`         | Undo (delete) the last saved clip         |
| `Backspace` | Delete last character during text input   |
| `Q`         | Quit                                      |

## Output Format

- Videos are saved as `.mp4` files using the `mp4v` codec
- Resolution and FPS match your camera's native settings
- Saved frames are **raw camera frames** (no hand skeleton overlay)
- File naming: `<label>/<label>_1.mp4`, `<label>/<label>_2.mp4`, ...
- The tool detects existing clips and continues numbering from where you left off

## Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| `opencv-python`| Camera capture, video writing, UI|
| `mediapipe`    | Real-time hand landmark detection|
| `numpy`        | Frame buffer management          |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open camera` | Make sure no other app is using the webcam. Try unplugging and re-plugging USB cameras. |
| Black window / no video | Grant camera permission in system settings (macOS: System Preferences → Privacy → Camera). |
| `ModuleNotFoundError` | Make sure the virtual environment is activated before running. |
| Laggy preview | Close other camera apps. MediaPipe runs per-frame detection — lower-resolution cameras will be smoother. |
| Windows: Python install fails | Download Python manually from [python.org](https://www.python.org/downloads/) and check **"Add Python to PATH"** during installation. |
