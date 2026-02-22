"""
Video data collection tool for SLT.

Opens the webcam with MediaPipe hand overlay, lets you type a label and
video count on-screen, then record / save / undo clips via keyboard.
Saved videos are raw camera frames (no overlay) at native resolution.

State machine: InputLabel -> InputCount -> Idle -> Recording -> Review
               (loops back to InputLabel after all clips for a label)
"""

import cv2
import time
import mediapipe as mp
import numpy as np
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_videos"

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_S    = cv2.FONT_HERSHEY_PLAIN
WIN_NAME  = "SLT — Data Collection"

STATE_INPUT_LABEL = 0
STATE_INPUT_COUNT = 1
STATE_IDLE        = 2
STATE_RECORDING   = 3
STATE_REVIEW      = 4


# ── Drawing helpers ───────────────────────────────────────────

def draw_dark_bar(frame, y, h, alpha=0.80):
    """Draw a semi-transparent dark bar across the frame."""
    overlay = frame.copy()
    fw = frame.shape[1]
    cv2.rectangle(overlay, (0, y), (fw, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_text_centered(frame, text, y, scale=0.7, color=(255, 255, 255),
                       thickness=2):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = (frame.shape[1] - size[0]) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def draw_input_prompt(frame, prompt, text_buf, cursor_on):
    """Draw a centered text-input overlay."""
    fh, fw = frame.shape[:2]
    bar_h = 120
    bar_y = fh // 2 - bar_h // 2
    draw_dark_bar(frame, bar_y, bar_h, alpha=0.90)

    draw_text_centered(frame, prompt, bar_y + 40, scale=0.7,
                       color=(180, 180, 180))

    cursor = "|" if cursor_on else " "
    display = text_buf + cursor
    draw_text_centered(frame, display, bar_y + 85, scale=1.0,
                       color=(0, 255, 200), thickness=2)


def draw_controls(frame, lines):
    """Draw control hints at the bottom of the frame."""
    fh, fw = frame.shape[:2]
    bar_h = 18 * len(lines) + 16
    draw_dark_bar(frame, fh - bar_h, bar_h, alpha=0.80)
    for i, line in enumerate(lines):
        y = fh - bar_h + 22 + i * 18
        cv2.putText(frame, line, (14, y), FONT_S, 1.0,
                    (160, 160, 160), 1)


def draw_top_bar(frame, label, saved_count, target_count, state_text,
                 state_color=(255, 255, 255)):
    """Draw the top status bar."""
    draw_dark_bar(frame, 0, 55, alpha=0.85)
    cv2.putText(frame, f'Label: {label}', (14, 25), FONT, 0.6,
                (0, 255, 200), 2)
    cv2.putText(frame, f'{saved_count}/{target_count}', (14, 48), FONT, 0.55,
                (200, 200, 200), 1)
    fw = frame.shape[1]
    sz = cv2.getTextSize(state_text, FONT, 0.6, 2)[0]
    cv2.putText(frame, state_text, (fw - sz[0] - 14, 25), FONT, 0.6,
                state_color, 2)


def draw_rec_indicator(frame, elapsed_sec):
    """Pulsing red REC dot + elapsed time."""
    show_dot = int(elapsed_sec * 3) % 2 == 0
    if show_dot:
        cv2.circle(frame, (30, 85), 10, (0, 0, 255), -1)
    cv2.putText(frame, f'REC  {elapsed_sec:.1f}s', (48, 92), FONT, 0.6,
                (0, 0, 255), 2)


# ── Main ──────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0 or cam_fps > 120:
        cam_fps = 30.0
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cam_w}x{cam_h} @ {cam_fps:.0f} fps")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1,
        max_num_hands=2,
    )

    state        = STATE_INPUT_LABEL
    text_buf     = ""
    label        = ""
    target_count = 0
    saved_count  = 0
    saved_files: list[Path] = []

    raw_frames: list[np.ndarray] = []
    rec_start  = 0.0
    review_frame: np.ndarray | None = None

    print(f"Output directory: {DATA_DIR}")
    print("Type in the OpenCV window. Press Q at any time to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        raw = frame.copy()

        # MediaPipe overlay (visual guide only)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms,
                                       mp_hands.HAND_CONNECTIONS)

        # Cursor blink (~2 Hz)
        cursor_on = int(time.time() * 2) % 2 == 0

        # ── STATE: Input Label ────────────────────────────────
        if state == STATE_INPUT_LABEL:
            draw_input_prompt(frame, "Label name:", text_buf, cursor_on)
            draw_controls(frame, [
                "Type label name, then press ENTER to confirm",
                "Q = quit",
            ])

        # ── STATE: Input Count ────────────────────────────────
        elif state == STATE_INPUT_COUNT:
            draw_input_prompt(frame, f"Videos for '{label}':",
                              text_buf, cursor_on)
            draw_controls(frame, [
                "Type number of videos, then press ENTER",
                "Q = quit",
            ])

        # ── STATE: Idle ───────────────────────────────────────
        elif state == STATE_IDLE:
            draw_top_bar(frame, label, saved_count, target_count,
                         "READY", (0, 255, 200))
            draw_controls(frame, [
                "SPACE = start recording",
                "U = undo last clip",
                "Q = quit",
            ])

        # ── STATE: Recording ─────────────────────────────────
        elif state == STATE_RECORDING:
            raw_frames.append(raw)
            elapsed = time.time() - rec_start
            draw_top_bar(frame, label, saved_count, target_count,
                         "RECORDING", (0, 0, 255))
            draw_rec_indicator(frame, elapsed)
            draw_controls(frame, [
                "SPACE = stop recording",
            ])

        # ── STATE: Review ─────────────────────────────────────
        elif state == STATE_REVIEW:
            if review_frame is not None:
                frame = review_frame.copy()
                # Re-draw landmarks on the frozen frame for consistency
                rgb_r = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_r = hands.process(rgb_r)
                if res_r.multi_hand_landmarks:
                    for hand_lms in res_r.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_lms,
                                               mp_hands.HAND_CONNECTIONS)

            n_frames = len(raw_frames)
            dur = n_frames / cam_fps if cam_fps > 0 else 0
            draw_dark_bar(frame, 0, 70, alpha=0.85)
            draw_text_centered(frame, "Review Clip", 28, scale=0.7,
                               color=(0, 200, 255))
            draw_text_centered(frame, f"{n_frames} frames | {dur:.1f}s", 55,
                               scale=0.5, color=(180, 180, 180))
            draw_controls(frame, [
                "O = save clip",
                "SPACE = discard & re-record",
                "Q = quit",
            ])

        cv2.imshow(WIN_NAME, frame)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        # --- Text input states ---
        if state in (STATE_INPUT_LABEL, STATE_INPUT_COUNT):
            if key == 13:  # Enter
                if state == STATE_INPUT_LABEL and text_buf.strip():
                    label = text_buf.strip()
                    text_buf = ""
                    saved_count = 0
                    saved_files.clear()
                    label_dir = DATA_DIR / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    existing = sorted(label_dir.glob(f"{label}_*.mp4"))
                    if existing:
                        last_num = max(
                            int(p.stem.split("_")[-1])
                            for p in existing
                            if p.stem.split("_")[-1].isdigit()
                        )
                        saved_count = last_num
                        saved_files.extend(existing)
                        print(f"  Found {len(existing)} existing clips "
                              f"for '{label}', continuing from "
                              f"{label}_{saved_count + 1}")
                    state = STATE_INPUT_COUNT
                elif state == STATE_INPUT_COUNT and text_buf.strip().isdigit():
                    target_count = int(text_buf.strip())
                    if target_count < 1:
                        target_count = 1
                    text_buf = ""
                    state = STATE_IDLE
                    print(f"  Recording '{label}': "
                          f"{saved_count}/{target_count} done")

            elif key in (8, 127):  # Backspace / Delete
                text_buf = text_buf[:-1]

            elif 32 <= key < 127 and key != ord(' '):
                ch = chr(key)
                if state == STATE_INPUT_COUNT:
                    if ch.isdigit():
                        text_buf += ch
                else:
                    text_buf += ch

        # --- Idle state ---
        elif state == STATE_IDLE:
            if key == ord(' '):
                raw_frames.clear()
                rec_start = time.time()
                state = STATE_RECORDING
                print(f"  REC started for {label}_{saved_count + 1}")

            elif key in (ord('u'), ord('U')):
                if saved_files:
                    last = saved_files.pop()
                    if last.exists():
                        last.unlink()
                        saved_count -= 1
                        print(f"  Undone: deleted {last.name} "
                              f"({saved_count}/{target_count})")
                    else:
                        print(f"  File already gone: {last.name}")
                else:
                    print("  Nothing to undo")

        # --- Recording state ---
        elif state == STATE_RECORDING:
            if key == ord(' '):
                state = STATE_REVIEW
                review_frame = frame.copy()
                print(f"  REC stopped: {len(raw_frames)} frames captured")

        # --- Review state ---
        elif state == STATE_REVIEW:
            if key == ord('o') or key == ord('O'):
                label_dir = DATA_DIR / label
                label_dir.mkdir(parents=True, exist_ok=True)
                saved_count += 1
                fname = label_dir / f"{label}_{saved_count}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    str(fname), fourcc, cam_fps, (cam_w, cam_h))
                for f in raw_frames:
                    writer.write(f)
                writer.release()

                saved_files.append(fname)
                raw_frames.clear()
                review_frame = None
                print(f"  Saved: {fname.name} "
                      f"({saved_count}/{target_count})")

                if saved_count >= target_count:
                    print(f"  All {target_count} clips for '{label}' done!\n")
                    text_buf = ""
                    state = STATE_INPUT_LABEL
                else:
                    state = STATE_IDLE

            elif key == ord(' '):
                raw_frames.clear()
                review_frame = None
                state = STATE_IDLE
                print("  Clip discarded")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDone. Videos saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
