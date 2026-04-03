"""
Fast webcam + rtmlib using PoseTracker.
YOLOX runs every 7th frame only. Pose model runs on cropped bbox (faster).

Press Q to quit, S to start/stop recording.
"""
import cv2
import numpy as np
import time

try:
    from rtmlib import Wholebody, PoseTracker
except ImportError:
    print("Install rtmlib: pip install rtmlib onnxruntime")
    exit(1)

# Keypoint indices
_LHAND = list(range(91, 112))
_RHAND = list(range(112, 133))
_FACE = [23+30, 23+8, 23+27, 23+0, 23+16]
_HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

GREEN, RED, BLUE, YELLOW, WHITE = (0,255,0), (0,0,255), (255,150,0), (0,255,255), (255,255,255)


def draw_keypoints(frame, kps, scr):
    for i, idx in enumerate(_LHAND):
        if scr[idx] > 0.3:
            cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 3, GREEN, -1)
    for a, b in _HAND_EDGES:
        ia, ib = _LHAND[a], _LHAND[b]
        if scr[ia] > 0.3 and scr[ib] > 0.3:
            cv2.line(frame, (int(kps[ia][0]), int(kps[ia][1])),
                    (int(kps[ib][0]), int(kps[ib][1])), GREEN, 1)
    for i, idx in enumerate(_RHAND):
        if scr[idx] > 0.3:
            cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 3, RED, -1)
    for a, b in _HAND_EDGES:
        ia, ib = _RHAND[a], _RHAND[b]
        if scr[ia] > 0.3 and scr[ib] > 0.3:
            cv2.line(frame, (int(kps[ia][0]), int(kps[ia][1])),
                    (int(kps[ib][0]), int(kps[ib][1])), RED, 1)
    for idx in _FACE:
        if scr[idx] > 0.3:
            cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 5, BLUE, -1)
    for idx in [5, 6, 7, 8, 9, 10]:
        if scr[idx] > 0.3:
            cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 5, YELLOW, -1)


def main():
    print("Initializing PoseTracker (det every 7 frames)...")
    tracker = PoseTracker(
        Wholebody,
        det_frequency=7,
        mode='lightweight',
        backend='onnxruntime',
        device='cpu',
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Camera ready. Q=Quit S=Record")

    recording = False
    recorded_frames = []
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        t0 = time.time()

        keypoints, scores = tracker(frame)

        t1 = time.time()
        fps = 1.0 / max(t1 - t0, 0.001)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        display = frame.copy()

        if keypoints is not None and len(keypoints) > 0:
            kps = keypoints[0]
            scr = scores[0]
            draw_keypoints(display, kps, scr)

            l_count = sum(1 for idx in _LHAND if scr[idx] > 0.3)
            r_count = sum(1 for idx in _RHAND if scr[idx] > 0.3)
            cv2.putText(display, f"L:{l_count}/21 R:{r_count}/21 FPS:{avg_fps:.0f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        if recording:
            cv2.rectangle(display, (0, 0), (639, 479), RED, 4)
            sec = len(recorded_frames) / 30.0
            cv2.putText(display, f"REC {sec:.1f}s",
                       (530, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            recorded_frames.append(frame.copy())

        cv2.putText(display, "S:Record Q:Quit",
                   (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.imshow("SLT Demo", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if not recording:
                recording = True
                recorded_frames = []
                print("Recording...")
            else:
                recording = False
                print(f"Stopped. {len(recorded_frames)} frames.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()
