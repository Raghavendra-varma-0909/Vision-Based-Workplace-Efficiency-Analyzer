# detector.py
"""
Employee Idle Detection — Camera-Selectable & Live Logging Version
------------------------------------------------------------------
Usage examples:
    python detector.py --employee_id E001
    python detector.py --video_source 1 --employee_id E001
    python detector.py --video office.mp4 --employee_id E002

Outputs:
 - live_status_log.csv  (updated every second)
 - idle_log.csv         (idle event summary)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import csv
import os
import sys
from collections import deque
from utils import log_idle
import datetime

# ----------------- Config Defaults -----------------
LIVE_LOG_PATH = "live_status_log.csv"
LIVE_LOG_HEADERS = ["timestamp", "employee_id", "status", "movement", "idle_seconds", "person_detected"]
FRAME_SAMPLE_RATE = 2
MOVEMENT_THRESHOLD = 15.0
IDLE_SECONDS_THRESHOLD = 30
MOV_WINDOW_SIZE = 5
# ---------------------------------------------------

mp_pose = mp.solutions.pose


def ensure_live_log():
    """Create live CSV if it doesn't exist."""
    if not os.path.exists(LIVE_LOG_PATH):
        with open(LIVE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LIVE_LOG_HEADERS)


def append_live_log(employee_id, status, movement, idle_s, person_detected):
    """Append one row to the live log (every second)."""
    with open(LIVE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            employee_id, status, round(movement, 2),
            round(idle_s, 2), int(person_detected)
        ])
        f.flush()
        os.fsync(f.fileno())


def get_landmark_positions(results, frame_w, frame_h, landmark_idxs=None):
    """Return flattened (x, y) coordinates for key landmarks."""
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    if landmark_idxs is None:
        landmark_idxs = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
        ]
    pts = []
    for idx in landmark_idxs:
        lm = landmarks[idx]
        x = lm.x * frame_w
        y = lm.y * frame_h
        pts.append((x, y))
    return np.array(pts).flatten()


def movement_magnitude(vec1, vec2):
    """Compute Euclidean distance."""
    if vec1 is None or vec2 is None:
        return np.inf
    return float(np.linalg.norm(vec1 - vec2))


def main(video_source=0, employee_id="E001", frame_sample_rate=FRAME_SAMPLE_RATE,
         movement_threshold=MOVEMENT_THRESHOLD, idle_seconds_threshold=IDLE_SECONDS_THRESHOLD,
         show_window=True):

    ensure_live_log()

    # --- Video source setup ---
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source ({video_source}).")
        print("👉 Try a different camera index using --video_source 1 or --video_source 2")
        return

    print(f"✅ Video source {video_source} opened successfully.\n")
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_vec = None
    idle_start_ts = None
    last_idle_log_ts = 0
    frame_idx = 0
    mov_window = deque(maxlen=MOV_WINDOW_SIZE)
    last_live_write = 0

    print("🟢 Starting detection... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n⚠️ No frame detected (end of video or camera issue).")
            break

        frame_idx += 1
        if frame_idx % frame_sample_rate != 0:
            if show_window:
                cv2.imshow("Feed (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        cur_vec = get_landmark_positions(results, w, h)
        mov = movement_magnitude(prev_vec, cur_vec) if prev_vec is not None else np.inf
        mov_window.append(mov)
        smooth_mov = float(np.mean(mov_window))
        timestamp = time.time()

        person_detected = smooth_mov != np.inf
        status_text = "UNKNOWN"
        idle_seconds_val = 0.0

        if not person_detected:
            status_text = "No person detected"
            idle_start_ts = None
        else:
            if smooth_mov < movement_threshold:
                if idle_start_ts is None:
                    idle_start_ts = timestamp
                idle_seconds_val = timestamp - idle_start_ts
                status_text = f"Idle ({int(idle_seconds_val)}s)"
                if idle_seconds_val >= idle_seconds_threshold and (timestamp - last_idle_log_ts) > idle_seconds_threshold:
                    log_idle(employee_id, idle_start_ts, timestamp,
                             notes=f"Auto-detected idle >= {idle_seconds_threshold}s")
                    last_idle_log_ts = timestamp
                    idle_start_ts = None
            else:
                status_text = "Active"
                idle_start_ts = None
                idle_seconds_val = 0.0

        # ---- Console output ----
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        sys.stdout.write(
            f"\r[{now_str}] Emp:{employee_id} | {status_text:<18} | "
            f"Mov:{smooth_mov:6.2f} | Idle_s:{int(idle_seconds_val):3d} | Person:{person_detected}"
        )
        sys.stdout.flush()

        # ---- Log every second ----
        if timestamp - last_live_write >= 1:
            append_live_log(employee_id, status_text, smooth_mov, idle_seconds_val, person_detected)
            last_live_write = timestamp

        prev_vec = cur_vec

        # ---- Video window ----
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if "Idle" in status_text else (0, 255, 0), 2)
        cv2.putText(frame, f"Movement: {smooth_mov:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        if show_window:
            cv2.imshow("Feed (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Stopping detection...")
                break

    pose.close()
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    print("\n✅ Detection finished.")
    print("🗂️ Logs saved to:")
    print(" - live_status_log.csv (live updates)")
    print(" - idle_log.csv (idle events)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to video file (optional)", default=None)
    ap.add_argument("--video_source", type=int, default=0,
                    help="Webcam index (0, 1, 2...)")
    ap.add_argument("--employee_id", default="E001", help="Employee ID for logs")
    ap.add_argument("--frame_sample_rate", type=int, default=FRAME_SAMPLE_RATE)
    ap.add_argument("--movement_threshold", type=float, default=MOVEMENT_THRESHOLD)
    ap.add_argument("--idle_seconds_threshold", type=int, default=IDLE_SECONDS_THRESHOLD)
    ap.add_argument("--no_window", action="store_true", help="Disable video window")
    args = ap.parse_args()

    source = args.video if args.video else args.video_source
    main(video_source=source,
         employee_id=args.employee_id,
         frame_sample_rate=args.frame_sample_rate,
         movement_threshold=args.movement_threshold,
         idle_seconds_threshold=args.idle_seconds_threshold,
         show_window=not args.no_window)
