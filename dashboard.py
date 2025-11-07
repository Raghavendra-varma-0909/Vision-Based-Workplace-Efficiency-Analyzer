# detector.py
"""
Employee Idle Detection — Real-time Logging Version
---------------------------------------------------
Every second, this script writes current status to 'live_status_log.csv'
and logs idle events separately to 'idle_log.csv'.

Usage:
    python detector.py --employee_id E001
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque
from utils import log_idle
import sys
import datetime
import csv
import os

mp_pose = mp.solutions.pose

LIVE_LOG_PATH = "live_status_log.csv"

def ensure_live_log():
    """Ensure live status log CSV exists."""
    if not os.path.exists(LIVE_LOG_PATH):
        with open(LIVE_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "employee_id", "status", "movement"])

def append_live_log(employee_id, status, movement):
    """Append one line of live status data."""
    with open(LIVE_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            employee_id, status, round(movement, 2)
        ])

def get_landmark_positions(results, frame_w, frame_h, landmark_idxs=None):
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
    if vec1 is None or vec2 is None:
        return np.inf
    return np.linalg.norm(vec1 - vec2)

def main(video_source=0, employee_id="E001", frame_sample_rate=2,
         movement_threshold=15.0, idle_seconds_threshold=30, show_window=True):

    ensure_live_log()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ ERROR: Could not open video source:", video_source)
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_vec = None
    idle_start_ts = None
    last_log_ts = 0
    frame_idx = 0
    mov_window = deque(maxlen=5)
    last_live_write = 0  # Track when we last wrote to CSV

    print("✅ Starting detection... Press 'q' to quit.\n")

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
        smooth_mov = np.mean(mov_window)

        timestamp = time.time()
        status_text = "UNKNOWN"
        color = (0, 255, 255)

        # ----- ACTIVITY LOGIC -----
        if smooth_mov == np.inf:
            status_text = "No person detected"
            idle_start_ts = None
        else:
            if smooth_mov < movement_threshold:
                if idle_start_ts is None:
                    idle_start_ts = timestamp
                idle_duration = timestamp - idle_start_ts
                status_text = f"Idle ({int(idle_duration)}s)"
                color = (0, 0, 255)
                if idle_duration >= idle_seconds_threshold:
                    if timestamp - last_log_ts > idle_seconds_threshold:
                        log_idle(employee_id, idle_start_ts, timestamp,
                                 notes=f"Auto-detected idle for {int(idle_duration)}s")
                        last_log_ts = timestamp
                        idle_start_ts = None
            else:
                status_text = "Active"
                color = (0, 255, 0)
                idle_start_ts = None

        # ---- LIVE CONSOLE OUTPUT ----
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        sys.stdout.write(
            f"\r[{now_str}] Employee: {employee_id} | Status: {status_text:<18} "
            f"| Movement: {smooth_mov:6.2f} | Thresholds: mov<{movement_threshold}, idle>{idle_seconds_threshold}s"
        )
        sys.stdout.flush()

        # ---- WRITE LIVE DATA EVERY 1 SECOND ----
        if timestamp - last_live_write >= 1:
            append_live_log(employee_id, status_text, smooth_mov)
            last_live_write = timestamp

        prev_vec = cur_vec

        # ---- VIDEO DISPLAY ----
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Movement: {smooth_mov:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        if show_window:
            cv2.imshow("Feed (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Exiting detection...")
                break

    pose.close()
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    print("\n✅ Detection finished. Logs saved to:")
    print(" - idle_log.csv (idle events)")
    print(" - live_status_log.csv (1-second live data)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to video file. If omitted, uses webcam 0", default=None)
    ap.add_argument("--employee_id", help="Employee ID for logs", default="E001")
    ap.add_argument("--frame_sample_rate", type=int, default=2)
    ap.add_argument("--movement_threshold", type=float, default=15.0)
    ap.add_argument("--idle_seconds_threshold", type=int, default=30)
    ap.add_argument("--no_window", action="store_true", help="Disable video window")
    args = ap.parse_args()

    source = args.video if args.video else 0
    main(video_source=source,
         employee_id=args.employee_id,
         frame_sample_rate=args.frame_sample_rate,
         movement_threshold=args.movement_threshold,
         idle_seconds_threshold=args.idle_seconds_threshold,
         show_window=not args.no_window)
