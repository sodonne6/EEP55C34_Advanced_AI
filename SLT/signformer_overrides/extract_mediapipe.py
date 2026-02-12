#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np

import cv2

import itertools


# --- MediaPipe compatibility patch for pose-format ---
import mediapipe as mp
try:
    # Newer MediaPipe often keeps solutions here
    from mediapipe.python import solutions as mp_solutions
except Exception:
    # Some builds expose this layout
    import mediapipe.solutions as mp_solutions

# Attach so pose_format.utils.holistic can do: mp.solutions.holistic
mp.solutions = mp_solutions

from pose_format.utils.holistic import load_holistic


#from pose_format.utils.holistic import load_holistic #We want to use our own implementation
#import sys
#sys.path.insert(1, '/home/usuaris/imatge/ltarres/02_EgoSign/visualization/poseformat/') #This path will need to change if we try to run it in Amada's user
#from pose_format.utils.holistic import load_holistic


def extract_poses(vid_file: Path, target_fps: int):
    video = cv2.VideoCapture(str(vid_file))
    try:
        raw_fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_int = int(round(raw_fps)) if raw_fps > 0 else int(target_fps)
        print(f"Video FPS raw: {raw_fps:.3f} -> stored fps: {fps_int}")

        # Try to get W/H from container, but fall back to first frame
        W0 = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H0 = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, first = video.read()
        if not ok or first is None:
            raise RuntimeError(f"No frames read from {vid_file}")

        def frame_iter(cap):
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame

        # STREAM frames (no big frames[] list)
        frames = itertools.chain([first], frame_iter(video))

        poses = load_holistic(
            frames,
            fps=fps_int,  # MUST be int
            width=W0 or first.shape[1],
            height=H0 or first.shape[0],
            depth=10,
            progress=True,
            additional_holistic_config={
                "min_detection_confidence": 0.2,
                "min_tracking_confidence": 0.3,
            },
        )

        # Use first frame dims for pixel-space clamping
        H, W = first.shape[:2]
        data = poses.body.data
        conf = poses.body.confidence

        start = 0
        for comp in poses.header.components:
            pts = comp.points
            n = len(pts) if isinstance(pts, (list, tuple)) else int(pts)
            name = getattr(comp, "name", "").upper()

            # 1) Kill world landmarks entirely (prevents "second person" in viewers)
            if name == "POSE_WORLD_LANDMARKS":
                conf[..., start:start+n] = 0.0
                start += n
                continue

            # 2) For pixel-space components, clamp out-of-bounds only
            x = data[..., start:start+n, 0]
            y = data[..., start:start+n, 1]
            c = conf[..., start:start+n]

            oob = (x < 0) | (x >= W) | (y < 0) | (y >= H)
            c[oob] = 0.0
            data[..., start:start+n, 0][oob] = np.clip(x[oob], 0, W - 1)
            data[..., start:start+n, 1][oob] = np.clip(y[oob], 0, H - 1)

            # Optional: threshold ONLY hands/face if you want
            if ("HAND" in name) or ("FACE" in name):
                low = c < 0.10
                c[low] = 0.0

            conf[..., start:start+n] = c
            start += n

        poses.body.data = data
        poses.body.confidence = conf

        # Interpolate if needed (decision based on raw fps)
        if target_fps is not None and abs(raw_fps - target_fps) > 0.25:
            poses = poses.interpolate(int(target_fps), kind="linear")

        return poses

    finally:
        video.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-file', type=str, required=True, help="Input video file (MP4)")
    parser.add_argument('--poses-file', type=str, required=True, help="File where the extracted poses will be saved")
    parser.add_argument('--fps', type=int, default=24, help="Target FPS for poses")
    return parser.parse_args()


def main():

    args = parse_args()

    video_file = Path(args.video_file).expanduser().resolve()
    poses_file = Path(args.poses_file).expanduser().resolve()

    assert video_file.is_file(), "The input file does not exist"
    poses_file.parent.mkdir(parents=True, exist_ok=True)
    poses = extract_poses(video_file, args.fps)
    with open(poses_file.as_posix(), "wb") as f:
        poses.write(f)


if __name__ == '__main__':
    main()
