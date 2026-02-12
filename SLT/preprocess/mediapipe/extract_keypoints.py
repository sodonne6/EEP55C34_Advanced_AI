import argparse
from pathlib import Path
import cv2
import numpy as np

#"C:\Users\irish\Downloads\CzkLI34HFIg_20-5-rgb_front.mp4"

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pose subset for SLT (upper-body + tiny face + hips for centering)
POSE_UPPER11 = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_ALL33 = list(range(33))
HAND21 = 21

def get_pose_conf(lm):
    # Pose landmarks include visibility/presence (depending on model / output)
    if hasattr(lm, "visibility"):
        try: return float(lm.visibility)
        except Exception: pass
    if hasattr(lm, "presence"):
        try: return float(lm.presence)
        except Exception: pass
    return 1.0

def handedness_lr(handedness_list):
    # handedness_list is list[Category], take best
    try:
        best = handedness_list[0]
        return best.category_name, float(best.score)  # "Left"/"Right", score
    except Exception:
        return None, 0.0

def normalize_xy(kpts_xyc, pose_layout):
    """
    kpts_xyc: [N,3] in normalized image coords
    pose_layout: list of pose indices included at the front of tensor
    Normalization:
      - center at midpoint of hips if available else shoulders
      - scale by shoulder width if available else 1.0
    """
    out = kpts_xyc.copy()

    # Find indices (in our stacked tensor) for shoulders / hips if present
    # We stack: pose first, then left hand, then right hand.
    pose_len = len(pose_layout)
    def pose_pos(blazepose_idx):
        if blazepose_idx in pose_layout:
            return pose_layout.index(blazepose_idx)
        return None

    li = pose_pos(23)  # left hip
    ri = pose_pos(24)  # right hip
    ls = pose_pos(11)  # left shoulder
    rs = pose_pos(12)  # right shoulder

    if li is not None and ri is not None:
        center = 0.5 * (out[li, :2] + out[ri, :2])
    elif ls is not None and rs is not None:
        center = 0.5 * (out[ls, :2] + out[rs, :2])
    else:
        center = np.array([0.5, 0.5], dtype=np.float32)

    if ls is not None and rs is not None:
        scale = np.linalg.norm(out[ls, :2] - out[rs, :2])
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
    else:
        scale = 1.0

    out[:, 0] = (out[:, 0] - center[0]) / scale
    out[:, 1] = (out[:, 1] - center[1]) / scale
    return out

def extract_video(video_path: Path,
                  pose_model: Path,
                  hand_model: Path,
                  pose_mode: str,
                  normalize: bool,
                  max_frames: int,
                  preview: bool):

    if pose_mode == "upper11":
        pose_layout = POSE_UPPER11
    elif pose_mode == "all33":
        pose_layout = POSE_ALL33
    else:
        raise ValueError("pose_mode must be 'upper11' or 'all33'")

    pose_n = len(pose_layout)
    total_n = pose_n + 2 * HAND21  # pose + left + right

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    # Create Task objects (VIDEO mode)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    frames = []
    t = 0

    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if max_frames > 0 and t >= max_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # "By the book": monotonically increasing timestamps in ms for video mode
            ts_ms = int((t * 1000.0) / fps)

            pose_res = pose_lm.detect_for_video(mp_image, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_image, ts_ms)

            out = np.zeros((total_n, 3), dtype=np.float32)  # x,y,conf

            # ---- Pose (front block)
            if pose_res.pose_landmarks:
                p = pose_res.pose_landmarks[0]  # list of 33 landmarks
                for j, idx in enumerate(pose_layout):
                    lm = p[idx]
                    out[j, 0] = lm.x
                    out[j, 1] = lm.y
                    out[j, 2] = get_pose_conf(lm)

            # ---- Hands (fixed slots: left then right)
            left_base = pose_n
            right_base = pose_n + HAND21

            if hand_res.hand_landmarks and hand_res.handedness:
                for one_hand_lms, one_handness in zip(hand_res.hand_landmarks, hand_res.handedness):
                    lr, score = handedness_lr(one_handness)
                    if lr == "Left":
                        base = left_base
                    elif lr == "Right":
                        base = right_base
                    else:
                        continue

                    for k in range(min(HAND21, len(one_hand_lms))):
                        out[base + k, 0] = one_hand_lms[k].x
                        out[base + k, 1] = one_hand_lms[k].y
                        out[base + k, 2] = score  # handedness confidence

            if normalize:
                out = normalize_xy(out, pose_layout)

            frames.append(out)

            if preview:
                # quick dots so you can verify extraction visually
                h, w = bgr.shape[:2]
                for n in range(total_n):
                    conf = out[n, 2]
                    if conf <= 0:
                        continue
                    x = int((out[n, 0] if not normalize else (out[n, 0] * 0.25 + 0.5)) * w)
                    y = int((out[n, 1] if not normalize else (out[n, 1] * 0.25 + 0.5)) * h)
                    cv2.circle(bgr, (x, y), 2, (255, 255, 255), -1)
                cv2.imshow("preview (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            t += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    kpts = np.stack(frames, axis=0) if frames else np.zeros((0, total_n, 3), dtype=np.float32)

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "pose_mode": pose_mode,
        "layout": f"pose{pose_n}+left21+right21",
        "channels": "x,y,conf",
        "normalized": bool(normalize),
    }
    return kpts, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--pose_mode", choices=["upper11", "all33"], default="upper11")
    ap.add_argument("--normalize", action="store_true", help="center+scale coords using hips/shoulders")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--pose_model", type=str, default="models/pose_landmarker_full.task")
    ap.add_argument("--hand_model", type=str, default="models/hand_landmarker.task")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    video = Path(args.video).resolve()
    outp = Path(args.out).resolve()

    pose_model = (here / args.pose_model).resolve() if not Path(args.pose_model).is_absolute() else Path(args.pose_model)
    hand_model = (here / args.hand_model).resolve() if not Path(args.hand_model).is_absolute() else Path(args.hand_model)

    kpts, meta = extract_video(
        video_path=video,
        pose_model=pose_model,
        hand_model=hand_model,
        pose_mode=args.pose_mode,
        normalize=args.normalize,
        max_frames=args.max_frames,
        preview=args.preview
    )

    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, kpts=kpts, meta=str(meta))
    print(f"Saved: {outp}")
    print(f"kpts: {kpts.shape}  (T,N,3)")
    print(f"meta: {meta}")

if __name__ == "__main__":
    main()
