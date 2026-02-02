import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import cv2

# Locate repo root robustly:
# .../project/SLT/scripts/i3d/extract_i3d_features_from_mp4.py
REPO_ROOT = Path(__file__).resolve().parents[3]
I3D_DIR = REPO_ROOT / "SLT" / "external" / "pytorch-i3d"
sys.path.insert(0, str(I3D_DIR))

from pytorch_i3d import InceptionI3d

CKPT = I3D_DIR / "models" / "rgb_imagenet.pt"

def read_all_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video.")
    return frames

def uniform_sample(frames, target_T: int):
    n = len(frames)
    if n <= target_T:
        return frames
    idx = np.linspace(0, n - 1, target_T).round().astype(int)
    return [frames[i] for i in idx]

def preprocess(frames, size=224):
    # Resize to 224x224, normalize to [-1, 1]
    proc = []
    for f in frames:
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        proc.append(f)
    arr = np.stack(proc, axis=0).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [-1, 1]
    # (T,H,W,C) -> (C,T,H,W)
    arr = np.transpose(arr, (3, 0, 1, 2))
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to .mp4")
    ap.add_argument("--out", required=True, help="Output .npy path")
    ap.add_argument("--frames", type=int, default=64, help="Number of frames to sample uniformly")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + checkpoint
    model = InceptionI3d(num_classes=400, in_channels=3).to(device).eval()
    state = torch.load(str(CKPT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    # Read + sample + preprocess
    frames = read_all_frames(str(video_path))
    sampled = uniform_sample(frames, args.frames)
    x_np = preprocess(sampled, size=224)              # (3,T,H,W)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,3,T,H,W)

    with torch.no_grad():
        feat = model.extract_features(x)  # (1,1024,T',1,1)
        seq = feat.mean(dim=[3, 4]).squeeze(0).transpose(0, 1).contiguous()  # (T',1024)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, seq.detach().cpu().numpy().astype(np.float32))

    print("Video frames read:", len(frames))
    print("Frames sampled:", len(sampled))
    print("extract_features shape:", tuple(feat.shape))
    print("Saved seq shape:", tuple(seq.shape), "->", out_path)

if __name__ == "__main__":
    main()
