import os
import sys
from pathlib import Path
import torch

# Add repo root to sys.path (…/project)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Add pytorch-i3d folder to path and import
I3D_DIR = REPO_ROOT / "SLT" / "external" / "pytorch-i3d"
sys.path.insert(0, str(I3D_DIR))

from pytorch_i3d import InceptionI3d

CKPT = REPO_ROOT / "SLT" / "external" / "pytorch-i3d" / "models" / "rgb_imagenet.pt"

def main():
    assert CKPT.exists(), f"Checkpoint not found: {CKPT}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InceptionI3d(num_classes=400, in_channels=3).to(device)
    model.eval()

    state = torch.load(str(CKPT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Loaded:", CKPT)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    x = torch.randn(1, 3, 32, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)

    print("Forward OK. Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
