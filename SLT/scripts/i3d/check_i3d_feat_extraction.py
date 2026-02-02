import sys
from pathlib import Path
import torch

# We are in: .../project/SLT/scripts/i3d/check_i3d_feat_extraction.py
# parents[0]=i3d, [1]=scripts, [2]=SLT, [3]=project
REPO_ROOT = Path(__file__).resolve().parents[3]
I3D_DIR = REPO_ROOT / "SLT" / "external" / "pytorch-i3d"
sys.path.insert(0, str(I3D_DIR))

from pytorch_i3d import InceptionI3d  # pytorch_i3d.py inside that folder

CKPT = I3D_DIR / "models" / "rgb_imagenet.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InceptionI3d(num_classes=400, in_channels=3).to(device).eval()
    state = torch.load(str(CKPT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    x = torch.randn(1, 3, 64, 224, 224, device=device)  # (B,C,T,H,W)

    with torch.no_grad():
        if not hasattr(model, "extract_features"):
            raise RuntimeError("This InceptionI3d class has no extract_features(). "
                               "Open pytorch_i3d.py or use external/pytorch-i3d/extract_features.py logic.")

        feat = model.extract_features(x)

    print("extract_features output shape:", tuple(feat.shape))
    # Typical: (B, 1024, T', 1, 1)  (exact T' depends on stride)

    if feat.dim() == 5:
        seq = feat.mean(dim=[3,4]).squeeze(0).transpose(0, 1).contiguous()  # (T', 1024)
        print("decoder sequence shape:", tuple(seq.shape))

if __name__ == "__main__":
    main()
