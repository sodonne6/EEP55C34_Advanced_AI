import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
i3d_repo = repo_root / "SLT" / "external" / "pytorch-i3d"
sys.path.insert(0, str(i3d_repo))

import torch
from pytorch_i3d import InceptionI3d

print("Imported InceptionI3d OK")

# Instantiate (400 classes is typical for Kinetics-400 checkpoints)
model = InceptionI3d(num_classes=400, in_channels=3)
model.eval()

# Dummy video tensor: (B, C, T, H, W)
x = torch.randn(1, 3, 16, 224, 224)
with torch.no_grad():
    y = model(x)

print("Forward OK. Output shape:", tuple(y.shape))
