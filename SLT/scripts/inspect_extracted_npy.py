import glob
import numpy as np

paths = glob.glob(r"SLT\data_test\how2sign_holistic\features\*.npy")
print("found", len(paths), "npy files")

Ds = set()
for p in paths:
    x = np.load(p)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    print(p.split("\\")[-1], "->", x.shape, x.dtype)
    Ds.add(x.shape[1])

print("unique feature dims D:", sorted(Ds))