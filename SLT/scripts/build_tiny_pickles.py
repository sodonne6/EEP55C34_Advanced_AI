import os, glob, gzip, pickle
import numpy as np
import pandas as pd

meta_val = r"C:\Users\irish\Computer_Electronic_Engineering_Year5\semester_2\Advanced_AI\project\.hf\hub\datasets--PSewmuthu--How2Sign_Holistic\snapshots\96a0da665eba6e5bc9bd4bf6803546e74b222bbf\how2sign_holistic_features\metadata\how2sign_realigned_val.csv"
feat_dir = r"SLT\data_test\how2sign_holistic\features"
out_dir  = r"SLT\data_test\how2sign_holistic\pickles"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(meta_val, sep="\t").fillna("")
sent_by_name = dict(zip(df["SENTENCE_NAME"].astype(str), df["SENTENCE"].astype(str)))

items = []
paths = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))

for p in paths:
    fname = os.path.basename(p)
    base = fname[:-len("_holistic.npy")] if fname.endswith("_holistic.npy") else os.path.splitext(fname)[0]

    if base not in sent_by_name:
        print("NO MATCH:", fname, "-> tried base:", base)
        continue

    #x = np.load(p).astype(np.float32)          # already (T, 1629)
    x = np.load(p).astype(np.float32)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)  # (T, 543, 3) -> (T, 1629)
    items.append({"sign": x, "text": sent_by_name[base]})
    print("MATCH:", fname, "->", base, "sign_shape=", x.shape)

if len(items) < 2:
    raise RuntimeError("Not enough matched items. Check filenames and metadata path.")

train_items = items[:-1]
dev_items = items[-1:]

train_path = os.path.join(out_dir, "train.pkl.gz")
dev_path   = os.path.join(out_dir, "dev.pkl.gz")

with gzip.open(train_path, "wb") as f:
    pickle.dump(train_items, f)
with gzip.open(dev_path, "wb") as f:
    pickle.dump(dev_items, f)

print("\nWROTE:")
print(" train:", train_path, "n=", len(train_items))
print(" dev  :", dev_path,   "n=", len(dev_items))
print("Example sign shape:", train_items[0]["sign"].shape)
print("Example text:", train_items[0]["text"])
