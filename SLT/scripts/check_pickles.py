import gzip, pickle

p = r"SLT\data_test\how2sign_holistic\pickles\train.pkl.gz"
with gzip.open(p, "rb") as f:
    items = pickle.load(f)

print("items:", len(items))
print("keys:", list(items[0].keys()))
print("sign shape:", items[0]["sign"].shape, items[0]["sign"].dtype)
print("text:", items[0]["text"][:80])
