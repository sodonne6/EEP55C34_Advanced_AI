from huggingface_hub import hf_hub_download

repo_id = "PSewmuthu/How2Sign_Holistic"

meta_files = [
  "how2sign_holistic_features/metadata/how2sign_realigned_train.csv",
  "how2sign_holistic_features/metadata/how2sign_realigned_val.csv",
  "how2sign_holistic_features/metadata/how2sign_realigned_test.csv",
]

for f in meta_files:
    p = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f)
    print("downloaded:", f)
    print("  ->", p)

