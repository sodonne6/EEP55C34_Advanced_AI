from huggingface_hub import hf_hub_download

repo_id = "PSewmuthu/How2Sign_Holistic"
rar_file = "how2sign_holistic_features/val/frontal.rar"

p = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rar_file)
print("downloaded:", rar_file)
print("  ->", p)
