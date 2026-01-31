from huggingface_hub import list_repo_files

repo_id = "PSewmuthu/How2Sign_Holistic"
files = list_repo_files(repo_id=repo_id, repo_type="dataset")

print("Total files:", len(files))

print("\nMetadata CSV candidates:")
for f in files:
    if "metadata" in f and f.endswith(".csv"):
        print("  ", f)

print("\nExample .npy candidates (first 20 frontal):")
n = 0
for f in files:
    if f.endswith(".npy") and ("frontal" in f or "front" in f):
        print("  ", f)
        n += 1
        if n >= 20:
            break
