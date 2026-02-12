from pathlib import Path
import urllib.request

# Official pre-trained bundles (Pose + Hand)
POSE_FULL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
POSE_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def dl(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n-> {out}")
    urllib.request.urlretrieve(url, out)

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    models = here / "models"

    # Choose one:
    dl(POSE_FULL, models / "pose_landmarker_full.task")
    # dl(POSE_LITE, models / "pose_landmarker_lite.task")

    dl(HAND, models / "hand_landmarker.task")
    print("Done.")
