"""
Build combined manifests for the ResNet50 + MediaPipe setup.

Outputs:
- how2sign_resnet50_train.tsv  (train/frontal + train/side combined)
- how2sign_resnet50_val.tsv    (val/frontal)

TSV format (columns):
index	id	signs_file	signs_offset	signs_length	signs_type	signs_lang	translation	translation_lang	glosses	topic	signer_id	signs_mediapipe_file

Assumptions:
- ResNet features saved as: <id>_resnet.npy
- MediaPipe features saved as: <id>_mediapipe.npy
- ResNet arrays are typically (T, 2048, 1, 1) or (T, 2048)
- MediaPipe arrays are (T, 33, 3)
- Translation labels are in how2sign_realigned_{split}_normalised.csv (TSV-separated)
  with columns: SENTENCE_NAME and SENTENCE_NORMALIZED
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


COL_ORDER = [
    "index",
    "id",
    "signs_file",
    "signs_offset",
    "signs_length",
    "signs_type",
    "signs_lang",
    "translation",
    "translation_lang",
    "glosses",
    "topic",
    "signer_id",
    "signs_mediapipe_file",
]


TAIL_RE = re.compile(
    r"(?:[-_]rgb_(?:front|side)|[-_](?:front|side))?(?:_(?:resnet|mediapipe))$",
    re.IGNORECASE,
)


def normalize_id(stem_or_id: str) -> str:
    """Map front/side feature stems and metadata IDs to a shared clip key."""
    return TAIL_RE.sub("", stem_or_id)


def load_labels(labels_tsv: Path) -> Dict[str, str]:
    """
    Load SENTENCE_NAME -> SENTENCE_NORMALIZED mapping.
    File is named *.csv but is actually tab-separated in your setup.
    """
    df = pd.read_csv(labels_tsv, sep="\t")
    if "SENTENCE_NAME" not in df.columns or "SENTENCE_NORMALIZED" not in df.columns:
        raise ValueError(
            f"Expected columns SENTENCE_NAME and SENTENCE_NORMALIZED in {labels_tsv}, got {list(df.columns)}"
        )
    labels: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw_id = str(row["SENTENCE_NAME"])
        norm_id = normalize_id(raw_id)
        labels.setdefault(norm_id, str(row["SENTENCE_NORMALIZED"]))
    return labels


def list_feature_pairs(
    resnet_dir: Path,
    mp_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Return list of (id, resnet_path, mp_path) where both exist.
    Uses naming:
      <id>_resnet.npy
      <id>_mediapipe.npy
    """
    resnet_dir = resnet_dir.resolve()
    mp_dir = mp_dir.resolve()

    pairs: List[Tuple[str, Path, Path]] = []

    for resnet_path in sorted(resnet_dir.glob("*_resnet.npy")):
        _id = resnet_path.name.replace("_resnet.npy", "")
        mp_path = mp_dir / f"{_id}_mediapipe.npy"
        if mp_path.exists():
            pairs.append((_id, resnet_path, mp_path))

    return pairs


def infer_length_from_files(resnet_path: Path, mp_path: Path) -> int:
    """
    Robustly infer T from either file without loading entire arrays into RAM.
    Prefer MediaPipe length (should be exact).
    """
    try:
        mp = np.load(str(mp_path), mmap_mode="r")
        return int(mp.shape[0])
    except Exception:
        r = np.load(str(resnet_path), mmap_mode="r")
        return int(r.shape[0])


def build_manifest_rows(
    pairs: List[Tuple[str, Path, Path]],
    labels: Dict[str, str],
    allowed_norm_ids: Set[str] | None = None,
    require_label: bool = False,
) -> List[dict]:
    rows: List[dict] = []
    for _id, resnet_path, mp_path in pairs:
        norm_id = normalize_id(_id)

        if allowed_norm_ids is not None and norm_id not in allowed_norm_ids:
            continue

        length = infer_length_from_files(resnet_path, mp_path)
        translation = labels.get(norm_id, "")

        if require_label and translation == "":
            continue

        rows.append(
            {
                "index": len(rows),
                "id": _id,
                "signs_file": str(resnet_path),
                "signs_offset": 0,
                "signs_length": length,
                "signs_type": "resnet50",
                "signs_lang": "asl",
                "translation": translation,
                "translation_lang": "en",
                "glosses": "",
                "topic": "",
                "signer_id": "",
                "signs_mediapipe_file": str(mp_path),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base_path",
        type=Path,
        default=Path("/content/drive/MyDrive/AAI_project"),
        help="Base project path (default: /content/drive/MyDrive/AAI_project)",
    )

    ap.add_argument(
        "--labels_train",
        type=Path,
        default=Path("/content/drive/MyDrive/AAI_project/manifests/SLT/how2sign/how2sign_realigned_train_normalised.csv"),
        help="Train labels file (tab-separated despite .csv extension)",
    )
    ap.add_argument(
        "--labels_val",
        type=Path,
        default=Path("/content/drive/MyDrive/AAI_project/manifests/SLT/how2sign/how2sign_realigned_val_normalised.csv"),
        help="Val labels file (tab-separated despite .csv extension)",
    )

    ap.add_argument(
        "--out_train",
        type=Path,
        default=Path("/content/drive/MyDrive/AAI_project/manifests/SLT/how2sign/how2sign_resnet50_train.tsv"),
        help="Output train manifest TSV",
    )
    ap.add_argument(
        "--out_val",
        type=Path,
        default=Path("/content/drive/MyDrive/AAI_project/manifests/SLT/how2sign/how2sign_resnet50_val.tsv"),
        help="Output val manifest TSV",
    )

    args = ap.parse_args()

    base = args.base_path.resolve()

    # Feature directories
    train_frontal_resnet = base / "data/SLT/how2sign/clips/train/frontal/features/resnet_features"
    train_frontal_mp = base / "data/SLT/how2sign/clips/train/frontal/features/mediapipe_features"

    train_side_resnet = base / "data/SLT/how2sign/clips/train/side/features/resnet_features"
    train_side_mp = base / "data/SLT/how2sign/clips/train/side/features/mediapipe_features"

    val_frontal_resnet = base / "data/SLT/how2sign/clips/val/frontal/features/resnet_features"
    val_frontal_mp = base / "data/SLT/how2sign/clips/val/frontal/features/mediapipe_features"

    # Basic existence checks
    for p in [
        train_frontal_resnet,
        train_frontal_mp,
        train_side_resnet,
        train_side_mp,
        val_frontal_resnet,
        val_frontal_mp,
        args.labels_train,
        args.labels_val,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing path: {p}")

    print("[INFO] Loading labels...")
    labels_train = load_labels(args.labels_train)
    labels_val = load_labels(args.labels_val)
    print(f"[INFO] Train labels: {len(labels_train)}")
    print(f"[INFO] Val labels:   {len(labels_val)}")

    print("[INFO] Listing train/frontal feature pairs...")
    train_frontal_pairs = list_feature_pairs(train_frontal_resnet, train_frontal_mp)
    print(f"[INFO] train/frontal pairs: {len(train_frontal_pairs)}")

    train_frontal_norm_ids = {normalize_id(_id) for _id, _, _ in train_frontal_pairs}
    print(f"[INFO] train/frontal normalized IDs: {len(train_frontal_norm_ids)}")

    print("[INFO] Listing train/side feature pairs...")
    train_side_pairs = list_feature_pairs(train_side_resnet, train_side_mp)
    print(f"[INFO] train/side pairs: {len(train_side_pairs)}")

    train_side_norm_ids = {normalize_id(_id) for _id, _, _ in train_side_pairs}
    overlap_norm_ids = train_frontal_norm_ids & train_side_norm_ids
    print(f"[INFO] train front/side normalized overlap IDs: {len(overlap_norm_ids)}")

    print("[INFO] Building combined train manifest...")
    train_front_rows = build_manifest_rows(train_frontal_pairs, labels_train)
    train_side_rows = build_manifest_rows(
        train_side_pairs,
        labels_train,
        allowed_norm_ids=train_frontal_norm_ids,
        require_label=True,
    )
    train_rows = train_front_rows + train_side_rows
    print(f"[INFO] train/frontal rows kept: {len(train_front_rows)}")
    print(f"[INFO] train/side rows kept (overlap + labeled): {len(train_side_rows)}")
    train_df = pd.DataFrame(train_rows)
    train_df["index"] = range(len(train_df))
    train_df = train_df[COL_ORDER]

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_train, sep="\t", index=False)
    print(f"[INFO] Wrote train manifest: {args.out_train} (rows={len(train_df)})")

    print("[INFO] Listing val/frontal feature pairs...")
    val_pairs = list_feature_pairs(val_frontal_resnet, val_frontal_mp)
    print(f"[INFO] val/frontal pairs: {len(val_pairs)}")

    print("[INFO] Building val manifest...")
    val_rows = build_manifest_rows(val_pairs, labels_val)
    val_df = pd.DataFrame(val_rows)
    val_df["index"] = range(len(val_df))
    val_df = val_df[COL_ORDER]

    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.out_val, sep="\t", index=False)
    print(f"[INFO] Wrote val manifest: {args.out_val} (rows={len(val_df)})")

    # Quick sanity checks
    missing_train_labels = (train_df["translation"] == "").sum()
    missing_val_labels = (val_df["translation"] == "").sum()
    print(f"[INFO] Train rows missing translation: {missing_train_labels}")
    print(f"[INFO] Val rows missing translation:   {missing_val_labels}")


if __name__ == "__main__":
    main()