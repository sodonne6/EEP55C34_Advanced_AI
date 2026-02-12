# batch_video_to_pose.py
#
# Batch-convert raw How2Sign rgb_front videos -> pose-format .pose files
# AND write the metadata TSV as examples/sign_language/datasets/how2sign.py expects:
#   How2Sign/metadata/cvpr23.mediapipe.<split>.how2sign.tsv
#   How2Sign/video_level/<split>/rgb_front/features/mediapipe/<mapped_id>.pose
#
# Key properties:
# - Resume-safe: if a .pose exists and is readable, it skips extraction
# - Robust Drive I/O: temp-file write + atomic replace + retries
# - Skip-on-error: logs errors to *.errors.tsv and continues

import argparse
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions


# IMPORTANT: collision-free mapping (keep group2!)
_FIND_GROUPS = r"(^.{11})(.*)-([0-9]+)(.*)"
_PRINT_GROUPS = r"\1\2-\3\4"


def map_sentence_name_to_pose_stem(sentence_name: str) -> str:
    return re.sub(_FIND_GROUPS, _PRINT_GROUPS, sentence_name)


def pick_columns(df: pd.DataFrame, id_col: Optional[str], text_col: Optional[str]) -> Tuple[str, str]:
    id_candidates = [id_col, "clip_id", "CLIP_ID", "id", "ID", "SENTENCE_NAME", "sentence_name"]
    txt_candidates = [
        text_col,
        "SENTENCE_NORMALIZED",
        "sentence_normalized",
        "SENTENCE",
        "sentence",
        "translation",
        "TRANSLATION",
    ]

    id_pick = next((c for c in id_candidates if c and c in df.columns), None)
    txt_pick = next((c for c in txt_candidates if c and c in df.columns), None)
    if id_pick is None or txt_pick is None:
        raise ValueError(
            "Could not find required columns.\n"
            f"Columns present: {list(df.columns)}\n"
            "Pass --id-col and --text-col explicitly if needed."
        )
    return id_pick, txt_pick


def safe_read_pose_len(pose_path: Path) -> Optional[int]:
    try:
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
        return int(pose.body.data.shape[0])
    except Exception:
        return None


def safe_write_pose(pose: Pose, pose_path: Path, retries: int = 5) -> bool:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            pose_path.parent.mkdir(parents=True, exist_ok=True)

            tmp = pose_path.with_suffix(pose_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pose.write(f)

            tmp.replace(pose_path)  # atomic replace on same filesystem
            return True
        except (FileNotFoundError, OSError) as e:
            last_err = e
            time.sleep(1.5 * attempt)

    print(f"[WARN] Failed to write pose after {retries} retries: {pose_path} | {last_err}")
    return False


def build_component(name: str, points: list[str], limbs: list[tuple[int, int]]) -> PoseHeaderComponent:
    return PoseHeaderComponent(
        name=name,
        points=points,
        limbs=limbs,
        colors=[(255, 0, 0)],
        point_format="XYZC",
    )


def extract_video_to_pose(
    video_path: Path,
    pose_mode: str = "upper11",
    max_frames: int = -1,
    min_detection_conf: float = 0.5,
    min_tracking_conf: float = 0.5,
) -> Pose:
    mp_h = mp.solutions.holistic

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read any frames: {video_path}")
        h, w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    BODY_POINTS = mp_h.PoseLandmark._member_names_
    BODY_LIMBS = [(int(a), int(b)) for a, b in mp_h.POSE_CONNECTIONS]
    HAND_POINTS = mp_h.HandLandmark._member_names_
    HAND_LIMBS = [(int(a), int(b)) for a, b in mp_h.HAND_CONNECTIONS]

    if pose_mode == "all33":
        pose_names = list(BODY_POINTS)
        pose_indices = list(range(33))
        pose_limb_src = BODY_LIMBS
    elif pose_mode == "upper11":
        pose_names = [
            "NOSE",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_EAR", "RIGHT_EAR",
        ]
        pose_indices = [getattr(mp_h.PoseLandmark, n).value for n in pose_names]
        subset_map = {old_i: new_i for new_i, old_i in enumerate(pose_indices)}
        pose_limb_src = [(subset_map[a], subset_map[b]) for a, b in BODY_LIMBS if a in subset_map and b in subset_map]
    else:
        raise ValueError("--pose-mode must be 'upper11' or 'all33'")

    components = [
        build_component("POSE_LANDMARKS", pose_names, pose_limb_src),
        build_component("LEFT_HAND_LANDMARKS", list(HAND_POINTS), HAND_LIMBS),
        build_component("RIGHT_HAND_LANDMARKS", list(HAND_POINTS), HAND_LIMBS),
    ]

    header = PoseHeader(
        version=0.1,
        dimensions=PoseHeaderDimensions(width=w, height=h, depth=1),
        components=components,
    )

    total_points = sum(len(c.points) for c in components)

    datas, confs = [], []

    with mp_h.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf,
    ) as holistic:
        frame_i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_i += 1
            if max_frames > 0 and frame_i > max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(frame_rgb)

            pose_xyz = np.zeros((len(pose_indices), 3), dtype=np.float32)
            pose_c = np.zeros((len(pose_indices),), dtype=np.float32)
            if res.pose_landmarks is not None:
                lm = res.pose_landmarks.landmark
                for j, idx in enumerate(pose_indices):
                    p = lm[idx]
                    pose_xyz[j, 0] = p.x * w
                    pose_xyz[j, 1] = p.y * h
                    pose_xyz[j, 2] = p.z
                    pose_c[j] = float(getattr(p, "visibility", 1.0))

            def hand_block(hand_lms):
                xyz = np.zeros((21, 3), dtype=np.float32)
                c = np.zeros((21,), dtype=np.float32)
                if hand_lms is not None:
                    lm2 = hand_lms.landmark
                    for j in range(21):
                        p2 = lm2[j]
                        xyz[j, 0] = p2.x * w
                        xyz[j, 1] = p2.y * h
                        xyz[j, 2] = p2.z
                        c[j] = 1.0
                return xyz, c

            lh_xyz, lh_c = hand_block(res.left_hand_landmarks)
            rh_xyz, rh_c = hand_block(res.right_hand_landmarks)

            xyz = np.concatenate([pose_xyz, lh_xyz, rh_xyz], axis=0)
            c = np.concatenate([pose_c, lh_c, rh_c], axis=0)

            if xyz.shape[0] != total_points:
                raise RuntimeError(f"Point count mismatch: got {xyz.shape[0]} expected {total_points}")

            datas.append(xyz)
            confs.append(c)

    cap.release()

    if not datas:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    data = np.expand_dims(np.stack(datas, axis=0), axis=1)  # (T,1,N,3)
    conf = np.expand_dims(np.stack(confs, axis=0), axis=1)  # (T,1,N)

    body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
    return Pose(header=header, body=body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--pose-mode", default="upper11", choices=["upper11", "all33"])
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--topic-col", default=None)
    ap.add_argument("--max-frames", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    csv_path = Path(args.csv)
    video_root = Path(args.video_root)

    out_pose_dir = base_dir / "How2Sign" / "video_level" / args.split / "rgb_front" / "features" / "mediapipe"
    out_pose_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = base_dir / "How2Sign" / "metadata" / f"cvpr23.mediapipe.{args.split}.how2sign.tsv"
    out_err = base_dir / "How2Sign" / "metadata" / f"cvpr23.mediapipe.{args.split}.how2sign.errors.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=None, engine="python")
    id_col, text_col = pick_columns(df, args.id_col, args.text_col)
    topic_col = args.topic_col if args.topic_col and args.topic_col in df.columns else None

    rows = []
    missing = 0
    written = 0
    skipped = 0
    failed = 0
    err_rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Extract {args.split}"):
        sent_name = str(r[id_col]).strip()
        sentence = str(r[text_col]).strip()

        video_path = video_root / f"{sent_name}.mp4"
        if not video_path.exists():
            cand = list(video_root.glob(f"{sent_name}.*"))
            if not cand:
                missing += 1
                continue
            video_path = cand[0]

        pose_stem = map_sentence_name_to_pose_stem(sent_name)
        pose_path = out_pose_dir / f"{pose_stem}.pose"

        try:
            T = None
            if pose_path.exists() and not args.overwrite:
                T = safe_read_pose_len(pose_path)

            if T is not None:
                skipped += 1
            else:
                pose = extract_video_to_pose(
                    video_path=video_path,
                    pose_mode=args.pose_mode,
                    max_frames=args.max_frames,
                )
                T = int(pose.body.data.shape[0])
                ok = safe_write_pose(pose, pose_path)
                if not ok:
                    failed += 1
                    err_rows.append({
                        "SENTENCE_NAME": sent_name,
                        "VIDEO_PATH": str(video_path),
                        "POSE_PATH": str(pose_path),
                        "ERROR": "failed to write pose (after retries)"
                    })
                    continue
                written += 1

            topic = int(r[topic_col]) if topic_col else 0
            rows.append({
                "SENTENCE_NAME": sent_name,
                "START_FRAME": 0,
                "END_FRAME": T - 1,
                "SENTENCE": sentence,
                "TOPIC_ID": topic,
            })

        except Exception as e:
            failed += 1
            err_rows.append({
                "SENTENCE_NAME": sent_name,
                "VIDEO_PATH": str(video_path),
                "POSE_PATH": str(pose_path),
                "ERROR": repr(e)
            })
            continue

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv(out_err, sep="\t", index=False)

    print("\n[DONE]")
    print(f"Pose dir : {out_pose_dir}")
    print(f"TSV      : {out_tsv}")
    print(f"Errors   : {out_err}")
    print(f"written={written} skipped={skipped} missing={missing} failed={failed}")


if __name__ == "__main__":
    main()
