import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from pose_format import Pose

from fairseq.data import FairseqDataset, BaseWrapperDataset, RandomCropDataset
from fairseq.data.data_utils import compute_mask_indices, numpy_seed
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)


class SignFeatsType(str, Enum):
    mediapipe = "mediapipe"
    openpose = "openpose"
    i3d = "i3d"
    CNN2d = "CNN2d"


class NormType(str, Enum):
    body = "body"
    kp_wise = "kp_wise"
    global_xyz = "global_xyz"
    normalize = "normalize"  # to add the same normalizaiton as original TD


def _as_featstype(x) -> SignFeatsType:
    """
    Normalize feats_type into a SignFeatsType enum safely.
    Accepts:
      - SignFeatsType enum
      - string like "mediapipe"
    """
    if isinstance(x, SignFeatsType):
        return x
    if isinstance(x, str):
        # allow either "mediapipe" or "SignFeatsType.mediapipe"-ish
        x = x.strip()
        # If it's already the value:
        for v in SignFeatsType:
            if x == v.value:
                return v
        # If it's a key:
        try:
            return SignFeatsType[x]
        except Exception:
            pass
    raise ValueError(f"Unknown feats_type={x} (type={type(x)})")


class SignFeatsDataset(FairseqDataset):
    def __init__(
        self,
        ids: List[str],
        feats_files: List[Union[Path, str]],
        feats_mediapipe_files: List[Union[Path, str]],
        offsets: List[int],
        sizes: List[int],
        feats_type: SignFeatsType,
        normalization: NormType = NormType.body,
        data_augmentation: bool = False,
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
    ):
        super().__init__()
        assert len(ids) == len(feats_files) == len(offsets) == len(sizes) == len(feats_mediapipe_files)

        self.ids = ids
        self.feats_files = feats_files
        self.feats_mediapipe_files = feats_mediapipe_files
        self.offsets = offsets
        self.sizes = sizes

        # normalize to enum once (important: avoids string/enum bugs)
        self.feats_type: SignFeatsType = _as_featstype(feats_type)

        self.normalization = normalization
        self.data_augmentation = data_augmentation
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.shuffle = shuffle
        self.skipped_ids = []

    def filter_by_length(self, min_sample_size, max_sample_size):
        # NOTE: This mutates lists while iterating. Keep as repo behavior (works for small sets),
        # but we copy ids/sizes for iteration.
        for _id, size in zip(self.ids[:], self.sizes[:]):
            if size < self.min_sample_size or size > self.max_sample_size:
                idx = self.ids.index(_id)
                self.feats_files.pop(idx)
                self.feats_mediapipe_files.pop(idx)
                self.offsets.pop(idx)
                self.sizes.pop(idx)
                self.ids.remove(_id)
                self.skipped_ids.append(_id)
        logger.info(f"Filtered {len(self.skipped_ids)} sentences, that were too short or too long.")

    @classmethod
    def from_manifest_file(cls, manifest_file: Union[str, Path], **kwargs):
        ids = []
        feats_files = []
        feats_mediapipe_files = []
        offsets = []
        sizes = []

        manifest = pd.read_csv(manifest_file, sep="\t")
        for _, row in manifest.iterrows():
            ids.append(row["id"])
            feats_files.append(row["signs_file"])
            feats_mediapipe_files.append(row.get("signs_mediapipe_file", row["signs_file"]))
            offsets.append(int(row["signs_offset"]))
            sizes.append(int(row["signs_length"]))

        logger.info(f"loaded {len(ids)} samples")

        # Use the first row (manifest could have multiple rows)
        feats_type = manifest["signs_type"].iloc[0]
        return cls(
            ids,
            feats_files=feats_files,
            feats_mediapipe_files=feats_mediapipe_files,
            offsets=offsets,
            sizes=sizes,
            feats_type=feats_type,
            **kwargs,
        )

    def __getitem__(self, index):
        _id = self.ids[index]
        feats_file = self.feats_files[index]
        feats_mediapipe_file = self.feats_mediapipe_files[index]
        offset = self.offsets[index]
        length = self.sizes[index]

        # ------------------------------------------------------------
        # CASE A) feats_type == mediapipe
        # Support BOTH:
        #   - .pose (pose_format Pose)
        #   - .npy (holistic array, e.g. (T, 543, 3))
        # ------------------------------------------------------------
        if self.feats_type == SignFeatsType.mediapipe:
            feats_file_str = str(feats_file)

            # A1) .pose path (old behavior)
            if feats_file_str.endswith(".pose"):
                with open(feats_file, "rb") as f:
                    pose = Pose.read(f.read())

                frames_list = list(range(offset, offset + length))
                frames_list = [fr for fr in frames_list if fr < pose.body.data.shape[0]]
                pose.body = pose.body.select_frames(frames_list)

                pose = self.postprocess_pose(pose)
                return {"id": index, "vid_id": _id, "source": pose}

            # A2) .npy holistic features (NEW behavior)
            with open(feats_file, "rb") as f:
                arr = np.load(f)

            # Slice frames like the .pose path does
            T = arr.shape[0]
            end = min(offset + length, T)
            arr = arr[offset:end]

            x = self.postprocess_array(arr, kind="mediapipe_npy")
            return {"id": index, "vid_id": _id, "source": x}

        # ------------------------------------------------------------
        # CASE B) feats_type in {i3d, openpose}
        # Keep existing behavior: main source is npy + an extra mediapipe npy.
        # ------------------------------------------------------------
        elif self.feats_type in (SignFeatsType.i3d, SignFeatsType.openpose):
            with open(feats_file, "rb") as f:
                pose = np.load(f)
            pose = self.postprocess_array(pose, kind=str(self.feats_type.value))

            with open(feats_mediapipe_file, "rb") as f:
                pose_mediapipe = np.load(f)

            return {"id": index, "vid_id": _id, "source": pose, "mediapipe_source": pose_mediapipe}

        else:
            raise NotImplementedError(f"Unsupported feats_type: {self.feats_type}")

    def __len__(self):
        return len(self.sizes)

    # -----------------------
    # Postprocess helpers
    # -----------------------
    def postprocess_pose(self, pose: Pose):
        """
        Old mediapipe/openpose Pose-format pathway.
        Only used when the file is a .pose.
        """
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        FACEMESH_CONTOURS_POINTS = [
            str(p)
            for p in sorted(
                set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup])
            )
        ]
        POSE_RM = [
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX",
        ]
        POSE_POINTS = [kp.name for kp in mp_holistic.PoseLandmark if kp.name not in POSE_RM]

        pose = pose.get_components(
            ["FACE_LANDMARKS", "POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS, "POSE_LANDMARKS": POSE_POINTS},
        )

        if self.normalization == NormType.body:
            normalize_info = pose.header.normalization_info(
                p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
            )
            pose.normalize(normalize_info)
        elif self.normalization == NormType.kp_wise:
            pose.normalize_distribution(axis=(0, 1))
        elif self.normalization == NormType.global_xyz:
            pose.normalize_distribution(axis=(0, 1, 2))

        if self.data_augmentation:
            pose = pose.augment2d()

        return pose.torch()

    def postprocess_array(self, arr: np.ndarray, kind: str):
        """
        Numpy arrays -> torch tensors.
        Use for i3d/openpose AND mediapipe .npy.
        """
        if torch.is_tensor(arr):
            return arr
        return torch.from_numpy(arr)

    # -----------------------
    # Collater
    # -----------------------
    def collater(self, samples):
        if len(samples) == 0:
            return {}

        # lengths
        if self.feats_type == SignFeatsType.mediapipe:
            max_length = max([s["source"].shape[0] for s in samples])
        elif self.feats_type in (SignFeatsType.i3d, SignFeatsType.openpose):
            max_length = max([s["source"].shape[0] for s in samples])
            max_mediapipe_length = max([s["mediapipe_source"].shape[0] for s in samples])
        else:
            raise NotImplementedError(f"Unsupported feats_type in collater: {self.feats_type}")

        ids = []
        padding_masks = []
        collated_sources = []

        mediapipe_padding_masks = []
        collated_mediapipe_sources = []

        for sample in samples:
            x = sample["source"]
            if not torch.is_tensor(x):
                x = torch.from_numpy(x)

            if self.feats_type == SignFeatsType.mediapipe:
                padding_mask = torch.zeros(x.shape[0], dtype=torch.bool)

                if padding_mask.all():
                    continue

                diff_length = max_length - len(padding_mask)

                ids.append(sample["id"])
                padding_masks.append(F.pad(padding_mask, (0, diff_length), value=True))

                # pad time dimension only
                if x.dim() == 3:
                    # (T, K, C)
                    collated_sources.append(F.pad(x, (0, 0, 0, 0, 0, diff_length), value=0))
                elif x.dim() == 2:
                    # (T, D)
                    collated_sources.append(F.pad(x, (0, 0, 0, diff_length), value=0))
                else:
                    raise ValueError(f"Unexpected mediapipe tensor shape: {tuple(x.shape)}")

            else:
                mp = sample["mediapipe_source"]
                if not torch.is_tensor(mp):
                    mp = torch.from_numpy(mp)

                padding_mask = torch.zeros(x.shape[0], dtype=torch.bool)
                mediapipe_padding_mask = torch.zeros(mp.shape[0], dtype=torch.bool)

                if padding_mask.all():
                    continue

                diff_length = max_length - len(padding_mask)
                diff_mp_length = max_mediapipe_length - len(mediapipe_padding_mask)

                ids.append(sample["id"])
                padding_masks.append(F.pad(padding_mask, (0, diff_length), value=True))
                mediapipe_padding_masks.append(F.pad(mediapipe_padding_mask, (0, diff_mp_length), value=True))

                # repo-style padding (time axis)
                collated_sources.append(F.pad(x, (0, 0, 0, diff_length), value=0))
                collated_mediapipe_sources.append(F.pad(mp, (0, 0, 0, 0, 0, diff_mp_length), value=0))

        if len(collated_sources) == 0:
            return {}

        # ---------------------------
        # mediapipe-only batch
        # IMPORTANT: alias keys expected by signformer forward()
        # ---------------------------
        if self.feats_type == SignFeatsType.mediapipe:
            src = torch.stack(collated_sources).float()          # (B, T, ...) e.g. (B,T,543,3)
            pad = torch.stack(padding_masks)                     # (B, T)

            return {
                "id": torch.LongTensor(ids),
                "net_input": {
                    # existing keys used in other parts
                    "src_tokens": src,
                    "encoder_padding_mask": pad,

                    # ✅ required by this repo's Sign2TextTransformerModel forward()
                    "src_mediapipe_tokens": src,
                    "mediapipe_padding_mask": pad,
                },
            }

        # ---------------------------
        # i3d/openpose batch
        # ---------------------------
        return {
            "id": torch.LongTensor(ids),
            "net_input": {
                "src_tokens": torch.stack(collated_sources).float(),
                "src_mediapipe_tokens": torch.stack(collated_mediapipe_sources).float(),
                "encoder_padding_mask": torch.stack(padding_masks),
                "mediapipe_padding_mask": torch.stack(mediapipe_padding_masks),
            },
        }

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = np.lexsort([np.random.permutation(len(self)), np.array(self.sizes)])
            return order[::-1]
        else:
            return np.arange(len(self))


class MaskSignFeatsDataset(BaseWrapperDataset):
    def __init__(self, dataset: SignFeatsDataset, **mask_compute_kwargs):
        super().__init__(dataset)
        self.mask_compute_kwargs = mask_compute_kwargs
        self._features_size_map = {}
        self._C = mask_compute_kwargs["encoder_embed_dim"]
        self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def _compute_mask_indices(self, dims, padding_mask):
        raise NotImplementedError("This feature is still not available")

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        raise NotImplementedError("This feature is still not available")

    def collater(self, samples):
        out = self.dataset.collater(samples)
        raise NotImplementedError("This feature is still not available")


class RandomCropSignFeatsDataset(RandomCropDataset):
    def __init__(self, dataset: SignFeatsDataset, truncation_length: int, **kwargs):
        super().__init__(dataset, truncation_length, **kwargs)

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            item_len = item["source"].size(0)
            excess = item_len - self.truncation_length
            if excess > 0:
                start_idx = np.random.randint(0, excess)
                item["source"] = item["source"][start_idx : start_idx + self.truncation_length]
            return item
