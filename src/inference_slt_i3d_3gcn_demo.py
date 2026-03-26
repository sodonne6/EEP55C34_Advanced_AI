from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import mediapipe as mp

POSE_N = 33
LH_N = 21
RH_N = 21
MP_TOTAL_N = POSE_N + LH_N + RH_N  # 75


def _load_python_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _register_graph_override(repo_root: Path, graph_py: Optional[Path]) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if graph_py is not None:
        graph_py = graph_py.resolve()
        print(f"[INFO] Registering custom graph module from: {graph_py}")
        _load_python_module_from_path("fairseq.models.sign_to_text.graph", graph_py)


def _resample_time_np(arr: np.ndarray, target_t: int) -> np.ndarray:
    src_t = int(arr.shape[0])
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0, got {target_t}")
    if src_t <= 0:
        raise ValueError("Cannot resample an empty sequence")
    if src_t == target_t:
        return arr
    if src_t == 1:
        return np.repeat(arr, target_t, axis=0)

    tail_shape = arr.shape[1:]
    x = torch.from_numpy(arr).float().reshape(src_t, -1).T.unsqueeze(0)
    x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)
    out = x.squeeze(0).T.contiguous().cpu().numpy().reshape(target_t, *tail_shape)
    return out.astype(np.float32)


def _uniform_temporal_subsample(frames: np.ndarray, max_frames: Optional[int]) -> np.ndarray:
    if max_frames is None or len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(np.int64)
    return frames[idx]


def _pad_short_clip(frames: np.ndarray, min_frames: int) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("No frames were provided")
    if len(frames) >= min_frames:
        return frames
    pad_n = min_frames - len(frames)
    pad = np.repeat(frames[-1:], pad_n, axis=0)
    return np.concatenate([frames, pad], axis=0)


def _decode_tokens(task, tokens: torch.Tensor) -> str:
    s = task.target_dictionary.string(tokens)
    bpe = getattr(task, "bpe", None)
    if bpe is not None:
        try:
            s = bpe.decode(s)
            return " ".join(str(s).split())
        except Exception:
            pass
    s = str(s).replace("▁", " ").strip()
    return " ".join(s.split())


def _ensure_spm_dict(spm_model: Path) -> Path:
    spm_model = spm_model.resolve()
    dict_path = Path(str(spm_model).replace(".model", ".txt"))

    if dict_path.exists():
        return dict_path

    try:
        import sentencepiece as spm
    except Exception as e:
        raise RuntimeError(
            f"Could not import sentencepiece to create a dictionary from {spm_model}"
        ) from e

    sp = spm.SentencePieceProcessor(model_file=str(spm_model))
    fairseq_specials = {"<pad>", "<s>", "</s>", "<unk>"}
    lines = []
    seen = set()

    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece in fairseq_specials or piece in seen:
            continue
        lines.append(f"{piece} 1")
        seen.add(piece)

    dict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Created SPM dict: {dict_path}")
    return dict_path


def _prepare_task_data_dir(data_dir: Optional[Path], spm_model: Path):
    if data_dir is not None:
        data_dir = data_dir.resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
        print(f"[INFO] Using provided task data dir: {data_dir}")
        return data_dir, None

    dict_src = _ensure_spm_dict(spm_model)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="slt_task_data_")
    temp_dir = Path(temp_dir_obj.name).resolve()

    shutil.copyfile(dict_src, temp_dir / "dict.txt")
    if dict_src.name != "dict.txt":
        shutil.copyfile(dict_src, temp_dir / dict_src.name)

    spm_yaml_path = spm_model.resolve().as_posix()
    config_text = (
        "vocab_filename: dict.txt\n"
        "bpe_tokenizer:\n"
        "  bpe: sentencepiece\n"
        f"  sentencepiece_model: '{spm_yaml_path}'\n"
        "prepend_tgt_lang_tag: false\n"
        "shuffle: false\n"
    )
    (temp_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    print(f"[INFO] Created minimal temporary task data dir: {temp_dir}")
    return temp_dir, temp_dir_obj


class MediaPipeHolistic75Extractor:
    """
    Extract 75 landmarks per frame:
      33 pose + 21 left hand + 21 right hand.
    Returns (T, 75, 3) float32.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_holistic = mp.solutions.holistic

    @staticmethod
    def _lm_list_to_np(landmarks, n_expected: int) -> np.ndarray:
        if landmarks is None:
            return np.zeros((n_expected, 3), dtype=np.float32)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
        if pts.shape != (n_expected, 3):
            out = np.zeros((n_expected, 3), dtype=np.float32)
            n = min(n_expected, pts.shape[0])
            out[:n] = pts[:n]
            return out
        return pts

    def extract(self, frames_bgr: np.ndarray) -> np.ndarray:
        if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
            raise ValueError(f"Expected frames with shape (T,H,W,3), got {frames_bgr.shape}")

        seq: List[np.ndarray] = []
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=False,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as holistic:
            for frame_bgr in frames_bgr:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                pose = self._lm_list_to_np(results.pose_landmarks, POSE_N)
                lh = self._lm_list_to_np(results.left_hand_landmarks, LH_N)
                rh = self._lm_list_to_np(results.right_hand_landmarks, RH_N)

                seq.append(np.concatenate([pose, lh, rh], axis=0))

        out = np.stack(seq, axis=0).astype(np.float32)
        if out.shape[1:] != (MP_TOTAL_N, 3):
            raise RuntimeError(f"Unexpected MediaPipe output shape: {out.shape}")
        return out


class PseudoI3DFromMediaPipe:
    """
    Cheap stand-in for the I3D main branch.

    The checkpoint still expects src_tokens with shape (T, 1024), so for demo mode
    we derive a deterministic 1024-d feature stream from the same 75-point MediaPipe
    landmarks instead of running the heavy I3D network.
    """

    def __init__(self, out_dim: int = 1024):
        self.out_dim = out_dim

    def extract_from_landmarks(self, mp_landmarks: np.ndarray) -> np.ndarray:
        if mp_landmarks.ndim != 3 or mp_landmarks.shape[1:] != (MP_TOTAL_N, 3):
            raise ValueError(f"Expected MediaPipe landmarks (T,75,3), got {mp_landmarks.shape}")

        mp = mp_landmarks.astype(np.float32)
        T = mp.shape[0]

        flat = mp.reshape(T, -1)
        vel = np.diff(flat, axis=0, prepend=flat[:1])
        acc = np.diff(vel, axis=0, prepend=vel[:1])

        pose = mp[:, 0:33, :]
        lh = mp[:, 33:54, :]
        rh = mp[:, 54:75, :]

        pose_ctr = pose.mean(axis=1)
        lh_ctr = lh.mean(axis=1)
        rh_ctr = rh.mean(axis=1)

        pose_spread = pose.std(axis=1)
        lh_spread = lh.std(axis=1)
        rh_spread = rh.std(axis=1)

        hands_gap = lh_ctr - rh_ctr

        summary = np.concatenate(
            [pose_ctr, lh_ctr, rh_ctr, pose_spread, lh_spread, rh_spread, hands_gap],
            axis=1,
        )

        base = np.concatenate([flat, vel, acc, summary], axis=1)

        mean = base.mean(axis=0, keepdims=True)
        std = base.std(axis=0, keepdims=True)
        base = (base - mean) / (std + 1e-5)

        if base.shape[1] < self.out_dim:
            reps = (self.out_dim + base.shape[1] - 1) // base.shape[1]
            base = np.tile(base, (1, reps))
        feats = base[:, : self.out_dim]
        return feats.astype(np.float32)


class SignLanguageTranslatorI3D3GCNDemo:
    def __init__(
        self,
        repo_root: Path,
        ckpt_path: Path,
        spm_model: Path,
        data_dir: Optional[Path] = None,
        device: str = "cpu",
        model_py: Optional[Path] = None,
        graph_py: Optional[Path] = None,
        max_input_frames: int = 64,
        min_input_frames: int = 24,
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.max_input_frames = max_input_frames
        self.min_input_frames = min_input_frames
        self._temp_task_data_dir = None
        print(f"[INFO] Demo translator device: {self.device}")

        self._load_translation_model(
            repo_root=repo_root,
            ckpt_path=ckpt_path,
            spm_model=spm_model,
            data_dir=data_dir,
            model_py=model_py,
            graph_py=graph_py,
        )
        self.mp_extractor = MediaPipeHolistic75Extractor()
        self.pseudo_i3d = PseudoI3DFromMediaPipe(out_dim=1024)

    def _load_translation_model(
        self,
        repo_root: Path,
        ckpt_path: Path,
        spm_model: Path,
        data_dir: Optional[Path] = None,
        model_py: Optional[Path] = None,
        graph_py: Optional[Path] = None,
    ) -> None:
        print("[INFO] Loading translation checkpoint...")

        repo_root = repo_root.resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from fairseq import tasks
        from fairseq import utils as fairseq_utils

        state = torch.load(str(ckpt_path.resolve()), map_location="cpu")
        cfg_raw = state.get("cfg")
        if cfg_raw is None:
            raise RuntimeError("Checkpoint missing 'cfg'")

        print(f"[INFO] Checkpoint cfg type: {type(cfg_raw)}")

        if OmegaConf.is_config(cfg_raw):
            cfg = OmegaConf.create(OmegaConf.to_container(cfg_raw, resolve=True))
        elif isinstance(cfg_raw, dict):
            cfg = OmegaConf.create(cfg_raw)
        else:
            try:
                cfg = OmegaConf.create(vars(cfg_raw))
            except Exception as e:
                raise RuntimeError(f"Unsupported checkpoint cfg type: {type(cfg_raw)}") from e

        if not hasattr(cfg, "common") or cfg.common is None:
            cfg.common = OmegaConf.create({})
        if not hasattr(cfg, "task") or cfg.task is None:
            cfg.task = OmegaConf.create({})
        if not hasattr(cfg, "model") or cfg.model is None:
            cfg.model = OmegaConf.create({})

        local_user_dir = repo_root / "examples" / "sign_language"
        old_user_dir = getattr(cfg.common, "user_dir", None)
        print(f"[INFO] Checkpoint user_dir: {old_user_dir}")
        print(f"[INFO] Using local fairseq user_dir: {local_user_dir}")

        if not local_user_dir.exists():
            raise RuntimeError(f"Expected fairseq user_dir does not exist: {local_user_dir}")

        cfg.common.user_dir = str(local_user_dir)
        fairseq_utils.import_user_module(cfg.common)

        _register_graph_override(repo_root=repo_root, graph_py=graph_py)

        if model_py is None:
            raise RuntimeError("model_py must be provided for the 3-GCN override")
        model_py = model_py.resolve()
        print(f"[INFO] Loading override model class from: {model_py}")
        override_mod = _load_python_module_from_path("sign2text_transformer_3_gcn_override_demo", model_py)

        if not hasattr(override_mod, "Sign2TextTransformerModel3GCN"):
            raise RuntimeError("Override model file does not define Sign2TextTransformerModel3GCN")
        ModelClass = getattr(override_mod, "Sign2TextTransformerModel3GCN")

        cfg.model._name = "sign2text_transformer_3gcn"
        cfg.task.feats_type = "i3d"
        cfg.model.feats_type = "i3d"

        print(f"[INFO] Forced model._name = {cfg.model._name}")
        print(f"[INFO] Forced task.feats_type = {cfg.task.feats_type}")
        print(f"[INFO] Forced model.feats_type = {cfg.model.feats_type}")

        _ensure_spm_dict(spm_model)

        task_data_dir, temp_dir_obj = _prepare_task_data_dir(
            data_dir=data_dir,
            spm_model=spm_model,
        )
        self._temp_task_data_dir = temp_dir_obj

        cfg.task.data = str(task_data_dir.resolve())
        cfg.task.bpe_sentencepiece_model = str(spm_model.resolve())
        if hasattr(cfg.task, "tokenizer_bpe_model"):
            cfg.task.tokenizer_bpe_model = str(spm_model.resolve())

        self.task = tasks.setup_task(cfg.task)
        self.model = ModelClass.build_model(cfg.model, self.task)

        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "fuse_proj"):
            raise RuntimeError("Built model is missing encoder.fuse_proj")

        print(f"[INFO] Built model class: {self.model.__class__.__name__}")
        print(f"[INFO] Built encoder fuse_proj shape: {tuple(self.model.encoder.fuse_proj.weight.shape)}")

        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] Translation checkpoint loaded")

    def _prepare_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = _uniform_temporal_subsample(frames, self.max_input_frames)
        frames = _pad_short_clip(frames, self.min_input_frames)
        return frames.astype(np.uint8)

    def translate_video(self, video_path: Path, beam: int = 3, max_len_b: int = 32) -> str:
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("No frames extracted from video")
        return self.translate_frames(np.asarray(frames, dtype=np.uint8), beam=beam, max_len_b=max_len_b)

    def translate_frames(self, frames_bgr: np.ndarray, beam: int = 3, max_len_b: int = 32) -> str:
        if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
            raise ValueError(f"Expected frames with shape (T,H,W,3), got {frames_bgr.shape}")

        frames_bgr = self._prepare_frames(frames_bgr)
        print(f"[INFO] Demo inference on {len(frames_bgr)} frames")

        print("[INFO] Extracting MediaPipe 75 landmarks...")
        mp_landmarks = self.mp_extractor.extract(frames_bgr)
        print(f"[INFO] MediaPipe landmarks shape: {mp_landmarks.shape}")

        print("[INFO] Building pseudo-I3D features from MediaPipe...")
        pseudo_i3d_features = self.pseudo_i3d.extract_from_landmarks(mp_landmarks)
        print(f"[INFO] Pseudo-I3D feature shape: {pseudo_i3d_features.shape}")

        return self._translate_features(pseudo_i3d_features, mp_landmarks, beam=beam, max_len_b=max_len_b)

    @torch.no_grad()
    def _translate_features(
        self,
        i3d_features: np.ndarray,
        mp_landmarks: np.ndarray,
        beam: int,
        max_len_b: int,
    ) -> str:
        sample = self._make_sample(i3d_features, mp_landmarks)

        gen_cfg = OmegaConf.create({
            "beam": int(beam),
            "max_len_a": 0.0,
            "max_len_b": int(max_len_b),
            "lenpen": 1.0,
            "no_repeat_ngram_size": 3,
        })

        generator = self.task.build_generator([self.model], gen_cfg)
        hypos = self.task.inference_step(generator, [self.model], sample)
        best = hypos[0][0]
        hypo_tokens = best["tokens"].detach().cpu()
        decoded = _decode_tokens(self.task, hypo_tokens)

        words = decoded.split()
        if words:
            deduped = [words[0]]
            for word in words[1:]:
                if word.lower() != deduped[-1].lower():
                    deduped.append(word)
            decoded = " ".join(deduped)

        return decoded

    def _make_sample(self, i3d_features: np.ndarray, mp_landmarks: np.ndarray) -> dict:
        if i3d_features.ndim != 2 or i3d_features.shape[1] != 1024:
            raise ValueError(f"Expected pseudo-I3D features (T,1024), got {i3d_features.shape}")
        if mp_landmarks.ndim != 3 or mp_landmarks.shape[1:] != (MP_TOTAL_N, 3):
            raise ValueError(f"Expected MediaPipe landmarks (T,75,3), got {mp_landmarks.shape}")

        t = min(int(i3d_features.shape[0]), int(mp_landmarks.shape[0]))
        if t <= 0:
            raise ValueError("Empty feature sequence")

        i3d_features = i3d_features[:t].astype(np.float32)
        mp_landmarks = mp_landmarks[:t].astype(np.float32)

        src_tokens = torch.from_numpy(i3d_features).unsqueeze(0).to(self.device)
        src_mp = torch.from_numpy(mp_landmarks).unsqueeze(0).to(self.device)

        encoder_padding_mask = torch.zeros((1, t), dtype=torch.bool, device=self.device)
        mediapipe_padding_mask = torch.zeros((1, t), dtype=torch.bool, device=self.device)

        return {
            "id": torch.LongTensor([0]).to(self.device),
            "net_input": {
                "src_tokens": src_tokens,
                "src_mediapipe_tokens": src_mp,
                "encoder_padding_mask": encoder_padding_mask,
                "mediapipe_padding_mask": mediapipe_padding_mask,
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sign Language to Text Demo Inference (MediaPipe-only pseudo-I3D + 3GCN)")
    ap.add_argument("--repo_root", required=True, type=Path, help="Fairseq repo root")
    ap.add_argument("--ckpt", required=True, type=Path, help="Translation checkpoint")
    ap.add_argument("--spm_model", required=True, type=Path, help="SentencePiece model")
    ap.add_argument("--data_dir", required=False, type=Path, default=None, help="Optional task data directory")
    ap.add_argument("--video", required=True, type=Path, help="Input video file")
    ap.add_argument("--model_py", required=True, type=Path, help="Path to sign2text_transformer_3_gcn.py")
    ap.add_argument("--graph_py", required=True, type=Path, help="Path to graph.py with 33-node graph")
    ap.add_argument("--beam", type=int, default=3, help="Beam search width")
    ap.add_argument("--max_len_b", type=int, default=32, help="Max decoded length")
    ap.add_argument("--max_input_frames", type=int, default=64, help="Uniformly subsample input to at most this many frames")
    ap.add_argument("--min_input_frames", type=int, default=24, help="Pad short clips to at least this many frames")
    ap.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = ap.parse_args()

    translator = SignLanguageTranslatorI3D3GCNDemo(
        repo_root=args.repo_root,
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        data_dir=args.data_dir,
        device=args.device,
        model_py=args.model_py,
        graph_py=args.graph_py,
        max_input_frames=args.max_input_frames,
        min_input_frames=args.min_input_frames,
    )

    translation = translator.translate_video(
        video_path=args.video,
        beam=args.beam,
        max_len_b=args.max_len_b,
    )

    print("\n" + "=" * 80)
    print(f"TRANSLATION: {translation}")
    print("=" * 80)


if __name__ == "__main__":
    main()