"""
Sign Language to Text Inference
Extracts features from video and translates to text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ResNet50Extractor:
    """Extract ResNet50 features from frames."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        print("[INFO] Loading ResNet50...")
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device).eval()
        
        print("[INFO] ResNet50 loaded")
    
    def extract(self, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Extract ResNet50 features from frames."""
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
        
        T = frames.shape[0]
        all_features = []
        
        for start_idx in range(0, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            batch_frames = frames[start_idx:end_idx]
            
            batch_tensor = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                batch_feats = self.model(batch_tensor).cpu().numpy()
                all_features.append(batch_feats)
        
        return np.vstack(all_features).astype(np.float32)


class MediaPipeExtractor:
    """Extract MediaPipe pose landmarks from frames."""
    
    def __init__(self):
        print("[INFO] Loading MediaPipe...")
        
        model_path = Path("/tmp/pose_landmarker_lite.task")
        if not model_path.exists():
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, str(model_path))
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        print("[INFO] MediaPipe loaded")
    
    def extract(self, frames: np.ndarray) -> np.ndarray:
        """Extract MediaPipe landmarks from frames."""
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        landmarks_sequence = []
        
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(image)
            
            if results.pose_landmarks:
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks[0]],
                    dtype=np.float32
                )
                landmarks = self._normalize_landmarks(landmarks)
                landmarks_sequence.append(landmarks)
            else:
                landmarks_sequence.append(np.zeros((33, 3), dtype=np.float32))
        
        return np.array(landmarks_sequence, dtype=np.float32)
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks by shoulder distance."""
        out = landmarks.copy()
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        
        if landmarks[LEFT_HIP, 0] != 0 and landmarks[RIGHT_HIP, 0] != 0:
            center = 0.5 * (out[LEFT_HIP, :2] + out[RIGHT_HIP, :2])
        elif landmarks[LEFT_SHOULDER, 0] != 0 and landmarks[RIGHT_SHOULDER, 0] != 0:
            center = 0.5 * (out[LEFT_SHOULDER, :2] + out[RIGHT_SHOULDER, :2])
        else:
            center = np.array([0.5, 0.5], dtype=np.float32)
        
        if landmarks[LEFT_SHOULDER, 0] != 0 and landmarks[RIGHT_SHOULDER, 0] != 0:
            scale = np.linalg.norm(out[LEFT_SHOULDER, :2] - out[RIGHT_SHOULDER, :2])
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
        else:
            scale = 1.0
        
        out[:, 0] = (out[:, 0] - center[0]) / (scale + 1e-6)
        out[:, 1] = (out[:, 1] - center[1]) / (scale + 1e-6)
        return out.astype(np.float32)


class SignLanguageTranslator:
    """Translate sign language video to text."""
    
    def __init__(
        self,
        repo_root: Path,
        ckpt_path: Path,
        spm_model: Path,
        data_dir: Path,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        self._load_models(repo_root, ckpt_path, spm_model, data_dir)
        self.resnet_extractor = ResNet50Extractor(device=str(self.device))
        self.mp_extractor = MediaPipeExtractor()
    
    def _load_models(
        self,
        repo_root: Path,
        ckpt_path: Path,
        spm_model: Path,
        data_dir: Path
    ):
        """Load fairseq translation models."""
        print("[INFO] Loading translation models...")
        
        sys.path.insert(0, str(repo_root.resolve()))
        from fairseq import tasks
        
        state = torch.load(str(ckpt_path.resolve()), map_location="cpu")
        cfg_raw = state.get("cfg")
        if cfg_raw is None:
            raise RuntimeError("Checkpoint missing 'cfg'")
        
        cfg = OmegaConf.create(OmegaConf.to_container(cfg_raw, resolve=True))
        
        # Ensure SPM dict exists
        dict_path = Path(str(spm_model).replace(".model", ".txt"))
        if not dict_path.exists():
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=str(spm_model))
                FAIRSEQ_SPECIALS = {"<pad>", "<s>", "</s>", "<unk>"}
                lines = []
                seen = set()
                for i in range(sp.get_piece_size()):
                    piece = sp.id_to_piece(i)
                    if piece in FAIRSEQ_SPECIALS or piece in seen:
                        continue
                    lines.append(f"{piece} 1")
                    seen.add(piece)
                dict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"[INFO] Created SPM dict: {dict_path}")
            except Exception as e:
                print(f"[WARN] Could not create SPM dict: {e}")
        
        cfg.task.data = str(data_dir.resolve())
        cfg.task.bpe_sentencepiece_model = str(spm_model.resolve())
        
        self.task = tasks.setup_task(cfg.task)
        self.model = self.task.build_model(cfg.model)
        
        try:
            self.model.load_state_dict(state["model"], strict=True)
        except RuntimeError:
            print("[WARN] strict=True failed; retrying strict=False")
            self.model.load_state_dict(state["model"], strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("[INFO] Translation models loaded")
    
    def translate(
        self,
        video_path: Path,
        beam: int = 5,
        max_len_b: int = 40
    ) -> str:
        """Translate sign language video to text."""
        print(f"\n[INFO] Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        frames_array = np.array(frames, dtype=np.uint8)
        print(f"[INFO] Extracted {len(frames)} frames")
        
        print("[INFO] Extracting ResNet50 features...")
        resnet_features = self.resnet_extractor.extract(frames_array)
        
        print("[INFO] Extracting MediaPipe landmarks...")
        mp_landmarks = self.mp_extractor.extract(frames_array)
        
        print("[INFO] Translating to text...")
        translation = self._translate_features(
            resnet_features, mp_landmarks, beam, max_len_b
        )
        
        return translation
    
    @torch.no_grad()
    def _translate_features(
        self,
        resnet: np.ndarray,
        mp: np.ndarray,
        beam: int,
        max_len_b: int
    ) -> str:
        """Translate features to text."""
        sample = self._make_sample(resnet, mp)
        
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
        decoded = self._decode_hypo(hypo_tokens)
        
        words = decoded.split()
        if words:
            deduped = [words[0]]
            for word in words[1:]:
                if word.lower() != deduped[-1].lower():
                    deduped.append(word)
            decoded = " ".join(deduped)
        
        return decoded
    
    def _make_sample(self, resnet: np.ndarray, mp: np.ndarray) -> dict:
        """Create fairseq sample."""
        T = int(min(resnet.shape[0], mp.shape[0]))
        if T <= 0:
            raise ValueError("Empty sequence")
        
        resnet = resnet[:T].astype(np.float32)
        mp = mp[:T].astype(np.float32)
        
        src_tokens = torch.from_numpy(resnet).unsqueeze(0).to(self.device)
        src_mp = torch.from_numpy(mp).unsqueeze(0).to(self.device)
        
        encoder_padding_mask = torch.zeros((1, T), dtype=torch.bool, device=self.device)
        mediapipe_padding_mask = torch.zeros((1, T), dtype=torch.bool, device=self.device)
        
        return {
            "id": torch.LongTensor([0]).to(self.device),
            "net_input": {
                "src_tokens": src_tokens,
                "src_mediapipe_tokens": src_mp,
                "encoder_padding_mask": encoder_padding_mask,
                "mediapipe_padding_mask": mediapipe_padding_mask,
            },
        }
    
    def _decode_hypo(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        s = self.task.target_dictionary.string(tokens)
        
        bpe = getattr(self.task, "bpe", None)
        if bpe is not None:
            try:
                s = bpe.decode(s)
                return " ".join(str(s).split())
            except Exception:
                pass
        
        s = str(s).replace("▁", " ").strip()
        return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser(description="Sign Language to Text Inference")
    ap.add_argument("--repo_root", required=True, type=Path, help="Fairseq repo root")
    ap.add_argument("--ckpt", required=True, type=Path, help="Translation checkpoint")
    ap.add_argument("--spm_model", required=True, type=Path, help="SentencePiece model")
    ap.add_argument("--data_dir", required=True, type=Path, help="Data directory")
    ap.add_argument("--video", required=True, type=Path, help="Input video file")
    ap.add_argument("--beam", type=int, default=5, help="Beam search width")
    ap.add_argument("--max_len_b", type=int, default=40, help="Max output length")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = ap.parse_args()
    
    translator = SignLanguageTranslator(
        repo_root=args.repo_root,
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        data_dir=args.data_dir,
        device=args.device
    )
    
    translation = translator.translate(args.video, args.beam, args.max_len_b)
    
    print("\n" + "=" * 80)
    print(f"TRANSLATION: {translation}")
    print("=" * 80)


if __name__ == "__main__":
    main()