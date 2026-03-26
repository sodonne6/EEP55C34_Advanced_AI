from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Union

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import pipeline

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tts"

DEFAULT_TTS_MODEL_NAME = os.environ.get("SLT_TTS_MODEL", "microsoft/speecht5_tts")
DEFAULT_XVECTOR_DATASET = os.environ.get("SLT_TTS_XVECTOR_DATASET", "Matthijs/cmu-arctic-xvectors")
DEFAULT_SPEAKER_ID = int(os.environ.get("SLT_TTS_SPEAKER_ID", "7306"))
DEFAULT_DEVICE = os.environ.get("SLT_TTS_DEVICE", "cpu")


def _patch_speecht5_dropout_bug() -> None:
    """
    Work around a dtype mismatch in SpeechT5's decoder prenet dropout path
    on older torch stacks by ensuring the 'false' branch is float-typed.

    We patch once per process.
    """
    from transformers.models.speecht5.modeling_speecht5 import SpeechT5SpeechDecoderPrenet

    if getattr(SpeechT5SpeechDecoderPrenet, "_slt_dropout_patch_applied", False):
        return

    def _consistent_dropout_fixed(self, inputs_embeds, p):
        p = float(p)
        if p <= 0.0:
            return inputs_embeds

        keep_prob = 1.0 - p
        if keep_prob <= 0.0:
            return torch.zeros_like(inputs_embeds)

        mask = torch.bernoulli(
            torch.full_like(inputs_embeds[0], keep_prob, dtype=inputs_embeds.dtype)
        )
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)

        return torch.where(
            all_masks == 1,
            inputs_embeds,
            torch.zeros_like(inputs_embeds),
        ) * (1.0 / keep_prob)

    SpeechT5SpeechDecoderPrenet._consistent_dropout = _consistent_dropout_fixed
    SpeechT5SpeechDecoderPrenet._slt_dropout_patch_applied = True
    print("[TTS] Applied SpeechT5 dropout dtype patch", flush=True)


class SpeechT5TTSEngine:
    def __init__(
        self,
        tts_model_name: str = DEFAULT_TTS_MODEL_NAME,
        xvector_dataset_name: str = DEFAULT_XVECTOR_DATASET,
        speaker_id: int = DEFAULT_SPEAKER_ID,
        device: str = DEFAULT_DEVICE,
    ):
        self.tts_model_name = tts_model_name
        self.xvector_dataset_name = xvector_dataset_name
        self.speaker_id = int(speaker_id)

        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.pipeline_device = 0 if use_cuda else -1
        self.torch_device = torch.device("cuda:0" if use_cuda else "cpu")

        self._synthesiser = None
        self._speaker_embedding = None

    def _ensure_loaded(self) -> None:
        _patch_speecht5_dropout_bug()

        if self._synthesiser is None:
            print(f"[TTS] Loading pipeline: {self.tts_model_name}", flush=True)
            self._synthesiser = pipeline(
                task="text-to-speech",
                model=self.tts_model_name,
                device=self.pipeline_device,
            )
            print("[TTS] Pipeline loaded", flush=True)

        if self._speaker_embedding is None:
            print(f"[TTS] Loading x-vector dataset: {self.xvector_dataset_name}", flush=True)
            embeddings_dataset = load_dataset(self.xvector_dataset_name, split="validation")
            speaker_embedding = torch.tensor(
                embeddings_dataset[self.speaker_id]["xvector"],
                dtype=torch.float32,
            ).unsqueeze(0)

            if self.torch_device.type == "cuda":
                speaker_embedding = speaker_embedding.to(self.torch_device)

            self._speaker_embedding = speaker_embedding
            print(f"[TTS] Loaded speaker embedding id: {self.speaker_id}", flush=True)

    def synthesize_to_file(self, text: str, out_wav: Union[Path, str]) -> Path:
        text = " ".join(str(text).split()).strip()
        if not text:
            raise ValueError("Cannot synthesize empty text")

        self._ensure_loaded()

        out_wav = Path(out_wav).resolve()
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        print(f"[TTS] Synthesizing: {text}", flush=True)
        out = self._synthesiser(
            text,
            forward_params={"speaker_embeddings": self._speaker_embedding},
        )

        sf.write(out_wav, out["audio"], samplerate=out["sampling_rate"])
        print(f"[TTS] Wrote audio: {out_wav}", flush=True)
        return out_wav


def main() -> None:
    ap = argparse.ArgumentParser(description="SpeechT5 TTS helper")
    ap.add_argument(
        "--text",
        type=str,
        default="Testing speechT5 base model please work",
        help="Text to synthesize",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "out.wav",
        help="Output wav path",
    )
    ap.add_argument(
        "--speaker_id",
        type=int,
        default=DEFAULT_SPEAKER_ID,
        help="Speaker embedding index from the x-vector dataset",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device for TTS inference",
    )
    args = ap.parse_args()

    engine = SpeechT5TTSEngine(
        speaker_id=args.speaker_id,
        device=args.device,
    )
    engine.synthesize_to_file(args.text, args.out)


if __name__ == "__main__":
    main()