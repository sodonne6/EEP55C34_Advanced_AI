# TTS Module using SpeechT5
# Uses real speaker embeddings from the CMU Arctic dataset
# Includes function that accepts text as input

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

class AAI_TTS:

    def __init__(self):
        """Initialize models and speaker embeddings"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Load SpeechT5 models
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # Load speaker embeddings
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
            revision="refs/convert/parquet"
        )

        # Choose speaker voice
        self.speaker_embedding = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(self.device)


    def generate_speech(self, text, output_file="speech.wav"):
        """Generate speech from input text"""

        # Process text
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate speech
        speech = self.tts_model.generate_speech(
            inputs["input_ids"],
            self.speaker_embedding,
            vocoder=self.vocoder
        )

        # Save audio
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)

        print(f"Speech saved to {output_file}")

        return output_file


# Example usage when running this file directly
if __name__ == "__main__":

    tts = AAI_TTS()

    text_input = input("Enter text to convert to speech: ")

    tts.generate_speech(text_input)


# Example usage from another file:

# from TTS.AAI_TTS import AAI_TTS
# tts = AAI_TTS()
# tts.generate_speech("Hello, this text came from the sign language model.")


# #----------------------------------------------

# # TTS Module using SpeechT5
# # Fine-tuned with the LJ Speech Dataset
# # Auto-plays audio after generation

# # ── MUST be set before any 'datasets' import ──
# import os
# os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

# import sys
# import torch
# import soundfile as sf
# from transformers import (
#     SpeechT5Processor,
#     SpeechT5ForTextToSpeech,
#     SpeechT5HifiGan,
#     Seq2SeqTrainer,
#     Seq2SeqTrainingArguments,
# )
# from datasets import load_dataset, Audio
# from dataclasses import dataclass
# from typing import Any, Dict, List, Union
# import numpy as np


# # ──────────────────────────────────────────────
# #  Auto-play helper
# # ──────────────────────────────────────────────

# def play_audio(file_path: str):
#     """Play a .wav file using the best available method for the current OS."""
#     if sys.platform.startswith("linux"):
#         os.system(f"aplay '{file_path}' 2>/dev/null || ffplay -nodisp -autoexit '{file_path}' 2>/dev/null")
#     elif sys.platform == "darwin":
#         os.system(f"afplay '{file_path}'")
#     elif sys.platform == "win32":
#         import winsound
#         winsound.PlaySound(file_path, winsound.SND_FILENAME)
#     else:
#         print(f"Auto-play not supported on this platform. Audio saved to: {file_path}")


# # ──────────────────────────────────────────────
# #  Data collator for SpeechT5
# # ──────────────────────────────────────────────

# # @dataclass
# # class SpeechT5DataCollator:
# #     """Pads a batch of (input_ids, labels, speaker_embeddings) for SpeechT5."""

# #     processor: Any
# #     decoder_start_token_id: int

# #     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
# #         input_ids      = [{"input_ids": f["input_ids"]} for f in features]
# #         label_features = [{"input_values": f["labels"]}  for f in features]
# #         spk_features   = [{"input_values": f["speaker_embeddings"]} for f in features]

# #         batch        = self.processor.pad(input_ids, return_tensors="pt")
# #         labels_batch = self.processor.pad(label_features=label_features, return_tensors="pt")
# #         spk_batch    = self.processor.pad(label_features=spk_features,   return_tensors="pt")

# #         labels = labels_batch["input_values"].masked_fill(
# #             labels_batch.attention_mask.ne(1), -100
# #         )

# #         # Strip decoder-start token if it was prepended
# #         if (labels[:, 0] == self.decoder_start_token_id).all():
# #             labels = labels[:, 1:]

# #         batch["labels"]             = labels
# #         batch["speaker_embeddings"] = spk_batch["input_values"]
# #         return batch

# @dataclass
# class SpeechT5DataCollator:
#     """Pads a batch of (input_ids, labels, speaker_embeddings) for SpeechT5."""

#     processor: Any
#     decoder_start_token_id: int

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         # Pad input_ids via the tokenizer directly
#         input_ids = [{"input_ids": f["input_ids"]} for f in features]
#         batch = self.processor.tokenizer.pad(input_ids, return_tensors="pt")

#         # Pad labels (mel spectrogram frames)
#         label_features = [{"input_values": f["labels"]} for f in features]
#         labels_batch = self.processor.feature_extractor.pad(
#             label_features, return_tensors="pt"
#         )

#         # Pad speaker embeddings
#         spk_features = [{"input_values": f["speaker_embeddings"]} for f in features]
#         spk_batch = self.processor.feature_extractor.pad(
#             spk_features, return_tensors="pt"
#         )

#         labels = labels_batch["input_values"].masked_fill(
#             labels_batch.attention_mask.ne(1), -100
#         )

#         if (labels[:, 0] == self.decoder_start_token_id).all():
#             labels = labels[:, 1:]

#         batch["labels"] = labels
#         batch["speaker_embeddings"] = spk_batch["input_values"]
#         return batch


# # ──────────────────────────────────────────────
# #  Helper: load xvector dataset once
# # ──────────────────────────────────────────────

# def _load_xvec_embedding(device: str) -> torch.Tensor:
#     """
#     Load a single LJ-Speech-representative speaker embedding from
#     Matthijs/cmu-arctic-xvectors.  Uses index 0 (safe for any split size).
#     Returns a (1, 512) float32 tensor on `device`.
#     """
#     xvec_ds = load_dataset(
#         "Matthijs/cmu-arctic-xvectors",
#         split="validation",
#         revision="refs/convert/parquet",
#     )
#     # Index 7306 is the canonical LJ-Speech embedding when the full validation
#     # split is available; fall back to 0 if the split is smaller.
#     idx = 7306 if len(xvec_ds) > 7306 else 0
#     emb = torch.tensor(xvec_ds[idx]["xvector"], dtype=torch.float32)
#     return emb.unsqueeze(0).to(device)


# # ──────────────────────────────────────────────
# #  Main TTS class
# # ──────────────────────────────────────────────

# class AAI_TTS:

#     def __init__(self, model_dir: str = "./speecht5_finetuned_ljspeech"):
#         """
#         Load (or fine-tune then load) SpeechT5 with LJ Speech embeddings.

#         Args:
#             model_dir: Directory to save / load the fine-tuned model.
#         """
#         self.device    = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model_dir = model_dir
#         print("Using device:", self.device)

#         self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#         self.vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

#         #if os.path.isdir(model_dir):
#         if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json")):
#             print(f"Loading fine-tuned model from '{model_dir}' …")
#             self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).to(self.device)
#         else:
#             print("No fine-tuned model found — running fine-tuning on LJ Speech …")
#             self.tts_model = self._finetune_on_ljspeech()

#         # Speaker embedding (cached; loaded once here for inference)
#         self.speaker_embedding = _load_xvec_embedding(self.device)

#     # ── Fine-tuning ────────────────────────────

#     def _finetune_on_ljspeech(self) -> SpeechT5ForTextToSpeech:
#         """Fine-tune SpeechT5 on a small slice of LJ Speech and return the model."""

#         print("Loading LJ Speech dataset …")
#         lj = load_dataset(
#             "lj_speech",
#             split="train[:200]",
#             revision="refs/convert/parquet",
#         )
#         lj = lj.cast_column("audio", Audio(sampling_rate=16_000))

#         # Load speaker embedding once for the whole fine-tune run
#         # (shape: (512,) — no batch dim needed inside map())
#         lj_speaker_emb = _load_xvec_embedding(self.device).squeeze(0).cpu()

#         def prepare(example):
#             audio = example["audio"]          # decoded by soundfile backend
#             text  = example["normalized_text"]

#             inputs = self.processor(
#                 text=text,
#                 audio_target=audio["array"],
#                 sampling_rate=16_000,
#                 return_attention_mask=False,
#             )

#             # labels may be (1, T) or (T,) depending on transformers version
#             labels = inputs["labels"]
#             if hasattr(labels, "shape") and labels.ndim == 2:
#                 labels = labels[0]
#             inputs["labels"]             = labels
#             inputs["speaker_embeddings"] = lj_speaker_emb
#             return inputs

#         print("Preprocessing dataset …")
#         lj = lj.map(prepare, remove_columns=lj.column_names)
#         lj = lj.filter(lambda x: len(x["input_ids"]) < 200)

#         model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

#         collator = SpeechT5DataCollator(
#             processor=self.processor,
#             decoder_start_token_id=model.config.decoder_start_token_id,
#         )

#         # gradient_checkpointing is disabled on CPU (no memory benefit, causes
#         # compatibility issues with predict_with_generate).
#         use_grad_ckpt = (self.device == "cuda")

#         training_args = Seq2SeqTrainingArguments(
#             output_dir=self.model_dir,
#             per_device_train_batch_size=4,
#             gradient_accumulation_steps=8,
#             learning_rate=1e-5,
#             warmup_steps=50,
#             max_steps=50,               # Raise to 1000–4000 for better quality
#             gradient_checkpointing=use_grad_ckpt,
#             fp16=(self.device == "cuda"),
#             eval_strategy="no",
#             save_steps=500,
#             logging_steps=10,
#             report_to=["none"],
#             predict_with_generate=True,
#         )

#         trainer = Seq2SeqTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=lj,
#             data_collator=collator,
#         )

#         print("Starting fine-tuning …")
#         trainer.train()

#         model.save_pretrained(self.model_dir)
#         self.processor.save_pretrained(self.model_dir)
#         print(f"Fine-tuned model saved to '{self.model_dir}'")

#         return model.to(self.device)

#     # ── Inference ─────────────────────────────

#     def generate_speech(
#         self,
#         text: str,
#         output_file: str = "speech.wav",
#         auto_play: bool = True,
#     ) -> str:
#         """
#         Generate speech from input text, save to file, and optionally play it.

#         Args:
#             text:        Input text to synthesise.
#             output_file: Path for the saved .wav file.
#             auto_play:   Whether to play the audio automatically after saving.

#         Returns:
#             Path to the saved audio file.
#         """
#         inputs = self.processor(text=text, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             speech = self.tts_model.generate_speech(
#                 inputs["input_ids"],
#                 self.speaker_embedding,
#                 vocoder=self.vocoder,
#             )

#         sf.write(output_file, speech.cpu().numpy(), samplerate=16_000)
#         print(f"Speech saved to '{output_file}'")

#         if auto_play:
#             print("Playing audio …")
#             play_audio(output_file)

#         return output_file


# # ──────────────────────────────────────────────
# #  CLI entry-point
# # ──────────────────────────────────────────────

# if __name__ == "__main__":
#     tts = AAI_TTS()
#     text_input = input("Enter text to convert to speech: ")
#     tts.generate_speech(text_input)


# # ──────────────────────────────────────────────
# #  Usage from another module:
# #
# #   from TTS.AAI_TTS import AAI_TTS
# #   tts = AAI_TTS()
# #   tts.generate_speech("Hello from the sign language model.")
# #
# #   # Suppress auto-play:
# #   tts.generate_speech("Silent save.", auto_play=False)
# # ──────────────────────────────────────────────

