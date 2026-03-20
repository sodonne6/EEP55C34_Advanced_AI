"""
Text to Speech Inference
Converts text to audio using SpeechT5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


class TextToSpeech:
    """Convert text to speech using SpeechT5."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        self._load_models()
    
    def _load_models(self):
        """Load SpeechT5 TTS models."""
        print("[INFO] Loading TTS models...")
        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        print("[INFO] Loading speaker embeddings...")
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
            revision="refs/convert/parquet"
        )
        
        self.speaker_embedding = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(self.device)
        
        print("[INFO] TTS models loaded")
    
    @torch.no_grad()
    def synthesize(self, text: str, output_file: str = "output.wav") -> str:
        """Convert text to speech."""
        print(f"\n[INFO] Generating speech from: {text}")
        
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        speech = self.tts_model.generate_speech(
            inputs["input_ids"],
            self.speaker_embedding,
            vocoder=self.vocoder
        )
        
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
        
        print(f"[INFO] Speech saved to {output_file}")
        return output_file


def main():
    ap = argparse.ArgumentParser(description="Text to Speech Inference")
    ap.add_argument("--text", required=True, type=str, help="Input text to convert to speech")
    ap.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = ap.parse_args()
    
    tts = TextToSpeech(device=args.device)
    tts.synthesize(args.text, args.output)
    
    print("\n" + "=" * 80)
    print(f"AUDIO SAVED: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()