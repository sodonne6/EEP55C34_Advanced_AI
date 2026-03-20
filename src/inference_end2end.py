"""
End-to-End Pipeline: Sign Language Video -> Text -> Speech
Combines sign_to_text and text_to_speech inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from inference_sign_to_text import SignLanguageTranslator
from inference_text_to_speech import TextToSpeech


class EndToEndPipeline:
    """Complete pipeline: Video -> Text -> Speech."""
    
    def __init__(
        self,
        repo_root: Path,
        ckpt_path: Path,
        spm_model: Path,
        data_dir: Path,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print("\n" + "=" * 80)
        print("INITIALIZING END-TO-END PIPELINE")
        print("=" * 80)
        
        self.translator = SignLanguageTranslator(
            repo_root=repo_root,
            ckpt_path=ckpt_path,
            spm_model=spm_model,
            data_dir=data_dir,
            device=str(self.device)
        )
        
        self.tts = TextToSpeech(device=str(self.device))
    
    def process(
        self,
        video_path: Path,
        output_audio: str = "output.wav",
        beam: int = 5,
        max_len_b: int = 40
    ) -> dict:
        """Process video end-to-end."""
        print("\n" + "=" * 80)
        print("SIGN LANGUAGE TO SPEECH PIPELINE")
        print("=" * 80)
        
        # Step 1: Translate video to text
        print("\nSTEP 1: SIGN LANGUAGE TO TEXT")
        print("-" * 80)
        translation = self.translator.translate(video_path, beam, max_len_b)
        print(f"Translation: {translation}")
        
        # Step 2: Convert text to speech
        print("\nSTEP 2: TEXT TO SPEECH")
        print("-" * 80)
        audio_file = self.tts.synthesize(translation, output_audio)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Input Video: {video_path}")
        print(f"Translation: {translation}")
        print(f"Output Audio: {audio_file}")
        print("=" * 80)
        
        return {
            "video": str(video_path),
            "translation": translation,
            "audio": audio_file
        }


def main():
    ap = argparse.ArgumentParser(description="End-to-End Sign Language to Speech Pipeline")
    ap.add_argument("--repo_root", required=True, type=Path, help="Fairseq repo root")
    ap.add_argument("--ckpt", required=True, type=Path, help="Translation checkpoint")
    ap.add_argument("--spm_model", required=True, type=Path, help="SentencePiece model")
    ap.add_argument("--data_dir", required=True, type=Path, help="Data directory")
    ap.add_argument("--video", required=True, type=Path, help="Input video file")
    ap.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    ap.add_argument("--beam", type=int, default=5, help="Beam search width")
    ap.add_argument("--max_len_b", type=int, default=40, help="Max output length")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = ap.parse_args()
    
    pipeline = EndToEndPipeline(
        repo_root=args.repo_root,
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        data_dir=args.data_dir,
        device=args.device
    )
    
    pipeline.process(args.video, args.output, args.beam, args.max_len_b)


if __name__ == "__main__":
    main()