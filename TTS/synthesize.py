"""This script 
1) Loads a pretrained SpeechT5 TTS pipeline (includes tokeniser/model/vocoder)
2) Loads a precomputed x-vector speaker embedding from a dataset (no SpeechBrain needed)
3) Synthesizes speech from text declared in the script
4) Writes outputs/out.wav
"""

import os
import torch
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset

#pretrained models
tts_model_name = "microsoft/speecht5_tts"
vocoder_model_name = "microsoft/speecht5_hifigan"
speaker_embed_model = "speechbrain/spkrec-ecapa-voxceleb"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
OUT_WAV = os.path.join(OUTPUT_DIR, "out.wav")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synthesiser = pipeline("text-to-speech", tts_model_name)

    # Load a ready-made x-vector speaker embedding (quick sanity-check)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    #this can be changed to any text - in the future this is how we can load text from the SLT model output
    text = "Testing speechT5 base model please work"
    out = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    sf.write(OUT_WAV, out["audio"], samplerate=out["sampling_rate"])
    print(f"Wrote: {OUT_WAV}")

if __name__ == "__main__":
    main()

