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
