# Use real speaker embeddings instead of random vectors
# This improves naturalness of SpeechT5 output

# Import Libraries
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load SpeechT5 models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load real speaker embeddings from CMU Arctic dataset
#embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation",
    revision="refs/convert/parquet"
)

# Choose a speaker embedding (change index for different voices)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speaker_embedding = speaker_embedding.to(device)

# Prepare input text
text = "Hello, this is a text to speech test generated using SpeechT5."
inputs = processor(text=text, return_tensors="pt")

# Move inputs to GPU if available
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate speech
speech = tts_model.generate_speech(
    inputs["input_ids"],
    speaker_embedding,
    vocoder=vocoder
)

# Save audio
sf.write("speech.wav", speech.cpu().numpy(), samplerate=16000)

print("Speech generated and saved as speech.wav")