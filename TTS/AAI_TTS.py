#THIS WORKS FOR JUST DOING TEXT TO SPEECH 
#install everything
#!pip install -q transformers datasets soundfile speechbrain torchaudio accelerate

#Import Libraries 
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

#Load models
#if cuda
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#if cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = tts_model.to(device)
vocoder = vocoder.to(device)

#Load speaker embedding
speaker_embedding = torch.randn(1, 512)

#Prep text
text = "Hello, this is a text to speech test in Google Colab."
inputs = processor(text=text, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}
speaker_embedding = speaker_embedding.to(device)

#generate speech
speech = tts_model.generate_speech(
    inputs["input_ids"],
    speaker_embedding,
    vocoder=vocoder
)

#Save and plauy audio
sf.write("speech.wav", speech.cpu().numpy(), samplerate=16000)
# from IPython.display import Audio
# Audio("speech.wav")
print("Speech generated and saved as speech.wav") 



