import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

import streamlit as st
import torch
import numpy as np
import io
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
    return processor, model

processor, model = load_model()

def text_to_audio(prompt: str):
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    audio_tensor = model.generate(**inputs, max_new_tokens=256)
    audio_np = audio_tensor[0].cpu().numpy()
    
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_np, samplerate=32000, format="WAV")
    audio_buffer.seek(0)
    
    return audio_buffer

st.title("Text-to-Audio Converter")
st.markdown("""
Enter your text prompt to generate audio using Facebook's MusicGen model.
For example: *80s pop track with bassy drums and synth*.
""")
prompt = st.text_area("Enter a description", height=150)

if st.button("Generate Audio"):
    if prompt.strip() == "":
        st.error("Please enter a valid text prompt!")
    else:
        with st.spinner("Generating audio..."):
            audio_bytes = text_to_audio(prompt)
        st.success("Audio generated!")
        st.audio(audio_bytes, format="audio/wav")
