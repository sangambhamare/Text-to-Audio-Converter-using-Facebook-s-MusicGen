import streamlit as st
import torch
import numpy as np
import io
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf

# Load the processor and model globally (this may take some time on first run)
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
    return processor, model

processor, model = load_model()

def text_to_audio(prompt: str):
    """
    Given a text prompt, generate an audio sample using MusicGen and return audio as a byte stream.
    """
    # Prepare the input by wrapping the prompt in a list.
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    # Generate audio using the model.
    audio_tensor = model.generate(**inputs, max_new_tokens=256)
    
    # Convert the generated audio tensor to a NumPy array.
    audio_np = audio_tensor[0].cpu().numpy()
    
    # Write the audio sample to an in-memory WAV file.
    audio_buffer = io.BytesIO()
    # MusicGen typically generates audio at 32000 Hz.
    sf.write(audio_buffer, audio_np, samplerate=32000, format="WAV")
    audio_buffer.seek(0)
    
    return audio_buffer

# Set up the Streamlit interface
st.title("Text-to-Audio Converter")
st.markdown("""
Enter your text prompt to generate audio using Facebook's MusicGen model.
For example: *80s pop track with bassy drums and synth*.
""")

# Text input for the prompt.
prompt = st.text_area("Enter a description", height=150)

# Generate the audio when the user clicks the button.
if st.button("Generate Audio"):
    if prompt.strip() == "":
        st.error("Please enter a valid text prompt!")
    else:
        with st.spinner("Generating audio..."):
            audio_bytes = text_to_audio(prompt)
        st.success("Audio generated!")
        st.audio(audio_bytes, format="audio/wav")
