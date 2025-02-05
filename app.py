import gradio as gr
import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
from tempfile import NamedTemporaryFile

# Load the processor and model globally (this may take some time the first time you run it)
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

def text_to_audio(prompt: str):
    """
    Given a text prompt, generate an audio sample using MusicGen and return the audio file path.
    """
    # Prepare the input: here we wrap the prompt in a list.
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    # Generate audio using the model.
    # The max_new_tokens parameter may be adjusted based on desired audio length.
    audio_tensor = model.generate(**inputs, max_new_tokens=256)
    
    # The generated audio is a tensor. Convert it to a NumPy array.
    # (Assume the model returns a tensor of shape [1, num_samples])
    audio_np = audio_tensor[0].cpu().numpy()
    
    # Write the audio sample to a temporary WAV file.
    # MusicGen typically generates audio at 32000 Hz.
    temp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio_np, samplerate=32000)
    
    # Return the path to the audio file.
    return temp_wav.name

# Define a Gradio interface:
iface = gr.Interface(
    fn=text_to_audio,
    inputs=gr.Textbox(lines=5, placeholder="Enter a description, e.g., '80s pop track with bassy drums and synth'", label="Text Prompt"),
    outputs=gr.Audio(label="Generated Audio"),
    title="Text to Audio Converter",
    description="Enter your text prompt to generate audio using Facebook's MusicGen model."
)

# Launch the interface.
if __name__ == "__main__":
    iface.launch()
