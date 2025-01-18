import sys
import os
from dataclasses import dataclass

def SETUP_PATH():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_parent_dir = os.path.join(current_dir, 'imports', 'Kokoro-82M')
    sys.path.append(models_parent_dir)


SETUP_PATH()

from models import build_model
import torch
from kokoro import generate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOICES = [
    'af', # Default voice is a 50-50 mix of Bella & Sarah
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
]

@dataclass(frozen=True)
class Voice:
    model: object
    voice_name: str
    voicepack: object

def build_voicepack(index):
    if not index < len(VOICES):
        raise Exception('index out of range')
    model = build_model('imports/Kokoro-82M/kokoro-v0_19.pth', device)
    voice_name = VOICES[index]
    voicepack = torch.load(f'imports/Kokoro-82M/voices/{voice_name}.pt', weights_only=True).to(device)

    return Voice(model, voice_name, voicepack)

def generate_audio(voice: Voice, text="Hello, sailor"):
    audio, out_ps = generate(voice.model, text, voice.voicepack, lang=voice.voice_name[0])
    return audio, out_ps


import sounddevice as sd

def play_audio(audio_array, sample_rate=24000):
    sd.play(audio_array, sample_rate)
    sd.wait()

EXAMPLE = """
import main as m
vp = m.build_voicepack(0)
audio, _ = m.generate_audio(vp, "Testing audio")
m.play_audio(audio)
"""
