from .chatterbox.tts import ChatterboxTTS
from random import random
import importlib
import subprocess
import sys
import torch
import torchaudio
import tempfile
from pathlib import Path


# Auto-install missing dependencies
required_packages = ["librosa>=0.11.0", "conformer>=0.1.0",
                     "perth>=1.0.0", "resemble-perth>=1.0.1"]
for package in required_packages:
    pkg_name = package.split(">=")[0]
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"[VoiceCloneNode] Installing missing package: {package}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])

# --- Now safe to import ---

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

DEFAULT_TEXT = "Hello. and welcome to my course teaching how to generate video and images using Comfy UI. This hands-on guide will help you to create stunning visuals and animations using Comfy UI's powerful node-based workflow. Whether you're a digital artist, content creator, creative developer, or AI enthusiast, this course will show you how to turn your ideas into stunning visuals with no coding required."


class VoiceCloneNode:

    """Custom TTS node that clones voice from a reference audio and speaks entered text."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = Path("ComfyUI/models/tts/chatterbox")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DEFAULT_TEXT}),
                "exaggeration": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.15,
                    "max": 2.0,
                    "step": 0.05
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "voice_embedding": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "SBCODE"

    def generate(
        self,
        text,
        exaggeration,
        temperature,
        cfg_weight,
        min_p,
        top_p,
        repetition_penalty,
        voice_embedding=None
    ):
        model = ChatterboxTTS.from_local(
            "ComfyUI/models/tts/chatterbox", device=device
        )

        model_sample_rate = 24000

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "voice_prompt.wav"

            if voice_embedding is not None:
                waveform = voice_embedding["waveform"]
                if waveform.ndim == 3:
                    waveform = waveform.squeeze(0)
                elif waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)

                if voice_embedding["sample_rate"] != model_sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, voice_embedding["sample_rate"], model_sample_rate
                    )

                torchaudio.save(str(temp_path), waveform, model_sample_rate)
                # torchaudio.save_with_torchcodec(
                #     str(temp_path), waveform, model_sample_rate)
                print(f"Temporary voice file saved to: {temp_path}")

                wav = model.generate(
                    text=text,
                    audio_prompt_path=str(temp_path),
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            else:
                wav = model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

        return ({
            "waveform": wav.unsqueeze(0),
            "sample_rate": model_sample_rate
        },)


NODE_CLASS_MAPPINGS = {"VoiceCloneNode": VoiceCloneNode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceCloneNode": "Voice Clone"}
