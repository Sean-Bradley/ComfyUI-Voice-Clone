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
from .chatterbox.tts import ChatterboxTTS
import perth

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

DEFAULT_TEXT = "Hello. and welcome to my course teaching how to generate video and images using Comfy UI. This hands-on guide will help you to create stunning visuals and animations using Comfy UI's powerful node-based workflow. Whether you're a digital artist, content creator, creative developer, or AI enthusiast, this course will show you how to turn your ideas into stunning visuals with no coding required."


class NullWatermarker:
    def apply_watermark(self, signal, sample_rate):
        print(
            "Using NullWatermarker: no watermark applied. sample_rate=" + str(sample_rate))
        return signal


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
                "disable_watermark": ("BOOLEAN", {
                    "default": False
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
        disable_watermark=False,
        voice_embedding=None
    ):
        model = ChatterboxTTS.from_local(
            "ComfyUI/models/tts/chatterbox", device=device
        )

        model_sample_rate = 24000
        if disable_watermark:
            model.watermarker = NullWatermarker()

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


class DetectWatermarkNode:
    """Detects a Perth watermark in an AUDIO input and returns True/False."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "detect"
    CATEGORY = "SBCODE"

    def detect(self, audio=None):
        try:
            if audio is None:
                print("[DetectWatermarkNode] No audio input provided")
                return (False,)

            waveform = audio.get("waveform")
            sr = audio.get("sample_rate")
            if waveform is None or sr is None:
                print("[DetectWatermarkNode] Invalid audio structure")
                return (False,)

            # Convert waveform tensor to mono numpy array
            import torch
            import numpy as _np

            if isinstance(waveform, torch.Tensor):
                arr = waveform.squeeze(0).cpu().numpy()
            else:
                arr = _np.asarray(waveform)

            if arr.ndim > 1:
                arr = arr.mean(axis=0)

            # Use PerthImplicitWatermarker to extract watermark
            watermarker = perth.PerthImplicitWatermarker()
            watermark = watermarker.get_watermark(arr, sample_rate=sr)
            detected = bool(watermark == 1.0 or watermark > 0.5)
            print(
                f"[DetectWatermarkNode] Extracted watermark: {watermark} -> detected={detected}")
            return (detected,)
        except Exception as e:
            print(f"[DetectWatermarkNode] Error detecting watermark: {e}")
            return (False,)


NODE_CLASS_MAPPINGS = {"VoiceCloneNode": VoiceCloneNode,
                       "DetectWatermarkNode": DetectWatermarkNode
                       }

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceCloneNode": "Voice Clone",
    "DetectWatermarkNode": "Detect Watermark"
}
