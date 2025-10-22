from random import random
import importlib
import subprocess
import sys
import torch
import torchaudio
import tempfile
from pathlib import Path
import re


# Auto-install missing dependencies
required_packages = ["librosa>=0.11.0", "conformer>=0.1.0",
                     "perth>=1.0.0", "resemble-perth>=1.0.1", "resampy>=0.4.3"]
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
        """
        Split input by sentences (with a fallback to word-based splits for very long sentences),
        generate each segment sequentially to reduce peak memory, and concatenate the outputs.
        """
        # lazy-load model once per node instance
        if getattr(self, "model", None) is None:
            self.model = ChatterboxTTS.from_local(str(self.model_path), device=device)

        model = self.model
        model_sample_rate = 24000
        if disable_watermark:
            model.watermarker = NullWatermarker()

        def split_into_sentences(s):
            parts = re.split(r'(?<=[.!?])\s+', s.strip())
            return [p for p in parts if p]

        def chunk_sentences_by_words(sentences, max_words=20):
            segments = []
            current = []
            current_words = 0
            for sent in sentences:
                word_count = len(sent.split())
                if word_count > max_words:
                    words = sent.split()
                    for i in range(0, len(words), max_words):
                        chunk = " ".join(words[i:i + max_words])
                        segments.append(chunk)
                    current = []
                    current_words = 0
                else:
                    if current_words + word_count <= max_words:
                        current.append(sent)
                        current_words += word_count
                    else:
                        segments.append(" ".join(current))
                        current = [sent]
                        current_words = word_count
            if current:
                segments.append(" ".join(current))
            return segments

        max_words_per_segment = 20
        sentences = split_into_sentences(text)
        segments = chunk_sentences_by_words(sentences, max_words=max_words_per_segment)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / "voice_prompt.wav"

                # prepare voice prompt file once if provided
                if voice_embedding is not None:
                    waveform = voice_embedding["waveform"]
                    if isinstance(waveform, torch.Tensor):
                        w = waveform
                    else:
                        w = torch.tensor(waveform)

                    # Normalize dims to (channels, time)
                    if w.ndim == 3:
                        w = w.squeeze(0)
                    if w.ndim == 1:
                        w = w.unsqueeze(0)

                    if voice_embedding["sample_rate"] != model_sample_rate:
                        w = torchaudio.functional.resample(
                            w, voice_embedding["sample_rate"], model_sample_rate
                        )

                    torchaudio.save(str(temp_path), w, model_sample_rate)
                    print(f"Temporary voice file saved to: {temp_path}")

                out_wavs = []
                for seg in segments:
                    if voice_embedding is not None:
                        wav_seg = model.generate(
                            text=seg,
                            audio_prompt_path=str(temp_path),
                            exaggeration=exaggeration,
                            temperature=temperature,
                            cfg_weight=cfg_weight,
                            min_p=min_p,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                        )
                    else:
                        wav_seg = model.generate(
                            text=seg,
                            exaggeration=exaggeration,
                            temperature=temperature,
                            cfg_weight=cfg_weight,
                            min_p=min_p,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                        )

                    # Normalize returned waveform to a torch tensor with shape (channels, time)
                    if isinstance(wav_seg, torch.Tensor):
                        seg_w = wav_seg
                    else:
                        seg_w = torch.tensor(wav_seg)

                    # Handle possible shapes:
                    # - (batch, channels, time) -> take first batch
                    # - (channels, time) -> ok
                    # - (time,) -> convert to (1, time)
                    # - (batch, time) or other -> try to reshape to (1, time)
                    if seg_w.ndim == 3:
                        seg_w = seg_w[0]
                    elif seg_w.ndim == 1:
                        seg_w = seg_w.unsqueeze(0)
                    elif seg_w.ndim == 2:
                        # assume (channels, time) or (batch, time); if batch-like, keep as-is
                        pass
                    else:
                        seg_w = seg_w.reshape(1, -1)

                    out_wavs.append(seg_w)

                if len(out_wavs) == 0:
                    final_wav = torch.zeros(1, 0)
                else:
                    # concatenate along time axis (dim=1) expecting (channels, time)
                    final_wav = torch.cat(out_wavs, dim=1)

            # Return with a leading batch dimension (1, channels, time)
            return ({"waveform": final_wav.unsqueeze(0), "sample_rate": model_sample_rate},)
        except Exception as e:
            print(f"[VoiceCloneNode] Generation error: {e}")
            return ({"waveform": torch.zeros(1, 1), "sample_rate": model_sample_rate},)


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
