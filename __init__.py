import random
import torch
import tempfile
import numpy as _np
import scipy.io.wavfile as scipy_wavfile
import scipy.signal as scipy_signal
from math import gcd
from pathlib import Path
import re
from .chatterbox.tts import ChatterboxTTS
from .chatterbox.tts_turbo import ChatterboxTurboTTS
from .chatterbox.vc import ChatterboxVC
import perth
import librosa


# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

DEFAULT_TEXT = "1 2,\nHello, and welcome to my course teaching how to generate images and video using Comfy U-I. This is amazing stuff, so lets get started."


class NullWatermarker:
    def apply_watermark(self, signal, sample_rate):
        print(
            "Using NullWatermarker: no watermark applied. sample_rate=" + str(sample_rate))
        return signal


class VoiceCloneNode:

    """Custom TTS node that clones voice from a reference audio and speaks entered text."""

    def __init__(self):
        super().__init__()
        self.clone_model = None
        self.model_path = Path("ComfyUI/models/tts/chatterbox")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DEFAULT_TEXT}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number",
                    "control_after_generate": "increment"
                }),
                # "exaggeration": ("FLOAT", {
                #     "default": 0.4,
                #     "min": 0.25,
                #     "max": 2.0,
                #     "step": 0.05
                # }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.15,
                    "max": 2.0,
                    "step": 0.05
                }),
                # "cfg_weight": ("FLOAT", {
                #     "default": 0.7,
                #     "min": 0.05,
                #     "max": 1.0,
                #     "step": 0.05
                # }),
                # "min_p": ("FLOAT", {
                #     "default": 0.00,
                #     "min": 0.0,
                #     "max": 1.0,
                #     "step": 0.01
                # }),
                "top_p": ("FLOAT", {
                    "default": .95,
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
                "top_k": ("FLOAT", {
                    "default": 1000,
                    "min": 0,
                    "max": 1000,
                    "step": 10
                }),
                "normalize": ("BOOLEAN", {
                    "default": False
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
        seed,
        temperature,
        top_p,
        repetition_penalty,
        top_k,
        normalize,
        disable_watermark,
        voice_embedding=None,
    ):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        _np.random.seed(seed)

        self.clone_model = ChatterboxTurboTTS.from_local(
            str(self.model_path), device=device)

        if disable_watermark:
            self.clone_model.watermarker = NullWatermarker()

        if voice_embedding is not None:
            waveform = voice_embedding["waveform"]

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_path = Path(tmpdir) / "voice_prompt.wav"

                    scipy_wavfile.write(
                        str(temp_path), self.clone_model.sr, waveform.numpy())

                    wav = self.clone_model.generate(text,
                                                    audio_prompt_path=temp_path,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    top_k=int(top_k),
                                                    repetition_penalty=repetition_penalty,
                                                    norm_loudness=normalize,
                                                    )
            except Exception as e:
                print(
                    f"[VoiceCloneNode] Generation error with voice prompt: {e}")
                return ({"waveform": torch.zeros(1), "sample_rate": self.clone_model.sr},)

        else:
            wav = self.clone_model.generate(text,
                                            temperature=temperature,
                                            top_p=top_p,
                                            top_k=int(top_k),
                                            repetition_penalty=repetition_penalty,
                                            norm_loudness=True,
                                            )

        return ({"waveform": wav.unsqueeze(0), "sample_rate": self.clone_model.sr},)


class VoiceReplaceNode:

    """Custom node that replaces a voice with a reference voice."""

    def __init__(self):
        super().__init__()
        self.replace_model = None
        self.model_path = Path("ComfyUI/models/tts/chatterbox")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "disable_watermark": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "SBCODE"

    def generate(
        self,
        audio,
        target_voice,
        disable_watermark=False,
    ):

        self.replace_model = ChatterboxVC.from_local(
            str(self.model_path), device=device)

        if disable_watermark:
            self.replace_model.watermarker = NullWatermarker()

        # try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            target_voice_path = Path(tmpdir) / "target_voice.wav"

            audio_waveform = audio["waveform"]
            target_voice_waveform = target_voice["waveform"]

            # Resample and convert to mono if necessary
            def process_waveform(waveform_tensor, orig_sr, target_sr):
                import librosa
                # remove batch dim, [channels, samples]
                waveform = waveform_tensor.squeeze(0)
                if waveform.dim() == 1:  # mono
                    audio_np = waveform.numpy()
                # [1, samples]
                elif waveform.dim() == 2 and waveform.shape[0] == 1:
                    audio_np = waveform.squeeze(0).numpy()
                # stereo [2, samples]
                elif waveform.dim() == 2 and waveform.shape[0] == 2:
                    # Convert to mono by averaging channels
                    audio_np = waveform.mean(dim=0).numpy()
                else:
                    # Fallback, flatten or something
                    audio_np = waveform.flatten().numpy()

                # Resample to target_sr
                if orig_sr != target_sr:
                    audio_np = librosa.resample(
                        audio_np, orig_sr=orig_sr, target_sr=target_sr)
                return audio_np

            audio_sr = audio.get('sample_rate', self.replace_model.sr)
            target_sr = target_voice.get('sample_rate', self.replace_model.sr)

            audio_processed = process_waveform(
                audio_waveform, audio_sr, self.replace_model.sr)
            target_processed = process_waveform(
                target_voice_waveform, target_sr, self.replace_model.sr)

            scipy_wavfile.write(
                str(audio_path), self.replace_model.sr, audio_processed)

            scipy_wavfile.write(
                str(target_voice_path), self.replace_model.sr, target_processed)

            out_wav = self.replace_model.generate(
                audio=str(audio_path),
                target_voice_path=str(target_voice_path),
            )

        return ({"waveform": out_wav.unsqueeze(0), "sample_rate": self.replace_model.sr},)
        # except Exception as e:
        #    print(f"[VoiceReplaceNode] Generation error: {e}")
        #    return ({"waveform": torch.zeros(1), "sample_rate": self.replace_model.sr},)


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
                       "VoiceReplaceNode": VoiceReplaceNode,
                       "DetectWatermarkNode": DetectWatermarkNode
                       }

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceCloneNode": "Voice Clone",
    "VoiceReplaceNode": "Voice Replace",
    "DetectWatermarkNode": "Detect Watermark"
}
