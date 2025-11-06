from random import random
import torch
import tempfile
import numpy as _np
import scipy.io.wavfile as scipy_wavfile
import scipy.signal as scipy_signal
from math import gcd
from pathlib import Path
import re
from .chatterbox.tts import ChatterboxTTS
from .chatterbox.vc import ChatterboxVC
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
            self.model = ChatterboxTTS.from_local(
                str(self.model_path), device=device)

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
        segments = chunk_sentences_by_words(
            sentences, max_words=max_words_per_segment)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / "voice_prompt.wav"

                # prepare voice prompt file once if provided
                if voice_embedding is not None:
                    waveform = voice_embedding["waveform"]
                    # Always ensure we have a tensor first
                    if isinstance(waveform, torch.Tensor):
                        print("waveform is tensor, shape=", waveform.shape)
                        w = waveform.clone()  # Make a copy to avoid modifying the input
                    else:
                        print("waveform is not tensor, type=", type(waveform))
                        w = torch.tensor(waveform)
                        print("converted shape=", w.shape)

                    # Normalize dims to (channels, samples)
                    print("Initial waveform shape:", w.shape)
                    
                    # Handle all possible shapes to get to (channels, samples)
                    if w.ndim == 3:  # (batch, channels, samples)
                        w = w.squeeze(0)  # -> (channels, samples)
                    elif w.ndim == 1:  # (samples,)
                        w = w.unsqueeze(0)  # -> (1, samples) mono
                    
                    # If we have a 2D tensor, ensure it's (channels, samples) not (samples, channels)
                    if w.ndim == 2:
                        if w.shape[0] > w.shape[1]:  # Likely (samples, channels)
                            w = w.t()  # -> (channels, samples)
                        
                    # Ensure proper channel count (1 for mono, 2 for stereo)
                    if w.shape[0] > 2:
                        w = w[:2]  # Keep only first two channels if more
                    
                    print("Normalized waveform shape:", w.shape)
                    
                    # Ensure we have a 2D tensor with shape (channels, time)
                    if w.ndim != 2:
                        w = w.view(1, -1)  # Force to (1, time) if shape is unexpected
                    
                    if voice_embedding["sample_rate"] != model_sample_rate:
                        src_sr = int(voice_embedding["sample_rate"])
                        tgt_sr = int(model_sample_rate)
                        print("Resampling from", src_sr, "to", tgt_sr)
                        # Use resample_poly for better quality. Reduce ratio by gcd.
                        w_np = w.cpu().numpy()
                        up = tgt_sr
                        down = src_sr
                        g = gcd(up, down)
                        up //= g
                        down //= g
                        w_res = scipy_signal.resample_poly(w_np, up, down, axis=-1)
                        w = torch.from_numpy(w_res).to(dtype=w.dtype)
                        print("Resampled shape:", w.shape)

                    print("Final waveform shape before save:", w.shape)
                    try:
                        # Ensure proper shape and format for WAV save using scipy
                        if not isinstance(w, torch.Tensor):
                            w = torch.tensor(w)
                        w = w.cpu().float()  # Ensure CPU tensor and float dtype

                        if w.ndim == 1:
                            w = w.unsqueeze(0)  # Add channels dimension
                        elif w.ndim == 3:
                            w = w.squeeze(0)  # Remove batch dimension

                        # Ensure we have (channels, samples)
                        if w.ndim != 2:
                            w = w.reshape(1, -1)

                        # Save as WAV file using scipy (shape -> (samples, channels))
                        temp_path = temp_path.with_suffix('.wav')
                        print(f"Saving audio to {temp_path}, shape={w.shape}, dtype={w.dtype}")
                        out_np = w.numpy().T  # (samples, channels)
                        # Convert to 16-bit PCM
                        out_np = _np.clip(out_np, -1.0, 1.0)
                        out_int16 = (_np.round(out_np * 32767.0)).astype(_np.int16)
                        scipy_wavfile.write(str(temp_path), model_sample_rate, out_int16)

                    except Exception as e:
                        print(f"Error saving temporary audio file: {e}")
                        raise
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
                    print("Created empty waveform")
                else:
                    # concatenate along time axis (dim=1) expecting (channels, time)
                    final_wav = torch.cat(out_wavs, dim=1)
                    print(f"Concatenated waveform shape: {final_wav.shape}")

                # final_wav is expected by ComfyUI to have a leading batch dimension
                # i.e. shape (batch, channels, samples). We'll return batch dimension = 1.
                if final_wav.ndim == 3:
                    out_wav = final_wav
                elif final_wav.ndim == 2:
                    out_wav = final_wav.unsqueeze(0)  # (1, channels, samples)
                elif final_wav.ndim == 1:
                    out_wav = final_wav.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
                else:
                    out_wav = final_wav.reshape(1, 1, -1)

                out_wav = out_wav.detach().cpu()
                print(f"Final returned waveform shape (with batch): {out_wav.shape}, dtype: {out_wav.dtype}")

                return ({"waveform": out_wav, "sample_rate": model_sample_rate},)
        except Exception as e:
            print(f"[VoiceCloneNode] Generation error: {e}")
            return ({"waveform": torch.zeros(1), "sample_rate": model_sample_rate},)


class VoiceReplaceNode:

    """Custom node that replaces a voice with a reference voice."""

    def __init__(self):
        super().__init__()
        self.model = None
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
        # lazy-load model once per node instance
        if getattr(self, "model", None) is None:
            self.model = ChatterboxVC.from_local(
                str(self.model_path), device=device)

        model = self.model
        model_sample_rate = 24000
        if disable_watermark:
            model.watermarker = NullWatermarker()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = Path(tmpdir) / "audio.wav"
                target_voice_path = Path(tmpdir) / "target_voice.wav"

                audio_waveform = audio["waveform"]
                target_voice_waveform = target_voice["waveform"]

                if isinstance(audio_waveform, torch.Tensor):
                    aw = audio_waveform
                else:
                    aw = torch.tensor(audio_waveform)

                if isinstance(target_voice_waveform, torch.Tensor):
                    taw = target_voice_waveform
                else:
                    taw = torch.tensor(target_voice_waveform)

                # Normalize dims to (channels, time)
                if aw.ndim == 3:
                    aw = aw.squeeze(0)
                if aw.ndim == 1:
                    aw = aw.unsqueeze(0)

                if taw.ndim == 3:
                    taw = taw.squeeze(0)
                if taw.ndim == 1:
                    taw = taw.unsqueeze(0)

                if audio["sample_rate"] != model_sample_rate:
                    src_sr_aw = int(audio["sample_rate"])
                    tgt_sr = int(model_sample_rate)
                    aw_np = aw.cpu().numpy()
                    up_aw = tgt_sr
                    down_aw = src_sr_aw
                    g_aw = gcd(up_aw, down_aw)
                    up_aw //= g_aw
                    down_aw //= g_aw
                    aw_res = scipy_signal.resample_poly(aw_np, up_aw, down_aw, axis=-1)
                    aw = torch.from_numpy(aw_res).to(dtype=aw.dtype)

                if target_voice["sample_rate"] != model_sample_rate:
                    src_sr_taw = int(target_voice["sample_rate"])
                    tgt_sr = int(model_sample_rate)
                    taw_np = taw.cpu().numpy()
                    up_taw = tgt_sr
                    down_taw = src_sr_taw
                    g_taw = gcd(up_taw, down_taw)
                    up_taw //= g_taw
                    down_taw //= g_taw
                    taw_res = scipy_signal.resample_poly(taw_np, up_taw, down_taw, axis=-1)
                    taw = torch.from_numpy(taw_res).to(dtype=taw.dtype)

                # Save temp files using scipy
                try:
                    aw_out = aw.cpu().float()
                    if aw_out.ndim == 1:
                        aw_out = aw_out.unsqueeze(0)
                    aw_np_out = aw_out.numpy().T
                    aw_np_out = _np.clip(aw_np_out, -1.0, 1.0)
                    aw_int16 = (_np.round(aw_np_out * 32767.0)).astype(_np.int16)
                    scipy_wavfile.write(str(audio_path), model_sample_rate, aw_int16)
                    print(f"Temporary audio file saved to: {audio_path}")
                except Exception as _e:
                    print(f"Failed to save audio_path with scipy: {_e}")

                try:
                    taw_out = taw.cpu().float()
                    if taw_out.ndim == 1:
                        taw_out = taw_out.unsqueeze(0)
                    taw_np_out = taw_out.numpy().T
                    taw_np_out = _np.clip(taw_np_out, -1.0, 1.0)
                    taw_int16 = (_np.round(taw_np_out * 32767.0)).astype(_np.int16)
                    scipy_wavfile.write(str(target_voice_path), model_sample_rate, taw_int16)
                    print(f"Temporary target voice file saved to: {target_voice_path}")
                except Exception as _e:
                    print(f"Failed to save target_voice_path with scipy: {_e}")

                out_wav = model.generate(
                    audio=str(audio_path),
                    target_voice_path=str(target_voice_path),
                )

            return ({"waveform": out_wav.unsqueeze(0), "sample_rate": model_sample_rate},)
        except Exception as e:
            print(f"[VoiceReplaceNode] Generation error: {e}")
            return ({"waveform": torch.zeros(1), "sample_rate": model_sample_rate},)


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
