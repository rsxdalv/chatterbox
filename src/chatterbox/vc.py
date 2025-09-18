# chatterbox_vc.py
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen

REPO_ID = "ResembleAI/chatterbox"


# --------------------------
# Audio utilities (no new deps)
# --------------------------

def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def _match_rms(x: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    """
    Simple RMS normalization to a linear target RMS (default ≈ -30 dBFS region).
    Keeps peak safety with a small limiter.
    """
    r = _rms(x)
    if r < 1e-9:
        return x
    y = x * (target_rms / r)
    # Simple peak limiter to avoid clipping
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 0.999:
        y = y / peak * 0.999
    return y.astype(np.float32)

def _butter_hp_sos(sr: int, cutoff_hz: float = 90.0) -> np.ndarray:
    """
    2nd-order Butterworth high-pass SOS design (manual, minimal).
    We rely on bilinear transform for stability. For simplicity and to avoid
    bringing in scipy, we precompute with a tiny internal helper.
    """
    # This small helper computes biquad HPF coefficients using RBJ cookbook formulas.
    # Return in SOS form [[b0,b1,b2,a0,a1,a2]] compatible with manual sosfilt below.
    # Ref: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    w0 = 2.0 * np.pi * (cutoff_hz / sr)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    Q = np.sqrt(0.5)  # ~ Butterworth
    alpha = sin_w0 / (2.0 * Q)

    b0 =  (1 + cos_w0) / 2.0
    b1 = -(1 + cos_w0)
    b2 =  (1 + cos_w0) / 2.0
    a0 =   1 + alpha
    a1 =  -2 * cos_w0
    a2 =   1 - alpha

    # Normalize
    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0

    # Convert to simple SOS single section
    sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float32)
    return sos

def _sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Minimal SOS filter (Direct Form II Transposed) for one or more sections.
    """
    y = x.astype(np.float32).copy()
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        # a0 assumed 1.0 due to normalization above
        z1 = 0.0
        z2 = 0.0
        out = np.empty_like(y)
        for i, v in enumerate(y):
            w = v - a1 * z1 - a2 * z2
            o = b0 * w + b1 * z1 + b2 * z2
            z2 = z1
            z1 = w
            out[i] = o
        y = out
    return y

def _hp_filter(y: np.ndarray, sr: int, cutoff_hz: float = 90.0) -> np.ndarray:
    sos = _butter_hp_sos(sr, cutoff_hz=cutoff_hz)
    return _sosfilt(sos, y)

def _voiced_center_crop(y: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """
    Choose a window of 'target_len' samples centered within voiced regions.
    Falls back to head-crop if insufficient voiced content.
    """
    if len(y) <= target_len:
        return y

    # Find voiced intervals using energy-based splitting
    intervals = librosa.effects.split(y, top_db=30)  # moderate VAD
    if len(intervals) == 0:
        return y[:target_len]

    # Build a boolean mask of voiced samples
    mask = np.zeros(len(y), dtype=bool)
    for s, e in intervals:
        mask[s:e] = True

    voiced_idx = np.flatnonzero(mask)
    if len(voiced_idx) < target_len:
        # Use the longest voiced interval expanded around
        longest = max(intervals, key=lambda x: x[1] - x[0])
        start = max(0, min(len(y) - target_len, (longest[0] + longest[1]) // 2 - target_len // 2))
        return y[start:start + target_len]

    # Center a window around the middle voiced index
    mid = int(voiced_idx[len(voiced_idx) // 2])
    start = max(0, min(len(y) - target_len, mid - target_len // 2))
    return y[start:start + target_len]

def _merge_ref_dicts(dicts):
    """
    Merge a list of ref_dicts. Float/complex tensors are averaged (on the
    first tensor's device, upcast to float32 for stability). Integer tensors
    (e.g., Long ids) and non-tensors keep the first occurrence.
    """
    if not dicts:
        return None

    all_keys = set().union(*(d.keys() for d in dicts))
    out = {}

    for k in all_keys:
        entries = [d[k] for d in dicts if k in d]

        # If any entry is not a tensor, keep the first as-is
        if not all(torch.is_tensor(t) for t in entries):
            out[k] = entries[0]
            continue

        # Float/complex → average; integer → keep first
        if all(t.is_floating_point() or t.is_complex() for t in entries):
            base = entries[0]
            base_dev = base.device
            float_tensors = [t.to(device=base_dev, dtype=torch.float32) for t in entries]
            out[k] = torch.stack(float_tensors, dim=0).mean(dim=0)
        else:
            out[k] = entries[0]

    return out

class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
        *,
        # Fine-tuning knobs (safe defaults):
        ref_highpass_hz: Optional[float] = 90.0,
        ref_target_rms: Optional[float] = 0.03,  # ~ -30 dBFS region
        src_target_rms: Optional[float] = None,  # keep None to leave source as-is
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        self.ref_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in (ref_dict or {}).items()} if ref_dict else None

        # Store knobs
        self._ref_highpass_hz = ref_highpass_hz
        self._ref_target_rms = ref_target_rms
        self._src_target_rms = src_target_rms

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states.get('gen', None)

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        # pull local copies
        local_path = None
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    # --------------------------
    # Conditioning API
    # --------------------------
    def _prepare_single_ref(self, wav_fpath: str) -> Dict[str, torch.Tensor]:
        """
        Load a single reference WAV, apply optional cleanup/normalization, crop
        to DEC_COND_LEN, and embed to get ref_dict entries.
        """
        y, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        # Optional cleanup on reference
        if self._ref_highpass_hz is not None and self._ref_highpass_hz > 0:
            y = _hp_filter(y, S3GEN_SR, cutoff_hz=float(self._ref_highpass_hz))

        if self._ref_target_rms is not None and self._ref_target_rms > 0:
            y = _match_rms(y, target_rms=float(self._ref_target_rms))

        # Smarter crop: voiced-center into exactly DEC_COND_LEN samples
        target_len = int(self.DEC_COND_LEN)
        y = _voiced_center_crop(y, S3GEN_SR, target_len)

        # Ensure length exactly target_len (pad/truncate)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

        # Embed
        ref_dict = self.s3gen.embed_ref(y, S3GEN_SR, device=self.device)
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in ref_dict.items()}

    def set_target_voice(self, wav_fpath: str):
        """Set reference conditioning from a single WAV file."""
        self.ref_dict = self._prepare_single_ref(wav_fpath)

    def set_target_voices(self, wav_fpaths):
        refs = []
        for p in wav_fpaths:
            try:
                refs.append(self._prepare_single_ref(p))
            except Exception as e:
                print(f"[WARN] Skipping reference {p}: {e}")

        merged = _merge_ref_dicts(refs)
        if merged is None:
            raise RuntimeError("No valid reference audios provided.")

        # Ensure final placement on the model device
        for k, v in merged.items():
            if torch.is_tensor(v):
                merged[k] = v.to(device=self.device)
        self.ref_dict = merged

    # --------------------------
    # Inference
    # --------------------------
    def generate(
        self,
        audio: str,
        target_voice_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Convert `audio` to target voice. If `target_voice_path` is provided,
        it replaces the current conditioning for this call.
        """
        if target_voice_path:
            if isinstance(target_voice_path, str):
                target_voice_path = [target_voice_path]
            #self.set_target_voice(target_voice_path)
            self.set_target_voices(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `set_target_voice(s)` first or specify `target_voice_path`"

        with torch.inference_mode():
            # Tokenizer path expects S3_SR (16 kHz)
            y16, _ = librosa.load(audio, sr=S3_SR)
            if self._src_target_rms is not None and self._src_target_rms > 0:
                y16 = _match_rms(y16, target_rms=float(self._src_target_rms))

            audio_16 = torch.from_numpy(y16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)