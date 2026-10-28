"""Audio effects (numpy/scipy only) used to build stylised voices such as Cyn.

All effects take and return mono float32 arrays in [-1, 1] plus the sample rate.
A chain is a list of dicts as written in config.yaml: {type: pitch, semitones: -4}.

Standalone test:  python -m voices.effects   (applies the chain to a synthetic tone)
"""
import numpy as np
from scipy import signal


def pitch(x: np.ndarray, sr: int, semitones: float = 0.0) -> np.ndarray:
    """Resampling pitch shift. Lowering pitch also lengthens audio; tts.py compensates by
    synthesizing faster (see duration_factor)."""
    if abs(semitones) < 1e-3:
        return x
    f = 2.0 ** (semitones / 12.0)
    up, down = 1000, int(round(1000 * f))
    return signal.resample_poly(x, up, down).astype(np.float32)


def ring_mod(x: np.ndarray, sr: int, freq: float = 50.0, mix: float = 0.5) -> np.ndarray:
    t = np.arange(len(x), dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return ((1 - mix) * x + mix * x * carrier).astype(np.float32)


def bitcrush(x: np.ndarray, sr: int, bits: int = 10, downsample: int = 2) -> np.ndarray:
    y = x
    if downsample > 1:
        y = np.repeat(x[::downsample], downsample)[: len(x)]
    levels = 2 ** (bits - 1)
    return (np.round(y * levels) / levels).astype(np.float32)


def overdrive(x: np.ndarray, sr: int, drive: float = 2.0) -> np.ndarray:
    y = np.tanh(x * drive) / np.tanh(drive)
    return y.astype(np.float32)


def lowpass(x: np.ndarray, sr: int, cutoff: float = 6000.0) -> np.ndarray:
    sos = signal.butter(4, min(cutoff, sr / 2 - 100), btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff: float = 100.0) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def reverb(x: np.ndarray, sr: int, decay: float = 0.5, wet: float = 0.2) -> np.ndarray:
    """Convolution reverb with a synthetic exponentially-decaying noise impulse response."""
    rng = np.random.default_rng(7)  # fixed seed -> identical room every time
    n = int(sr * decay)
    if n < 16:
        return x
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.0 * np.arange(n) / n).astype(np.float32)
    ir = signal.sosfilt(signal.butter(2, 4500, btype="low", fs=sr, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    tail = signal.fftconvolve(x, ir)[: len(x) + n].astype(np.float32)
    dry = np.pad(x, (0, len(tail) - len(x)))
    return ((1 - wet) * dry + wet * tail).astype(np.float32)


_EFFECTS = {
    "pitch": pitch,
    "ring_mod": ring_mod,
    "bitcrush": bitcrush,
    "overdrive": overdrive,
    "lowpass": lowpass,
    "highpass": highpass,
    "reverb": reverb,
}


def duration_factor(chain: list[dict]) -> float:
    """Total duration multiplier introduced by pitch effects (e.g. 1.26 for -4 semitones)."""
    f = 1.0
    for fx in chain or []:
        if fx.get("type") == "pitch":
            f /= 2.0 ** (float(fx.get("semitones", 0)) / 12.0)
    return f


def apply_chain(x: np.ndarray, sr: int, chain: list[dict]) -> np.ndarray:
    y = x.astype(np.float32)
    for fx in chain or []:
        params = {k: v for k, v in fx.items() if k != "type"}
        fn = _EFFECTS.get(fx.get("type"))
        if fn is None:
            raise ValueError(f"Unknown effect type: {fx.get('type')}")
        y = fn(y, sr, **params)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak * 0.89  # normalise, leave headroom
    return y.astype(np.float32)


if __name__ == "__main__":
    sr = 22050
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    chain = [
        {"type": "pitch", "semitones": -4},
        {"type": "ring_mod", "freq": 55, "mix": 0.3},
        {"type": "bitcrush", "bits": 10, "downsample": 2},
        {"type": "overdrive", "drive": 2.2},
        {"type": "reverb", "decay": 0.5, "wet": 0.2},
    ]
    out = apply_chain(tone, sr, chain)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    print(f"OK: in {len(tone)} samples -> out {len(out)} samples, "
          f"duration_factor={duration_factor(chain):.3f}, peak={np.max(np.abs(out)):.2f}")
