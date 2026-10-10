"""Exp3 - audio feature extraction over timestamps (numpy + soundfile for MP3)."""
import os
import struct
import wave

import numpy as np


def _load_wav_wave(filepath):
    """Fast stdlib WAV loader. Returns (mono float32 data, samplerate)."""
    with wave.open(filepath, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 1:  # 8-bit unsigned
        data = np.array(struct.unpack(f"{n_frames * n_channels}B", raw),
                        dtype=np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:  # 16-bit signed
        data = np.array(struct.unpack(f"<{n_frames * n_channels}h", raw),
                        dtype=np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth} (use 8/16-bit WAV)")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate


def _load_audio(filepath):
    """Load WAV via stdlib; MP3/OGG/FLAC via soundfile. Returns (mono, sr)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".wav":
        try:
            return _load_wav_wave(filepath)
        except Exception:
            pass  # fall through to soundfile for odd WAV encodings
    try:
        import soundfile as sf
    except ImportError:
        raise ValueError(f"{ext or 'audio'} needs `pip install soundfile` "
                         "(WAV works without it).")
    data, sr = sf.read(filepath, dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def extract_audio_features(filepath, window_sec=1.0, min_samples=10):
    """Split audio into windows and extract 3 features per timestamp.

    Supports WAV (stdlib) + MP3/OGG/FLAC (via soundfile).
    Returns dict with sr/duration/window/rows.
    Row: {timestamp, t_start, rms, zcr, centroid}
    """
    data, framerate = _load_audio(filepath)
    duration = len(data) / float(framerate)

    if duration / window_sec < min_samples:  # guarantee >=10 timestamps
        window_sec = duration / min_samples
    win = max(1, int(framerate * window_sec))
    n_win = int(len(data) // win)

    rows = []
    for i in range(n_win):
        seg = data[i * win:(i + 1) * win]
        t_start = round(i * window_sec, 2)
        t_end = round((i + 1) * window_sec, 2)
        rms = float(np.sqrt(np.mean(seg ** 2)))                       # loudness
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0)     # noisiness
        windowed = seg * np.hanning(len(seg))                         # brightness
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / framerate)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-10))
        rows.append({"timestamp": f"{t_start:.2f} - {t_end:.2f} s",
                     "t_start": t_start, "rms": round(rms, 4),
                     "zcr": round(zcr, 4), "centroid": round(centroid, 1)})
    return {"sr": framerate, "channels": 1 if data.ndim == 1 else data.shape[1],
            "duration": round(duration, 2), "window": round(window_sec, 2),
            "rows": rows}
