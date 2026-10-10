"""Exp3 - video feature extraction over frames (Pillow + numpy; cv2 optional).

GIF path works with current lab deps (no opencv needed).
MP4/AVI path uses cv2 only if installed, else raises a clear message.
Features per frame timestamp: brightness, contrast, motion vs prev frame.
Application: lecture-video scene analysis (static slide vs demo vs cut).
"""
import os

import numpy as np
from PIL import Image


def _frame_feats(gray, rgb, prev_gray):
    gray = gray.astype(np.float32)
    brightness = float(gray.mean())                       # mean intensity
    contrast = float(gray.std())                          # intensity spread
    gx = np.abs(np.diff(gray, axis=1)).mean()             # edge density
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge = round(float((gx + gy) / 2), 2)
    if prev_gray is None:
        motion = 0.0
    else:
        motion = float(np.abs(gray - prev_gray).mean())   # inter-frame change
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    colorfulness = round(float(np.abs(r - g).mean() + np.abs(g - b).mean()), 1)
    return round(brightness, 1), round(contrast, 1), round(motion, 2), edge, colorfulness


def _extract_gif(filepath, max_frames=12):
    im = Image.open(filepath)
    frame_ms = im.info.get("duration", 200) or 200
    n = getattr(im, "n_frames", 1)
    if n <= max_frames:
        idx = list(range(n))  # take every frame; short GIFs keep all frames
    else:
        idx = sorted(set(np.linspace(0, n - 1, max(max_frames, 10)).astype(int).tolist()))
    rows, prev = [], None
    for k, fi in enumerate(idx):
        im.seek(fi)
        fr = im.convert("RGB").resize((160, 120))
        gray = np.array(fr.convert("L"))
        rgb = np.array(fr)
        b, c, m, e, col = _frame_feats(gray, rgb, prev)
        ts = fi * frame_ms / 1000.0
        rows.append({"frame": fi, "timestamp": f"{ts:.2f} s",
                     "t_start": round(ts, 2), "brightness": b,
                     "contrast": c, "motion": m, "edge": e,
                     "colorfulness": col})
        prev = gray.astype(np.float32)
    dur = n * frame_ms / 1000.0
    return {"source": "GIF/Pillow", "n_frames_total": n,
            "duration": round(dur, 2), "rows": rows}


def _extract_cv2(filepath, max_frames=12):
    import cv2  # optional
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not open video (unsupported codec?).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    take = sorted(set(np.linspace(0, max(0, total - 1),
                                  min(total or max_frames, max(10, max_frames))).astype(int).tolist()))
    if not take:
        raise ValueError("No frames found.")
    rows, prev, k = [], None, 0
    for fi in take:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        b, c, m, e, col = _frame_feats(gray, rgb, prev)
        rows.append({"frame": fi, "timestamp": f"{fi / fps:.2f} s",
                     "t_start": round(fi / fps, 2), "brightness": b,
                     "contrast": c, "motion": m, "edge": e,
                     "colorfulness": col})
        prev = gray.astype(np.float32)
        k += 1
    cap.release()
    return {"source": "Video/cv2", "n_frames_total": total,
            "duration": round(total / fps, 2) if total else 0.0, "rows": rows}


def extract_video_features(filepath, max_frames=12):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".gif", ".png", ".apng", ".webp", ".bmp", ".jpg", ".jpeg"):
        return _extract_gif(filepath, max_frames)
    try:
        return _extract_cv2(filepath, max_frames)
    except ImportError:
        raise ValueError("MP4/AVI needs opencv (pip install opencv-python-headless). "
                         "Otherwise upload a .gif.")
