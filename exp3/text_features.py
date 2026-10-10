"""Exp3 - text feature extraction over chunks (stdlib only).

Treats each sentence/chunk as a timestamp sample so we get >=10 samples.
Features per sample: word count, avg word length, sentiment polarity.
Application: review/transcript sentiment trend over document position.
"""
import re

POS = {"good", "great", "excellent", "love", "loved", "amazing", "awesome",
       "best", "nice", "happy", "like", "liked", "wonderful", "perfect",
       "brilliant", "fantastic", "enjoy", "enjoyed", "recommend", "superb"}
NEG = {"bad", "poor", "terrible", "awful", "hate", "hated", "worst", "boring",
       "disappoint", "disappointed", "waste", "horrible", "annoying", "slow",
       "weak", "lame", "flop", "dull", "pathetic", "useless"}


def _polarity(words):
    if not words:
        return 0.0
    p = sum(1 for w in words if w in POS)
    n = sum(1 for w in words if w in NEG)
    return round((p - n) / len(words), 3)


def extract_text_features(filepath, min_samples=10):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = text.strip()
    if not text:
        raise ValueError("Empty text file.")

    sentences = [s.strip() for s in
                 re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= min_samples:
        chunks = sentences
        mode = "sentence"
    else:  # fall back to equal word-chunks so we always reach >=10 samples
        words_all = text.split()
        k = min_samples
        size = max(1, (len(words_all) + k - 1) // k)
        chunks = [" ".join(words_all[i:i + size])
                  for i in range(0, len(words_all), size)]
        mode = "word-chunk"

    rows = []
    for i, ch in enumerate(chunks):
        words = re.findall(r"[A-Za-z']+", ch.lower())
        wc = len(words)
        avg_len = round(sum(len(w) for w in words) / wc, 2) if wc else 0.0
        pol = _polarity(words)
        rows.append({"sample": f"{i + 1:02d} ({mode} {i + 1}/{len(chunks)})",
                     "t_start": i, "words": wc, "avg_len": avg_len,
                     "polarity": pol,
                     "preview": ch[:90] + ("..." if len(ch) > 90 else "")})
    return {"n_chars": len(text), "n_samples": len(rows),
            "mode": mode, "rows": rows, "full_text": text}
