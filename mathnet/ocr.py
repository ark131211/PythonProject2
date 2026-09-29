"""OCR: картинка строки с задачей -> текст в формате MathNet (например "x^2-5x+6=0").

Свёрточная сеть (CRNN без рекуррентной части) с CTC-выходом. Здесь — общая
подготовка изображения и инференс на чистом NumPy; обучение — в train_ocr.py.
"""
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

OCR_CHARS = "0123456789+-*/^()=x'%от|√<>,yalog"  # индекс 0 зарезервирован под CTC-blank
OCR_H, OCR_W = 32, 320


def prepare(img: Image.Image) -> np.ndarray:
    """PIL-картинка -> (32, 320) float32, чернила ≈1, фон ≈0, текст прижат влево."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1)
    if np.median(a) > 0.5:  # тёмный текст на светлом фоне -> инвертируем
        a = 1 - a

    # обрезаем поля по «чернилам», оставив немного места вокруг
    ink = a > 0.5
    rows = np.where(ink.sum(1) >= max(1, 0.005 * a.shape[1]))[0]
    cols = np.where(ink.sum(0) >= 1)[0]
    if len(rows) and len(cols):
        m = max(2, int(0.15 * (rows[-1] - rows[0] + 1)))
        a = a[max(rows[0] - m, 0): rows[-1] + m + 1, max(cols[0] - m, 0): cols[-1] + m + 1]

    h, w = a.shape
    new_w = min(OCR_W, max(8, round(w * OCR_H / h)))
    small = Image.fromarray((a * 255).astype(np.uint8)).resize((new_w, OCR_H), Image.BILINEAR)
    out = np.zeros((OCR_H, OCR_W), np.float32)
    out[:, :new_w] = np.asarray(small, np.float32) / 255
    return out


def ctc_greedy(logits: np.ndarray) -> tuple[str, float]:
    """logits (T, n_classes) -> (текст, уверенность)."""
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    best = probs.argmax(1)
    text, prev, conf = [], 0, 1.0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            text.append(OCR_CHARS[k - 1])
            conf *= float(probs[t, k])
        prev = k
    return "".join(text), conf


# ---------------------------------------------------------------- NumPy-инференс

def conv2d(x, w, b):
    """x (C,H,W), w (O,C,kh,kw), b (O,), 'same'-паддинг."""
    O, C, kh, kw = w.shape
    x = np.pad(x, ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2)))
    win = sliding_window_view(x, (kh, kw), axis=(1, 2))  # C,H,W,kh,kw
    H, W = win.shape[1:3]
    cols = win.transpose(1, 2, 0, 3, 4).reshape(H * W, C * kh * kw)
    return (cols @ w.reshape(O, -1).T + b).T.reshape(O, H, W)


def maxpool(x, ph, pw):
    C, H, W = x.shape
    return x[:, : H // ph * ph, : W // pw * pw].reshape(C, H // ph, ph, W // pw, pw).max((2, 4))


class OCRNet:
    def __init__(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        data = np.load(weights_path)
        self.w = {k: data[k].astype(np.float32) for k in data.files}
        self.pools = json.loads(weights_path.with_suffix(".json").read_text())["pools"]

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x (32, 320) -> (T, n_classes)."""
        w, h = self.w, x[None]
        for i, (ph, pw) in enumerate(self.pools):
            h = np.maximum(conv2d(h, w[f"conv{i}.weight"], w[f"conv{i}.bias"]), 0)
            if ph > 1 or pw > 1:
                h = maxpool(h, ph, pw)
        h = h[:, 0, :]  # (C, T) — высота схлопнута до 1
        i = 0
        while f"seq{i}.weight" in w:
            h = np.maximum(conv2d(h[:, None], w[f"seq{i}.weight"][:, :, None], w[f"seq{i}.bias"])[:, 0], 0)
            i += 1
        return h.T @ w["head.weight"].T + w["head.bias"]

    def recognize(self, img: Image.Image) -> tuple[str, float]:
        return ctc_greedy(self.logits(prepare(img)))
