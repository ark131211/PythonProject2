"""Синтетические рукописные строки задач для обучения OCR (нужно только для train_ocr.py).

Строка собирается из настоящих рукописных символов: цифры — MNIST, остальные
символы — HASYv2 (их рисовали сотни людей). Символов, которых в наборах нет
(=, скобки, запятая, штрих), рисуются процедурно «дрожащими» штрихами.
Часть авторов отложена для проверки: val=True берёт только их символы.

Данные: data_raw/ (MNIST *.gz и распакованный HASYv2).
"""
import csv
import gzip
import math
import random
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RAW = Path(__file__).resolve().parent.parent / "data_raw"

HASY_CLASSES = {  # символ метки -> классы HASY
    "+": ["+"], "-": ["-"], "*": ["\\cdot", "\\times", "\\ast"], "/": ["/", "\\div"],
    "|": ["|", "\\mid"], "√": ["\\sqrt{}"], "<": ["<"], ">": [">"], "<=": ["\\leq"], ">=": ["\\geq"],
    "x": ["x"], "y": ["y"], "a": ["a"], "l": ["l"], "o": ["o"], "g": ["g"], "'": ["\\prime"],
    **{str(d): [str(d)] for d in range(10)},
}
# высота символа относительно высоты цифры и сдвиг центра от средней линии (в долях высоты цифры)
SHAPE = {
    "x": (0.62, 0.2), "a": (0.62, 0.2), "o": (0.62, 0.2), "y": (0.85, 0.4), "g": (0.85, 0.4), "l": (1.1, -0.05),
    "+": (0.6, 0.0), "-": (0.08, 0.0), "=": (0.4, 0.0), "*": (0.45, 0.0), "/": (1.05, 0.0), "|": (1.3, 0.0),
    "(": (1.3, 0.0), ")": (1.3, 0.0), "√": (1.2, -0.05), "<": (0.6, 0.0), ">": (0.6, 0.0), "<=": (0.75, 0.0),
    ">=": (0.75, 0.0), ",": (0.3, 0.55), "'": (0.35, -0.55), "^": (0.35, -0.45),
}


def _crop(m: np.ndarray) -> np.ndarray:
    ys, xs = np.where(m > 0.2)
    if len(ys) == 0:
        return m
    return m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


@lru_cache(maxsize=None)
def glyph_bank(val: bool) -> dict[str, list[np.ndarray]]:
    """Словарь символ -> список масок (float, 1 = чернила)."""
    bank: dict[str, list[np.ndarray]] = {k: [] for k in HASY_CLASSES}
    for split, is_val in (("train", False), ("t10k", True)):  # в MNIST тест написан другими людьми
        if is_val != val:
            continue
        with gzip.open(RAW / f"{split}-images-idx3-ubyte.gz") as f:
            imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(RAW / f"{split}-labels-idx1-ubyte.gz") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        for d in range(10):
            for img in imgs[labels == d][:3000]:
                bank[str(d)].append(_crop(img.astype(np.float32) / 255))
    wanted = {cls: sym for sym, classes in HASY_CLASSES.items() for cls in classes}
    by_sym: dict[str, dict[str, list[str]]] = {}
    with open(RAW / "hasy-data-labels.csv") as f:
        for row in csv.DictReader(f):
            if (sym := wanted.get(row["latex"])) is not None:
                by_sym.setdefault(sym, {}).setdefault(row["user_id"], []).append(row["path"])
    for sym, users in by_sym.items():
        # для каждого символа откладываем в проверку авторов, пока не наберётся ~15% образцов
        order = sorted(users, key=lambda u: zlib.crc32(f"{sym}|{u}".encode()))  # стабильно между процессами
        total, taken, val_users = sum(map(len, users.values())), 0, set()
        for u in order:
            if taken >= 0.15 * total:
                break
            if taken + len(users[u]) <= 0.3 * total:
                val_users.add(u)
                taken += len(users[u])
        for u, paths in users.items():
            if (u in val_users) == val:
                for p in paths:
                    m = 1 - np.asarray(Image.open(RAW / p).convert("L"), np.float32) / 255
                    bank[sym].append(_crop(m))
    return bank


def stroke_glyph(sym: str) -> np.ndarray:
    """Процедурный «рукописный» символ: =, (, ), запятая, штрих, ^."""
    S = 64
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    w = random.randint(3, 7)
    j = lambda v: v + random.uniform(-3, 3)

    def curve(points):
        d.line([(j(x), j(y)) for x, y in points], fill=255, width=w, joint="curve")

    if sym == "=":
        tilt = random.uniform(-4, 4)
        for y in (22, 22 + random.uniform(14, 22)):
            curve([(6, y + tilt), (32, y + random.uniform(-2, 2)), (58, y - tilt)])
    elif sym in "()":
        pts = []
        bow = random.uniform(10, 20)
        for t in np.linspace(-1, 1, 9):
            x = 32 + (-bow if sym == "(" else bow) * (1 - t * t)
            pts.append((x, 32 + 28 * t))
        curve(pts)
    elif sym == ",":
        curve([(32, 20), (30, 34), (24, 48)])
    elif sym == "'":
        curve([(38, 10), (30, 40)])
    elif sym == "^":
        curve([(14, 44), (32, 14), (50, 44)])
    return _crop(np.asarray(img.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255)


def get_glyph(sym: str, val: bool) -> np.ndarray:
    bank = glyph_bank(val)
    if sym in "=(),^":
        return stroke_glyph(sym)
    if sym == "'" and random.random() < 0.5:
        return stroke_glyph(sym)
    return random.choice(bank[sym] or glyph_bank(False)[sym])  # у редких символов может не быть отложенных авторов


def tokens(q: str):
    """Разбор метки: (символ, уровень) и маркеры черты корня. Уровень 1 — степень, -1 — индекс."""
    out, i = [], 0
    caret_literal = random.random() < 0.05
    while i < len(q):
        if q.startswith("log", i):
            out += [("l", 0), ("o", 0), ("g", 0)]
            j = i + 3
            while j < len(q) and q[j].isdigit():
                out.append((q[j], -1))
                j += 1
            i = j
        elif q[i] == "^" and not caret_literal and i + 1 < len(q):
            if q[i + 1] == "(":
                depth, j = 0, i + 1
                while True:
                    depth += {"(": 1, ")": -1}.get(q[j], 0)
                    if depth == 0:
                        break
                    j += 1
                out += [(c, 1) for c in q[i + 2:j]]
                i = j + 1
            else:
                out.append((q[i + 1], 1))
                i += 2
        elif q[i] == "√" and i + 1 < len(q) and q[i + 1] == "(" and random.random() < 0.8:
            depth, j = 0, i + 1
            while True:
                depth += {"(": 1, ")": -1}.get(q[j], 0)
                if depth == 0:
                    break
                j += 1
            out += [("√", 0), ("BAR_START", 0)] + [(c, 0) for c in q[i + 2:j]] + [("BAR_END", 0)]
            i = j + 1
        elif q[i:i + 2] in ("<=", ">="):
            out.append((q[i:i + 2], 0))
            i += 2
        else:
            out.append((q[i], 0))
            i += 1
    return out


def render_hand(q: str, val: bool = False) -> np.ndarray:
    """Маска рукописной строки (float, 1 = чернила)."""
    H = random.randint(40, 64)                # высота цифры в пикселях
    slant = random.uniform(-0.25, 0.25)       # общий наклон почерка
    thick = random.choice([0, 0, 1, 1, 2])    # утолщение штриха
    canvas = np.zeros((H * 4, int(H * (len(q) + 4) * 1.1)), np.float32)
    x, base = H * 0.5, H * 2.3                # base — средняя линия строки
    bar_x0 = None
    drift = random.uniform(-0.06, 0.06)       # строка «уплывает» вверх/вниз
    for sym, level in tokens(q):
        if sym == "BAR_START":
            bar_x0 = x - H * 0.15
            continue
        if sym == "BAR_END":
            if bar_x0 is not None:
                y = int(base - H * 0.75)
                canvas[max(y - 1, 0): y + 2, int(bar_x0): int(x)] = 1
            bar_x0 = None
            continue
        g = get_glyph(sym, val)
        rel_h, dy = SHAPE.get(sym, (1.0, 0.0))
        scale = 0.58 if level else 1.0
        target_h = max(4, rel_h * H * scale * random.uniform(0.85, 1.15))
        if sym == "-":  # у минуса задаём ширину, а не высоту
            target_w = H * scale * random.uniform(0.45, 0.7)
            gh, gw = g.shape
            new = (max(2, int(target_w)), max(2, int(gh * target_w / gw * 0.6) + 2))
        else:
            gh, gw = g.shape
            ratio = min(gw / max(gh, 1), 1.6)
            new = (max(2, int(target_h * ratio)), max(2, int(target_h)))
        pil = Image.fromarray((g * 255).astype(np.uint8)).resize(new, Image.BILINEAR)
        pil = pil.rotate(random.uniform(-8, 8), resample=Image.BILINEAR, expand=True)
        if thick:
            pil = pil.filter(ImageFilter.MaxFilter(2 * thick + 1))
        m = np.asarray(pil, np.float32) / 255
        mh, mw = m.shape
        cy = base + dy * H + {0: 0, 1: -0.55 * H, -1: 0.4 * H}[level] + random.uniform(-0.06, 0.06) * H
        cy += drift * x
        y0, x0 = int(cy - mh / 2), int(x)
        if y0 < 0 or x0 + mw >= canvas.shape[1] or y0 + mh >= canvas.shape[0]:
            continue
        canvas[y0:y0 + mh, x0:x0 + mw] = np.maximum(canvas[y0:y0 + mh, x0:x0 + mw], m)
        x += mw + H * random.uniform(0.05, 0.3) * (0.6 if level else 1)
    canvas = canvas[:, : int(x + H)]
    # наклон почерка
    h, w = canvas.shape
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    img = img.transform((w, h), Image.AFFINE, (1, slant, -slant * h / 2, 0, 1, 0), resample=Image.BILINEAR)
    return np.asarray(img, np.float32) / 255, H
