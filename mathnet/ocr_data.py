"""Синтетические «фотографии» задач для обучения OCR (нужно только для train_ocr.py).

Задачи берутся из того же генератора, что и для MathNet, и рисуются системными
шрифтами macOS: степени — верхним индексом, умножение как · × *, деление как
/ : ÷, минус как - − –. Затем «портим» картинку: бумага, клетка, соседние строки,
поворот, наклон, размытие, шум, неравномерный свет, низкое разрешение, JPEG.
"""
import io
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .data import sample

SUP = Path("/System/Library/Fonts/Supplemental")
SYS = Path("/System/Library/Fonts")

TRAIN_FONTS = [
    SUP / "Arial.ttf", SUP / "Arial Bold.ttf", SUP / "Arial Italic.ttf", SUP / "Arial Narrow.ttf",
    SUP / "Times New Roman.ttf", SUP / "Times New Roman Bold.ttf", SUP / "Times New Roman Italic.ttf",
    SUP / "Georgia.ttf", SUP / "Georgia Bold.ttf", SUP / "Georgia Italic.ttf",
    SUP / "Courier New.ttf", SUP / "Courier New Bold.ttf", SUP / "Tahoma.ttf", SUP / "Tahoma Bold.ttf",
    SUP / "Trebuchet MS.ttf", SUP / "Trebuchet MS Bold.ttf", SUP / "Microsoft Sans Serif.ttf",
    SUP / "Comic Sans MS.ttf", SUP / "PTSans.ttc", SUP / "PTMono.ttc", SUP / "Baskerville.ttc",
    SUP / "Charter.ttc", SUP / "STIXTwoText.ttf", SUP / "Iowan Old Style.ttc", SUP / "Arial Unicode.ttf",
    SYS / "Helvetica.ttc", SYS / "Times.ttc", SYS / "Menlo.ttc", SYS / "SFNS.ttf", SYS / "NewYork.ttf",
]
# Шрифты, которых сеть не видит при обучении, — для честной проверки.
VAL_FONTS = [SUP / "Verdana.ttf", SUP / "Verdana Bold.ttf", SYS / "Palatino.ttc", SUP / "PTSerif.ttc"]

VARIANTS = {"-": ["-", "−", "–"], "*": ["·", "×", "*"], "/": ["/", ":", "÷"], "'": ["'", "′", "’"]}
VARIANT_W = {"-": [4, 5, 1], "*": [4, 4, 2], "/": [6, 3, 1], "'": [5, 3, 2]}


@lru_cache(maxsize=None)
def has_glyph(path: Path, ch: str) -> bool:
    f = ImageFont.truetype(str(path), 40)

    def mask(c):
        img = Image.new("L", (80, 80))
        ImageDraw.Draw(img).text((10, 10), c, font=f, fill=255)
        return np.asarray(img)

    m = mask(ch)
    return bool(m.any()) and not np.array_equal(m, mask(""))


@lru_cache(maxsize=512)
def font(path: Path, size: int):
    return ImageFont.truetype(str(path), size)


def usable_fonts(paths):
    need = "0123456789+=()x%от"
    return [p for p in paths if p.exists() and all(has_glyph(p, c) for c in need)]


FALLBACK = SUP / "Arial Unicode.ttf"  # для символов, которых нет в шрифте (√, ≥, ...)
OPS = "+-−–=·×:÷<>≥≤"


def matching(s: str, i: int) -> int:
    """Индекс закрывающей скобки для s[i] == '('."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[j], 0)
        if depth == 0:
            return j
    return len(s) - 1


class LineDrawer:
    """Рисует задачу как в учебнике: индексы, черта корня, варианты знаков."""

    def __init__(self, draw, fpath, size, fill, spaced):
        self.d, self.fpath, self.size, self.fill, self.spaced = draw, fpath, size, fill, spaced
        # стиль выбирается один раз на строку, как в настоящем тексте
        self.caret_literal = random.random() < 0.1
        self.sub_base = random.random() < 0.7
        self.sup_group = random.random() < 0.8
        self.root_bar = random.random() < 0.7
        self.ge_symbol = random.random() < 0.7

    def glyph(self, ch, x, y, size):
        path = self.fpath if has_glyph(self.fpath, ch) else FALLBACK
        f = font(path, size)
        self.d.text((x, y), ch, font=f, fill=self.fill)
        return x + f.getlength(ch)

    def text(self, s, x, y, level=0):
        """level: 0 — строка, 1 — верхний индекс, -1 — нижний индекс."""
        size = self.size if level == 0 else int(self.size * 0.62)
        dy = {0: 0, 1: -self.size * 0.18, -1: self.size * 0.42}[level]
        i = 0
        while i < len(s):
            c = s[i]
            if s.startswith("log", i):
                x = self._plain("log", x, y + dy, size)
                j = i + 3
                while j < len(s) and s[j].isdigit():
                    j += 1
                base = s[i + 3:j]
                x = self.text(base, x + size * 0.02, y, -1) if self.sub_base else self._plain(base, x, y + dy, size)
                i = j
                continue
            if c == "^" and not self.caret_literal and level == 0 and i + 1 < len(s):
                if s[i + 1] == "(" and self.sup_group:
                    j = matching(s, i + 1)
                    x = self.text(s[i + 2:j], x + size * 0.03, y, 1)
                    i = j + 1
                elif s[i + 1] != "(":
                    x = self.text(s[i + 1], x + size * 0.03, y, 1)
                    i += 2
                else:
                    x = self._plain("^", x, y + dy, size)
                    i += 1
                continue
            if c == "√" and i + 1 < len(s) and s[i + 1] == "(" and self.root_bar:
                j = matching(s, i + 1)
                x = self.glyph("√", x, y + dy, size)
                x0 = x
                x = self.text(s[i + 2:j], x + size * 0.05, y, level) + size * 0.05
                top = y + dy + size * random.uniform(0.02, 0.12)
                self.d.line([(x0 - size * 0.05, top), (x, top)], fill=self.fill, width=max(1, size // 18))
                i = j + 1
                continue
            if s.startswith(">=", i) or s.startswith("<=", i):
                sym = s[i:i + 2]
                if self.ge_symbol:
                    sym = "≥" if sym == ">=" else "≤"
                x = self._op(sym, x, y + dy, size, level)
                i += 2
                continue
            if c in VARIANTS:
                opts = [(v, w) for v, w in zip(VARIANTS[c], VARIANT_W[c]) if has_glyph(self.fpath, v)]
                c = random.choices([v for v, _ in opts], [w for _, w in opts])[0]
            x = self._op(c, x, y + dy, size, level) if c in OPS else self._plain(c, x, y + dy, size)
            if c == ",":
                x += size * random.uniform(0.2, 0.8)
            i += 1
        return x

    def _plain(self, s, x, y, size):
        for ch in s:
            x = self.glyph(ch, x, y, size)
        return x

    def _op(self, s, x, y, size, level):
        pad = size * 0.25 if self.spaced and level == 0 else 0
        return self._plain(s, x + pad, y, size) + pad


def draw_line(draw, xy, question, fpath, size, fill, spaced):
    return LineDrawer(draw, fpath, size, fill, spaced).text(question, xy[0], xy[1])


def render(question: str, fonts) -> Image.Image:
    fpath = random.choice(fonts)
    size = random.randint(28, 60)
    spaced = random.random() < 0.5
    W, H = int(size * (len(question) + 4) * 1.1), int(size * 5)
    bg, ink = random.randint(150, 255), random.randint(0, 90)
    if bg - ink < 90:
        ink = bg - 90

    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img)
    if random.random() < 0.2:  # тетрадная клетка
        step = random.randint(int(size * 0.5), int(size * 1.0))
        c = bg - random.randint(20, 60)
        off = random.randint(0, step)
        for gx in range(off, W, step):
            d.line([(gx, 0), (gx, H)], fill=c, width=1)
        for gy in range(off, H, step):
            d.line([(0, gy), (W, gy)], fill=c, width=1)

    x0, y0 = size, int(size * 2)
    x1 = draw_line(d, (x0, y0), question, fpath, size, ink, spaced)
    bbox = (x0, y0 - int(size * 0.25), int(x1), y0 + int(size * 1.15))

    if random.random() < 0.25:  # соседние строки текста, чуть заходящие в кадр
        gap = size * random.uniform(1.25, 1.6)
        for dy in (-gap, gap):
            if random.random() < 0.6:
                draw_line(d, (x0 + random.randint(-size, size), y0 + dy), sample()[0], fpath, size, ink, spaced)

    m = lambda: int(size * random.uniform(0.0, 0.35))
    img = img.crop((max(bbox[0] - m(), 0), max(bbox[1] - m(), 0), min(bbox[2] + m(), W), min(bbox[3] + m(), H)))

    if random.random() < 0.8:
        img = img.rotate(random.uniform(-3, 3), resample=Image.BILINEAR, expand=True, fillcolor=bg)
    if random.random() < 0.3:
        sh = random.uniform(-0.15, 0.15)
        img = img.transform(img.size, Image.AFFINE, (1, sh, -sh * img.size[1] / 2, 0, 1, 0),
                            resample=Image.BILINEAR, fillcolor=bg)
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.045 * size)))

    # «камера»: итоговая высота символа от 12 до 40 пикселей
    scale = random.uniform(12, 40) / size
    img = img.resize((max(8, int(img.size[0] * scale)), max(8, int(img.size[1] * scale))), Image.BILINEAR)

    a = np.asarray(img, np.float32)
    h, w = a.shape
    if random.random() < 0.6:  # неравномерное освещение
        gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        a = a * (1 - random.uniform(0, 0.35) * (random.random() * gx + random.random() * gy) / 2)
    if random.random() < 0.4:  # текстура бумаги
        tex = np.asarray(Image.fromarray(np.random.randint(0, 255, (4, 4), np.uint8)).resize((w, h), Image.BICUBIC), np.float32)
        a = a + (tex - 128) / 128 * random.uniform(0, 15)
    a = a + np.random.randn(h, w) * random.uniform(0, 12)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(25, 95))
    return Image.open(io.BytesIO(buf.getvalue()))


def ocr_label(question: str) -> str:
    """Что OCR должен прочитать: формулу без словесного условия ("...;один корень")."""
    return question.split(";")[0].replace(" ", "")


def ocr_sample(fonts):
    q = ocr_label(sample()[0])
    if random.random() < 0.3 and "%" in q:
        q = q.replace("%от", "% от ")
    return render(q, fonts), q.replace(" ", "")
