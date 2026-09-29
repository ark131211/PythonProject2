import os
import re
from pathlib import Path

# Модель маленькая: многопоточный BLAS на слабом CPU только тормозит.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from mathnet.data import BLOCK_SIZE, STOI
from mathnet.model import MathNet
from mathnet.ocr import OCRNet

WEIGHTS = Path(__file__).parent / "weights"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
net = MathNet(WEIGHTS / "mathnet.npz")
ocr = OCRNet(WEIGHTS / "ocr.npz")

REPLACEMENTS = {
    "×": "*", "·": "*", "∙": "*", "÷": "/", "−": "-", "–": "-", "—": "-",
    "²": "^2", "³": "^3", "**": "^", "≥": ">=", "≤": "<=", "{": "(", "}": ")",
    "sqrt": "√", "log_": "log", "lg(": "log10(", "√ ": "√",
    **{chr(0x2080 + i): str(i) for i in range(10)},  # подстрочные цифры: log₂ -> log2
}
CONDITIONS = [  # задачи с параметром: фраза пользователя -> условие в формате сети
    (r"один корень|единственн\w* (?:корень|решение)|одно решение", "один корень"),
    (r"нет корней|не имеет (?:корней|решений)|нет решений", "нет корней"),
    (r"два (?:различных )?(?:корня|решения)", "два корня"),
]


def normalize(text: str) -> str:
    """Приводит ввод пользователя к формату, на котором обучена сеть."""
    s = text.strip().lower()
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    s = re.sub(r"(?<=[\d)])\s*:\s*(?=[\d(])", "/", s)  # 12:4 -> 12/4
    if m := re.search(r"(\d+)\s*%\s*(?:от|из)?\s*(\d+)", s):
        return f"{m[1]}% от {m[2]}"
    is_derivative = bool(re.search(r"производн|d/dx", s)) or re.fullmatch(r"\s*\(.*\)'\s*", s)

    suffix = ""
    for pattern, cond in CONDITIONS:
        if re.search(pattern, s):
            s, suffix = re.sub(pattern, " ", s), ";" + cond
    root = r"кор(?:ень|н\w*)"
    if m := re.search(rf"{root}\s*[xх]\s*=\s*(-?\d+)|[xх]\s*=\s*(-?\d+)\s*(?:является|-)?\s*{root}", s):
        s, suffix = s[:m.start()] + " " + s[m.end():], f";x={m[1] or m[2]}"

    s = re.sub(r"\s+и\s+|[;\n]", ",", s)             # системы: "... и ...", построчно
    s = re.sub(r"[а-яё]{2,}|d/dx", " ", s)             # убираем русские слова
    s = re.sub(r"(?:(?<=[\s,])|^)[aа](?=[\s,]|$)", " ", s)  # "при каком a уравнение ..."
    s = s.replace("а", "a").replace("х", "x")          # кириллические a и x в формуле
    if s.count("=") == 2 and "," not in s:             # "x+y=5 x-y=1"
        s = re.sub(r"(=\s*-?\d+)\s+(?=[-\dxy(])", r"\1,", s)
    s = re.sub(r"\s+", "", s).strip(",.:").rstrip("?").removesuffix("=")
    if is_derivative and not s.endswith("'"):
        s = f"({s.strip('()')})'"
    return s + suffix


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/api/solve")
def solve():
    raw = (request.get_json(silent=True) or {}).get("problem", "")
    problem = normalize(raw)
    if not problem:
        return jsonify(error="Введите задачу."), 400
    bad = sorted({c for c in problem if c not in STOI or c in "?._"})
    if bad:
        return jsonify(error=f"Сеть не знает символы: {' '.join(bad)}"), 400
    if len(problem) > BLOCK_SIZE // 2:
        return jsonify(error="Слишком длинная задача."), 400

    solution, confidence = net.solve(problem)
    return jsonify(problem=problem, steps=solution.split(";"), confidence=confidence)


@app.post("/api/ocr")
def read_photo():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Прикрепите фото."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(file.stream))
    except UnidentifiedImageError:
        return jsonify(error="Не получилось открыть картинку."), 400
    text, confidence = ocr.recognize(img)
    if not text:
        return jsonify(error="Не нашёл на фото текста задачи."), 422
    return jsonify(text=text, confidence=confidence)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
