import os
import re
from pathlib import Path

# Модель маленькая: многопоточный BLAS на слабом CPU только тормозит.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from mathnet.model import MathNet
from mathnet.ocr import OCRNet
from mathnet.pipeline import SolverPool, solve_problem, step_tex

WEIGHTS = Path(__file__).parent / "weights"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
net = MathNet(WEIGHTS / "mathnet.npz")
ocr = OCRNet(WEIGHTS / "ocr.npz")
pool = SolverPool()

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
    (r"(?:хотя бы один корень|имеет (?:корни|решения)|есть корни)", "есть корни"),
]
NUMBERS = {"одно": 1, "один": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6}


def normalize(text: str) -> str:
    """Приводит ввод пользователя к формату, на котором обучена сеть."""
    s = text.strip().lower()
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    s = re.sub(r"(?<=[\d)])\s*:\s*(?=[\d(])", "/", s)  # 12:4 -> 12/4
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)                   # 0,5 -> 0.5
    s = re.sub(r"\blg\s*([\dx]+)", r"log10(\1)", s)              # lg x -> log10(x)
    s = re.sub(r"\blog(\d+)\s+([\dx]+)", r"log\1(\2)", s)       # log2 x -> log2(x)
    s = re.sub(r"\bln\s*([\dx]+)", r"ln(\1)", s)
    s = s.replace("tg", "tan").replace("ctan", "cot")
    s = re.sub(r"\b(sin|cos|tan|cot)\s*([\dx]+)", r"\1(\2)", s)  # sin x -> sin(x)
    if m := re.search(r"(\d+)\s*%\s*(?:от|из)?\s*(\d+)", s):
        return f"{m[1]}% от {m[2]}"
    is_derivative = bool(re.search(r"производн|d/dx", s)) or re.fullmatch(r"\s*\(.*\)'\s*", s)

    suffix = ""
    if m := re.search(r"ровно (\d+|" + "|".join(NUMBERS) + r")\s+\w*(?:решени|корн)\w*", s):
        n = m[1] if m[1].isdigit() else NUMBERS[m[1]]
        s, suffix = s[:m.start()] + " " + s[m.end():], f";ровно {n}"
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
    if len(problem) > 200:
        return jsonify(error="Слишком длинная задача."), 400
    body, _, cond = problem.partition(";")
    result = solve_problem(problem, net, pool)
    result["problem_tex"] = step_tex(body) + (rf"\quad\text{{({cond})}}" if cond else "")
    return jsonify(result), (200 if result["source"] != "none" else 422)


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
