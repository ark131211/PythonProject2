import os
import re
from pathlib import Path

# Модель маленькая: многопоточный BLAS на слабом CPU только тормозит.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from flask import Flask, jsonify, render_template, request

from mathnet.data import BLOCK_SIZE, STOI
from mathnet.model import MathNet

WEIGHTS = Path(__file__).parent / "weights" / "mathnet.npz"

app = Flask(__name__)
net = MathNet(WEIGHTS)

REPLACEMENTS = {
    "×": "*", "·": "*", "∙": "*", "÷": "/", ":": "/", "−": "-", "–": "-", "—": "-",
    "²": "^2", "³": "^3", "х": "x", "**": "^",
}
WORDS = re.compile(r"(решите|решить|реши|уравнение|вычислите|вычисли|посчитайте|посчитай|"
                   r"найдите|найди|сколько будет|чему равно|производную|производная|"
                   r"функции|d/dx)", re.I)


def normalize(text: str) -> str:
    """Приводит ввод пользователя к формату, на котором обучена сеть."""
    s = text.strip().lower()
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    if m := re.search(r"(\d+)\s*%\s*(?:от|из)?\s*(\d+)", s):
        return f"{m[1]}% от {m[2]}"
    is_derivative = bool(re.search(r"производн|d/dx", s)) or re.fullmatch(r"\s*\(.*\)'\s*", s)
    s = WORDS.sub("", s)
    s = re.sub(r"\s+", "", s).rstrip("?").removesuffix("=")
    if is_derivative and not s.endswith("'"):
        s = f"({s.strip('()')})'"
    return s


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
