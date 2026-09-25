"""Генератор обучающих задач с пошаговыми решениями и символьный токенизатор.

Каждый пример — строка вида  "<задача>?<решение>."  Модель учится продолжать
текст после "?" и останавливается на ".".
"""
import math
import random

PAD = "_"
SEP = "?"
EOS = "."
CHARS = PAD + SEP + EOS + "0123456789+-*/^()=x';% отD"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
BLOCK_SIZE = 64


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def paren(n: int) -> str:
    return f"({n})" if n < 0 else str(n)


def term(coef: int, var: str, first: bool) -> str:
    """Форматирует слагаемое coef*var: 1x -> x, -1x -> -x, знак + между членами."""
    if coef == 0:
        return ""
    body = var if abs(coef) == 1 and var else f"{abs(coef)}{var}"
    if coef < 0:
        return "-" + body
    return body if first else "+" + body


def poly(coefs: list[tuple[int, str]]) -> str:
    out = ""
    for c, v in coefs:
        out += term(c, v, first=(out == ""))
    return out or "0"


# ---------------------------------------------------------------- задачи

def gen_addsub():
    a, b = random.randint(0, 999), random.randint(0, 999)
    if random.random() < 0.5:
        return f"{a}+{b}", f"{a + b}"
    return f"{a}-{b}", f"{a - b}"


def gen_mul():
    a = random.randint(2, 99)
    b = random.randint(2, 99)
    if b < 10:
        return f"{a}*{b}", f"{a * b}"
    t, u = b // 10 * 10, b % 10
    if u == 0:
        return f"{a}*{b}", f"{a * b}"
    return f"{a}*{b}", f"{a}*{t}+{a}*{u}={a * t}+{a * u}={a * b}"


def gen_div():
    b = random.randint(2, 12)
    q = random.randint(1, 12)
    return f"{b * q}/{b}", f"{q}"


def gen_expr():
    a, b, c = (random.randint(1, 20) for _ in range(3))
    kind = random.randint(0, 3)
    if kind == 0:
        return f"{a}+{b}*{c}", f"{a}+{b * c}={a + b * c}"
    if kind == 1:
        return f"{a}-{b}*{c}", f"{a}-{b * c}={a - b * c}"
    if kind == 2:
        return f"({a}+{b})*{c}", f"{a + b}*{c}={(a + b) * c}"
    return f"({a}-{b})*{c}", f"{a - b}*{c}={(a - b) * c}"


def gen_linear():
    a = random.choice([i for i in range(-9, 10) if i != 0])
    x = random.randint(-20, 20)
    b = random.randint(-30, 30)
    c = a * x + b
    lhs = poly([(a, "x"), (b, "")])
    q = f"{lhs}={c}"
    steps = []
    if b != 0:
        moved = f"{c}-{b}" if b > 0 else f"{c}+{-b}"
        steps.append(f"{term(a, 'x', True)}={moved}")
        steps.append(f"{term(a, 'x', True)}={c - b}")
    if a != 1:
        steps.append(f"x={c - b}/{paren(a)}")
    steps.append(f"x={x}")
    return q, ";".join(steps)


def gen_quadratic():
    r1, r2 = random.randint(-9, 9), random.randint(-9, 9)
    p, q = -(r1 + r2), r1 * r2
    eq = poly([(1, "x^2"), (p, "x"), (q, "")]) + "=0"
    d = p * p - 4 * q
    four_q = 4 * q
    d_expr = f"{p * p}-{four_q}" if four_q >= 0 else f"{p * p}+{-four_q}"
    s = int(round(d ** 0.5))
    hi, lo = max(r1, r2), min(r1, r2)
    if d == 0:
        sol = f"D={d_expr}=0;x={-p}/2={r1}"
    else:
        sol = (f"D={d_expr}={d};"
               f"x1=({-p}+{s})/2={hi};x2=({-p}-{s})/2={lo}")
    return eq, sol


def gen_derivative():
    deg = random.randint(1, 3)
    coefs = [random.randint(-9, 9) for _ in range(deg + 1)]  # от x^deg до x^0
    if coefs[0] == 0:
        coefs[0] = random.choice([1, 2, 3, -1, -2])
    names = {3: "x^3", 2: "x^2", 1: "x", 0: ""}
    f = poly([(c, names[deg - i]) for i, c in enumerate(coefs)])
    d = poly([(c * (deg - i), names[deg - i - 1])
              for i, c in enumerate(coefs) if deg - i >= 1])
    return f"({f})'", d


def gen_percent():
    p = random.choice([1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100])
    step = 100 // math.gcd(p, 100)  # n кратно step, чтобы ответ был целым
    n = step * random.randint(1, 1000 // step)
    return f"{p}% от {n}", f"{n}*{p}/100={n * p // 100}"


TASKS = {
    "addsub": (gen_addsub, 0.22),
    "mul": (gen_mul, 0.20),
    "div": (gen_div, 0.04),
    "expr": (gen_expr, 0.10),
    "linear": (gen_linear, 0.16),
    "quadratic": (gen_quadratic, 0.16),
    "derivative": (gen_derivative, 0.07),
    "percent": (gen_percent, 0.05),
}


def sample(task: str | None = None) -> tuple[str, str]:
    if task is None:
        names = list(TASKS)
        task = random.choices(names, weights=[TASKS[n][1] for n in names])[0]
    return TASKS[task][0]()


if __name__ == "__main__":
    for name in TASKS:
        for _ in range(3):
            q, a = sample(name)
            s = q + SEP + a + EOS
            assert all(c in STOI for c in s) and len(s) <= BLOCK_SIZE, s
            print(f"{name:11s} {q}  ->  {a}")
