"""Генератор обучающих задач с пошаговыми решениями и символьный токенизатор.

Каждый пример — строка вида  "<задача>?<решение>."  Модель учится продолжать
текст после "?" и останавливается на ".".
"""
import math
import random

PAD = "_"
SEP = "?"
EOS = "."
# Новые символы только дописываются в конец, чтобы старые веса оставались совместимы.
CHARS = PAD + SEP + EOS + "0123456789+-*/^()=x';% отD" + "|√<>,yalog" + "динкреьйвая"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
BLOCK_SIZE = 160


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


def signed(n: int) -> str:
    """Число со знаком для продолжения выражения: 5 -> +5, -5 -> -5, 0 -> ''."""
    return f"+{n}" if n > 0 else (str(n) if n < 0 else "")


def poly(coefs: list[tuple[int, str]]) -> str:
    out = ""
    for c, v in coefs:
        out += term(c, v, first=(out == ""))
    return out or "0"


# ---------------------------------------------------------------- задачи

def gen_addsub():
    if random.random() < 0.45:  # со знаками, как в промежуточных шагах: -190-(-114)
        a, b = random.randint(-999, 999), random.randint(-999, 999)
        op = random.choice("+-")
        return f"{a}{op}{paren(b)}", f"{a + b if op == '+' else a - b}"
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


def lin_steps(a: int, b: int, c: int) -> list[str]:
    """Шаги решения a*x+b=c (корень целый)."""
    x = (c - b) // a
    steps = []
    if b != 0:
        steps.append(f"{term(a, 'x', True)}={c - b}")
    if a != 1 or not steps:
        steps.append(f"x={x}")
    return steps


def quad_steps(p: int, q: int) -> tuple[list[str], list[int]]:
    """Шаги решения x^2+px+q=0 через дискриминант (корни целые)."""
    d = p * p - 4 * q
    four_q = 4 * q
    d_expr = f"{p * p}-{four_q}" if four_q >= 0 else f"{p * p}+{-four_q}"
    if d == 0:
        return [f"D={d_expr}=0", f"x={-p}/2={-p // 2}"], [-p // 2]
    s = math.isqrt(d)
    hi, lo = (-p + s) // 2, (-p - s) // 2
    return [f"D={d_expr}={d}", f"x1=({-p}+{s})/2={hi}", f"x2=({-p}-{s})/2={lo}"], [hi, lo]


def gen_quadratic():
    r1, r2 = random.randint(-9, 9), random.randint(-9, 9)
    p, q = -(r1 + r2), r1 * r2
    eq = poly([(1, "x^2"), (p, "x"), (q, "")]) + "=0"
    return eq, ";".join(quad_steps(p, q)[0])


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


# ---------------------------------------------------------------- старшие классы

NO_ROOTS = "нет корней"


def gen_quadratic_a():
    """ax^2+bx+c=0 с a≠1."""
    a = random.choice([2, 3, 4, 5, -1, -2, -3])
    r1, r2 = random.randint(-6, 6), random.randint(-6, 6)
    b, c = -a * (r1 + r2), a * r1 * r2
    eq = poly([(a, "x^2"), (b, "x"), (c, "")]) + "=0"
    d, fac = b * b - 4 * a * c, 4 * a * c
    d_expr = f"{b * b}-{fac}" if fac >= 0 else f"{b * b}+{-fac}"
    if d == 0:
        return eq, f"D={d_expr}=0;x={-b}/{paren(2 * a)}={r1}"
    s = math.isqrt(d)
    return eq, (f"D={d_expr}={d};x1=({-b}+{s})/{paren(2 * a)}={(-b + s) // (2 * a)};"
                f"x2=({-b}-{s})/{paren(2 * a)}={(-b - s) // (2 * a)}")


def gen_abs():
    """|ax+b|=c."""
    a = random.choice([1, 2, 3, -1, -2, -3])
    x1 = random.randint(-9, 9)
    x2 = x1 + 2 * random.randint(-6, 6)
    b, c = -a * (x1 + x2) // 2, abs(a * (x1 - x2) // 2)
    lhs = poly([(a, "x"), (b, "")])
    if random.random() < 0.12:
        c = -random.randint(1, 9)
        return f"|{lhs}|={c}", f"{c}<0;{NO_ROOTS}"
    if c == 0:
        return f"|{lhs}|=0", ";".join([f"{lhs}=0"] + lin_steps(a, b, 0))
    return f"|{lhs}|={c}", ";".join([f"{lhs}={c}"] + lin_steps(a, b, c) + [f"{lhs}={-c}"] + lin_steps(a, b, -c))


LOG_MAX_N = {2: 10, 3: 6, 5: 4, 10: 4}


def gen_log():
    """Значения логарифмов и свойства log(a)+log(b), log(a)-log(b)."""
    kind = random.random()
    if kind < 0.4:
        b = random.choice(list(LOG_MAX_N))
        n = random.randint(0, LOG_MAX_N[b])
        return f"log{b}({b ** n})", f"{b ** n}={b}^{n};{n}"
    b = random.choice([2, 3, 6, 10, 12, 15])
    n = random.randint(2, 3 if b in (2, 3, 6, 10) else 2)
    N = b ** n
    if kind < 0.7:
        divs = [d for d in range(2, N) if N % d == 0]
        A = random.choice(divs)
        return f"log{b}({A})+log{b}({N // A})", f"log{b}({A}*{N // A})=log{b}({N})={n}"
    B = random.randint(2, 9)
    return f"log{b}({N * B})-log{b}({B})", f"log{b}({N * B}/{B})=log{b}({N})={n}"


def gen_log_eq():
    """log_b(kx+m)=n."""
    b = random.choice([2, 3, 5])
    n = random.randint(0, {2: 5, 3: 3, 5: 2}[b])
    k = random.choice([1, 2, 3, -1, -2])
    x = random.randint(-10, 10)
    m = b ** n - k * x
    inner = poly([(k, "x"), (m, "")])
    return f"log{b}({inner})={n}", ";".join([f"{inner}={b}^{n}", f"{inner}={b ** n}"] + lin_steps(k, m, b ** n))


def gen_sqrt():
    kind = random.random()
    if kind < 0.2:
        n = random.randint(1, 30)
        return f"√{n * n}", f"{n}"
    if kind < 0.4:
        n = random.randint(2, 30)
        divs = [d for d in range(2, n * n) if (n * n) % d == 0]
        if not divs:
            return f"√{n * n}", f"{n}"
        A = random.choice(divs)
        return f"√{A}*√{n * n // A}", f"√({A}*{n * n // A})=√{n * n}={n}"
    if kind < 0.7:  # √(kx+m)=c
        k = random.choice([1, 2, 3, -1, -2])
        if random.random() < 0.1:
            c = -random.randint(1, 9)
            return f"√({poly([(k, 'x'), (random.randint(-9, 9), '')])})={c}", f"{c}<0;{NO_ROOTS}"
        c, x = random.randint(0, 9), random.randint(-10, 10)
        m = c * c - k * x
        inner = poly([(k, "x"), (m, "")])
        return f"√({inner})={c}", ";".join([f"{inner}={c * c}"] + lin_steps(k, m, c * c))
    # √(x+a)=x+b: возводим в квадрат, решаем квадратное, отбрасываем посторонние корни
    while True:
        r1, r2 = random.randint(-6, 8), random.randint(-6, 8)
        if (r1 + r2) % 2 == 1:
            break
    b = (1 - r1 - r2) // 2
    a = b * b - r1 * r2
    p, q = 2 * b - 1, b * b - a
    lhs, rhs = poly([(1, "x"), (a, "")]), poly([(1, "x"), (b, "")])
    steps = [f"{lhs}={poly([(1, 'x^2'), (2 * b, 'x'), (b * b, '')])}", f"{poly([(1, 'x^2'), (p, 'x'), (q, '')])}=0"]
    qs, roots = quad_steps(p, q)
    steps += qs
    good = []
    for r in roots:
        v = r + b
        steps.append(f"{r}{signed(b)}={v}" + (">=0" if v >= 0 else "<0") if b else f"{r}" + (">=0" if v >= 0 else "<0"))
        if v >= 0:
            good.append(r)
    steps += [f"x={r}" for r in good] or [NO_ROOTS]
    return f"√({lhs})={rhs}", ";".join(steps)


def gen_exp():
    """b^(kx+m)=b^n."""
    b = random.choice([2, 3, 5])
    n = random.randint(0, {2: 8, 3: 5, 5: 3}[b])
    k = random.choice([1, 1, 2, -1])
    x = random.randint(-6, 6)
    m = n - k * x
    e = poly([(k, "x"), (m, "")])
    lhs = f"{b}^x" if e == "x" else f"{b}^({e})"
    if e == "x":
        return f"{lhs}={b ** n}", f"{b ** n}={b}^{n};x={n}"
    return f"{lhs}={b ** n}", ";".join([f"{b ** n}={b}^{n}", f"{e}={n}"] + lin_steps(k, m, n))


FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def gen_ineq():
    """ax+b<c (со сменой знака при делении на отрицательное)."""
    a = random.choice([i for i in range(-9, 10) if i != 0])
    x0, b = random.randint(-15, 15), random.randint(-20, 20)
    c = a * x0 + b
    op = random.choice(list(FLIP))
    lhs = poly([(a, "x"), (b, "")])
    steps = []
    if b != 0:
        steps.append(f"{term(a, 'x', True)}{op}{c - b}")
    if a != 1 or not steps:
        steps.append(f"x{FLIP[op] if a < 0 else op}{x0}")
    return f"{lhs}{op}{c}", ";".join(steps)


def gen_system():
    """Система двух линейных уравнений, метод Крамера."""
    while True:
        a1, b1, a2, b2 = (random.choice([i for i in range(-5, 6) if i != 0]) for _ in range(4))
        D = a1 * b2 - a2 * b1
        if D != 0:
            break
    x, y = random.randint(-9, 9), random.randint(-9, 9)
    c1, c2 = a1 * x + b1 * y, a2 * x + b2 * y
    q = f"{poly([(a1, 'x'), (b1, 'y')])}={c1},{poly([(a2, 'x'), (b2, 'y')])}={c2}"
    Dx, Dy = c1 * b2 - c2 * b1, a1 * c2 - a2 * c1
    def det(name, p, q, r, t):  # p*q-r*t с промежуточными произведениями
        return f"{name}={p}*{paren(q)}-{paren(r)}*{paren(t)}={p * q}-{paren(r * t)}={p * q - r * t}"

    sol = [det("D", a1, b2, a2, b1), det("Dx", c1, b2, c2, b1), det("Dy", a1, c2, a2, c1),
           f"x={Dx}/{paren(D)}={x}", f"y={Dy}/{paren(D)}={y}"]
    return q, ";".join(sol)


ONE, NONE, TWO = "один корень", "нет корней", "два корня"


def gen_param():
    """Типовые задачи с параметром a."""
    kind = random.randint(0, 6)
    if kind in (0, 1, 2):  # x^2+ax+k^2=0
        k = random.randint(1, 9)
        k2 = 4 * k * k
        eq = f"x^2+ax+{k * k}=0"
        if kind == 0:
            return f"{eq};{ONE}", f"D=a^2-{k2}=0;a^2={k2};a={2 * k};a={-2 * k}"
        if kind == 1:
            return f"{eq};{TWO}", f"D=a^2-{k2}>0;a^2>{k2};a<{-2 * k};a>{2 * k}"
        return f"{eq};{NONE}", f"D=a^2-{k2}<0;a^2<{k2};{-2 * k}<a<{2 * k}"
    if kind in (3, 4, 5):  # x^2+px+a=0
        p = 2 * random.choice([i for i in range(-7, 8) if i != 0])
        p2 = p * p
        eq = poly([(1, "x^2"), (p, "x")]) + "+a=0"
        if kind == 3:
            return f"{eq};{ONE}", f"D={p2}-4a=0;4a={p2};a={p2 // 4}"
        if kind == 4:
            return f"{eq};{NONE}", f"D={p2}-4a<0;4a>{p2};a>{p2 // 4}"
        r = random.choice([i for i in range(-6, 7) if i != 0])
        v = r * r + p * r
        return (f"{eq};x={r}",
                f"{paren(r)}^2{signed(p)}*{paren(r)}+a=0;{r * r}{signed(p * r)}+a=0;a={-v}")
    # ax+b=c имеет корень x=r
    a = random.choice([i for i in range(-9, 10) if i != 0])
    r = random.choice([i for i in range(-9, 10) if i != 0])
    b = random.randint(-20, 20)
    c = a * r + b
    steps = [f"{term(r, 'a', True)}{signed(b)}={c}"]
    if b != 0:
        steps.append(f"{term(r, 'a', True)}={c - b}")
    if r != 1:
        steps.append(f"a={a}")
    return f"ax{signed(b)}={c};x={r}", ";".join(steps)


TASKS = {
    "addsub": (gen_addsub, 0.10),
    "mul": (gen_mul, 0.07),
    "div": (gen_div, 0.02),
    "expr": (gen_expr, 0.04),
    "linear": (gen_linear, 0.06),
    "quadratic": (gen_quadratic, 0.06),
    "derivative": (gen_derivative, 0.03),
    "percent": (gen_percent, 0.02),
    "quadratic_a": (gen_quadratic_a, 0.07),
    "abs": (gen_abs, 0.07),
    "log": (gen_log, 0.07),
    "log_eq": (gen_log_eq, 0.06),
    "sqrt": (gen_sqrt, 0.10),
    "exp": (gen_exp, 0.05),
    "ineq": (gen_ineq, 0.06),
    "system": (gen_system, 0.07),
    "param": (gen_param, 0.10),
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
