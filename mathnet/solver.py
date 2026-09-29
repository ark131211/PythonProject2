"""Символьный решатель (SymPy) — точные решения с шагами для задач, где MathNet
не справился, и эталон для проверки ответов MathNet.

Вход — задача в нормализованном формате сайта (как для MathNet), например
"x^2-5x+6=0", "log2(x)+log2(x-2)=3", "x+y=5,x-y=1", "|x^2-4x|=a;ровно 3".
"""
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                        rationalize, standard_transformations)

x, y, a = sp.symbols("x y a", real=True)
LOCALS = {"x": x, "y": y, "a": a, "sqrt": sp.sqrt, "log": sp.log, "Abs": sp.Abs, "pi": sp.pi, "e": sp.E,
          "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "ln": sp.log}
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor, rationalize)
COUNT_WORDS = {"один корень": 1, "нет корней": 0, "два корня": 2}


class SolveError(Exception):
    pass


@dataclass
class Step:
    text: str
    tex: str = ""


@dataclass
class Solution:
    steps: list[Step] = field(default_factory=list)
    answer_text: str = ""
    answer_tex: str = ""
    kind: str = ""            # expr | equation | inequality | system | param
    var: str = "x"
    answer: object = None     # эталон для сравнения с ответом MathNet (число, выражение или множество)

    def add(self, text, tex=""):
        self.steps.append(Step(text, tex))


# ---------------------------------------------------------------- разбор

def _matching(s, i):
    depth = 0
    for j in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[j], 0)
        if depth == 0:
            return j
    raise SolveError("Непарные скобки")


def to_sympy_syntax(s: str) -> str:
    """Синтаксис сайта -> синтаксис SymPy: log2(x) -> log(x,2), √ -> sqrt, |u| -> Abs(u)."""
    s = s.replace("tg(", "tan(").replace("ctg(", "cot(")
    out, i = [], 0
    while i < len(s):
        m = re.match(r"log(\d+)\(", s[i:])
        if m:
            j = _matching(s, i + len(m.group(0)) - 1)
            inner = to_sympy_syntax(s[i + len(m.group(0)): j])
            out.append(f"log({inner},{m.group(1)})")
            i = j + 1
            continue
        if s[i] == "√":
            if i + 1 < len(s) and s[i + 1] == "(":
                j = _matching(s, i + 1)
                out.append(f"sqrt({to_sympy_syntax(s[i + 2:j])})")
                i = j + 1
            else:
                m = re.match(r"[\d.]+|[a-z]", s[i + 1:])
                if not m:
                    raise SolveError("Не понял выражение под корнем")
                out.append(f"sqrt({m.group(0)})")
                i += 1 + len(m.group(0))
            continue
        out.append(s[i])
        i += 1
    s = "".join(out)
    # модуль: '|' открывает, если перед ним начало/оператор/скобка
    res, opened = [], 0
    for k, c in enumerate(s):
        if c == "|":
            prev = s[:k].rstrip()[-1:] if s[:k].strip() else ""
            if prev in ("", "(", "+", "-", "*", "/", "^", "=", ",", "<", ">") or (prev == "|" and opened):
                res.append("Abs(")
                opened += 1
            else:
                res.append(")")
                opened -= 1
        else:
            res.append(c)
    return "".join(res)


def expr(s: str):
    try:
        return parse_expr(to_sympy_syntax(s), local_dict=LOCALS, transformations=TRANSFORMS, evaluate=True)
    except SolveError:
        raise
    except Exception as e:
        raise SolveError(f"Не смог разобрать «{s}»") from e


REL = [">=", "<=", ">", "<", "="]


def split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for c in s:
        depth += {"(": 1, ")": -1}.get(c, 0)
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    return parts + [cur]


def relation(s: str):
    """'lhs op rhs' -> (lhs, op, rhs) или (expr, None, None)."""
    for op in REL:
        depth = 0
        for i in range(len(s)):
            depth += {"(": 1, ")": -1}.get(s[i], 0)
            if depth == 0 and s.startswith(op, i):
                if op in "<>" and s[i + 1: i + 2] == "=":
                    continue
                if op == "=" and i > 0 and s[i - 1] in "<>":
                    continue
                return expr(s[:i]), op, expr(s[i + len(op):])
    return expr(s), None, None


# ---------------------------------------------------------------- форматирование

def txt(e) -> str:
    s = sp.sstr(e).replace("**", "^").replace("sqrt", "√").replace("oo", "∞").replace("pi", "π")
    s = re.sub(r"Abs\(([^()]*(?:\([^()]*\))*[^()]*)\)", r"|\1|", s)
    s = re.sub(r"(\d)\*([a-zπ(√])", r"\1\2", s)
    s = re.sub(r"π\*([a-z])", r"π\1", s)
    return s.replace("*", "·")


def tex(e) -> str:
    return sp.latex(e)


def set_text(S, var="x") -> tuple[str, str]:
    """Множество решений -> ('x = 1; x = 2', TeX) по-русски."""
    try:
        return _set_text(S, var)
    except Exception:
        return f"{var} ∈ {txt(S)}", rf"{var} \in {tex(S)}"


def _set_text(S, var):
    if isinstance(S, (sp.ImageSet, sp.Union)) and all(isinstance(p, sp.ImageSet) for p in (S.args if isinstance(S, sp.Union) else [S])):
        n = sp.Symbol("n", integer=True)
        forms = [p.lamda.expr.subs(p.lamda.variables[0], n) for p in (S.args if isinstance(S, sp.Union) else [S])]
        return ("; ".join(f"{var} = {txt(f)}" for f in forms) + ", n ∈ ℤ",
                r";\ ".join(f"{var} = {tex(f)}" for f in forms) + r",\ n \in \mathbb{Z}")
    if S == sp.S.EmptySet:
        return "нет решений", r"\varnothing"
    if isinstance(S, sp.FiniteSet):
        vals = sorted(S, key=lambda v: float(v))
        return "; ".join(f"{var} = {txt(v)}" for v in vals), r";\ ".join(f"{var} = {tex(v)}" for v in vals)
    if S == sp.S.Reals:
        return f"{var} — любое число", rf"{var} \in \mathbb{{R}}"
    comp = sp.Complement(sp.S.Reals, S)
    if isinstance(comp, sp.FiniteSet):
        vals = sorted(comp, key=lambda v: float(v))
        return (f"{var} ≠ " + ", ".join(txt(v) for v in vals),
                rf"{var} \ne " + ",\\ ".join(tex(v) for v in vals))
    pieces = S.args if isinstance(S, sp.Union) else (S,)
    t, T = [], []
    for p in sorted(pieces, key=lambda p: float(p.inf) if p.inf.is_finite else -1e18):
        if isinstance(p, sp.FiniteSet):
            t += [f"{{{txt(v)}}}" for v in p]
            T += [f"\\{{{tex(v)}\\}}" for v in p]
            continue
        lo = "(-∞" if p.inf == -sp.oo else ("(" if p.left_open else "[") + txt(p.inf)
        hi = "+∞)" if p.sup == sp.oo else txt(p.sup) + (")" if p.right_open else "]")
        Lo = r"(-\infty" if p.inf == -sp.oo else ("(" if p.left_open else "[") + tex(p.inf)
        Hi = r"+\infty)" if p.sup == sp.oo else tex(p.sup) + (")" if p.right_open else "]")
        t.append(f"{lo}; {hi}")
        T.append(rf"{Lo};\ {Hi}")
    return f"{var} ∈ " + " ∪ ".join(t), rf"{var} \in " + r" \cup ".join(T)


# ---------------------------------------------------------------- решение

def solve(problem: str) -> Solution:
    body, _, cond = problem.partition(";")
    if "a" in body.replace("log", "").replace("tan", "").replace("abs", "") and cond:
        return solve_param(body, cond)
    if re.fullmatch(r"\(.*\)'", body):
        return solve_derivative(body[1:-2])
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)% от (\d+(?:\.\d+)?)", problem):
        p, n = sp.nsimplify(m[1]), sp.nsimplify(m[2])
        sol = Solution(kind="expr", answer=n * p / 100)
        sol.add(f"{m[1]}% от {m[2]} = {m[2]}·{m[1]}/100", rf"{m[1]}\%\ \text{{от}}\ {m[2]} = \frac{{{m[2]}\cdot {m[1]}}}{{100}}")
        sol.answer_text, sol.answer_tex = txt(sol.answer), tex(sol.answer)
        return sol
    parts = split_top(body, ",")
    if len(parts) == 2 and all("=" in p for p in parts):
        return solve_system(parts)
    lhs, op, rhs = relation(body)
    if op is None:
        return solve_expr(lhs)
    if op == "=":
        return solve_equation(lhs, rhs)
    return solve_inequality(lhs, op, rhs)


def solve_expr(e) -> Solution:
    sol = Solution(kind="expr")
    val = sp.nsimplify(sp.simplify(e))
    sol.answer = val
    if val.free_symbols:
        f = sp.factor(val)
        sol.add(f"Упростим: {txt(val)}", tex(val))
        if f != val:
            sol.add(f"Разложим на множители: {txt(f)}", tex(f))
    else:
        sol.add(f"{txt(e)} = {txt(val)}", f"{tex(e)} = {tex(val)}")
        if not val.is_Rational:
            sol.add(f"≈ {sp.N(val, 6)}", rf"\approx {sp.N(val, 6)}")
    sol.answer_text, sol.answer_tex = txt(val), tex(val)
    return sol


def solve_derivative(body: str) -> Solution:
    f = expr(body)
    d = sp.simplify(sp.diff(f, x))
    sol = Solution(kind="expr", answer=d)
    sol.add(f"f(x) = {txt(f)}", f"f(x) = {tex(f)}")
    sol.add(f"f'(x) = {txt(d)}", f"f'(x) = {tex(d)}")
    sol.answer_text, sol.answer_tex = txt(d), tex(d)
    return sol


def _domain(f):
    try:
        return continuous_domain(f, x, sp.S.Reals)
    except Exception:
        return sp.S.Reals


def solve_equation(lhs, rhs) -> Solution:
    sol = Solution(kind="equation")
    f = sp.simplify(lhs - rhs)
    dom = _domain(lhs) & _domain(rhs)
    if dom != sp.S.Reals:
        t, T = set_text(dom)
        sol.add(f"ОДЗ: {t}", rf"\text{{ОДЗ: }} {T}")
    poly = sp.expand(f) if (lhs - rhs).is_polynomial(x) else None
    if poly is not None and poly.is_polynomial(x) and sp.degree(poly, x) >= 1:
        _polynomial_steps(sol, sp.Poly(poly, x))
    elif f.has(sp.Abs):
        sol.add("Раскроем модули по определению и решим на каждом промежутке")
    elif f.has(sp.sqrt) or any(isinstance(p, sp.Pow) and p.exp.is_Rational and p.exp.q == 2 for p in f.atoms(sp.Pow)):
        sol.add("Уединим корень и возведём обе части в квадрат; лишние корни отсеем проверкой")
    elif f.has(sp.log):
        sol.add("Воспользуемся свойствами логарифмов и перейдём к уравнению без логарифмов")
    roots = sp.solveset(sp.Eq(lhs, rhs), x, sp.S.Reals)
    if isinstance(roots, sp.ConditionSet) or roots is None:
        roots = _candidates_with_check(sol, lhs, rhs, dom)
    elif f.has(sp.sqrt, sp.log) and isinstance(roots, sp.FiniteSet):
        try:  # покажем и отброшенные посторонние корни
            _candidates_with_check(sol, lhs, rhs, dom)
        except SolveError:
            pass
    if isinstance(roots, sp.FiniteSet):
        for r in sorted(roots, key=float):
            L, R = sp.simplify(lhs.subs(x, r)), sp.simplify(rhs.subs(x, r))
            sol.add(f"Проверка x = {txt(r)}: {txt(L)} = {txt(R)} ✓", rf"x={tex(r)}:\ {tex(L)} = {tex(R)}\ \checkmark")
    sol.answer = roots
    sol.answer_text, sol.answer_tex = set_text(roots)
    if roots == sp.S.EmptySet:
        sol.answer_text = "нет корней"
    return sol


def _candidates_with_check(sol: Solution, lhs, rhs, dom):
    """Запасной путь: кандидаты от sp.solve (после логарифмирования/возведения в степень),
    затем строгая проверка подстановкой; посторонние корни отбрасываем."""
    f = sp.expand_log(sp.logcombine(lhs - rhs, force=True), force=False)
    try:
        cands = set(sp.solve(sp.Eq(lhs, rhs), x, check=False)) | set(sp.solve(f, x, check=False))
    except Exception as e:
        raise SolveError("Уравнение не решается точно") from e
    good = []
    for c in cands:
        c = sp.nsimplify(sp.simplify(c))
        if not c.is_real:
            continue
        L, R = lhs.subs(x, c), rhs.subs(x, c)
        if c in dom and L.is_real and R.is_real and sp.simplify(L - R) == 0:
            good.append(c)
        else:
            sol.add(f"x = {txt(c)} — посторонний корень (не входит в ОДЗ или не подходит при подстановке)",
                    rf"x = {tex(c)}\ \text{{— посторонний корень}}")
    if not cands:
        raise SolveError("Уравнение не решается точно")
    return sp.FiniteSet(*good)


def _polynomial_steps(sol: Solution, P: sp.Poly):
    deg = P.degree()
    sol.add(f"Перенесём всё в одну часть: {txt(P.as_expr())} = 0", f"{tex(P.as_expr())} = 0")
    if deg == 2:
        A, B, C = P.all_coeffs()
        D = sp.simplify(B ** 2 - 4 * A * C)
        sol.add(f"D = b² − 4ac = ({txt(B)})² − 4·{txt(A)}·({txt(C)}) = {txt(D)}",
                rf"D = b^2-4ac = ({tex(B)})^2 - 4\cdot {tex(A)}\cdot ({tex(C)}) = {tex(D)}")
        if D.is_negative:
            sol.add("D < 0 — действительных корней нет", r"D<0\ \Rightarrow\ \varnothing")
        elif D == 0:
            sol.add(f"D = 0 — один корень: x = −b/(2a) = {txt(-B / (2 * A))}", rf"x = -\frac{{b}}{{2a}} = {tex(-B / (2 * A))}")
        else:
            r1, r2 = sp.simplify((-B - sp.sqrt(D)) / (2 * A)), sp.simplify((-B + sp.sqrt(D)) / (2 * A))
            sol.add(f"x = (−b ± √D)/(2a): x₁ = {txt(r1)}, x₂ = {txt(r2)}",
                    rf"x_{{1,2}} = \frac{{-b\pm\sqrt{{D}}}}{{2a}}:\ x_1 = {tex(r1)},\ x_2 = {tex(r2)}")
    elif deg == 4 and all(c == 0 for c in P.all_coeffs()[1::2]):
        t = sp.Symbol("t")
        A, _, B, _, C = P.all_coeffs()
        sol.add(f"Биквадратное: замена t = x² ≥ 0: {txt(A * t ** 2 + B * t + C)} = 0",
                rf"t = x^2 \ge 0:\ {tex(A * t ** 2 + B * t + C)} = 0")
        ts = sp.solveset(A * t ** 2 + B * t + C, t, sp.S.Reals)
        sol.add(f"t = {', '.join(txt(v) for v in ts)}; берём t ≥ 0 и x = ±√t",
                rf"t \in {tex(ts)};\ x = \pm\sqrt{{t}}")
    elif deg >= 3:
        fac = sp.factor(P.as_expr())
        if fac != P.as_expr():
            sol.add(f"Разложим на множители: {txt(fac)} = 0", f"{tex(fac)} = 0")


def _sign_table(sol, f, var=x):
    """Метод интервалов: нули и точки разрыва f, знак на каждом промежутке."""
    pts = set()
    z = sp.solveset(f, var, sp.S.Reals)
    if isinstance(z, sp.FiniteSet):
        pts |= set(z)
    num, den = sp.fraction(sp.together(f))
    d0 = sp.solveset(den, var, sp.S.Reals)
    if isinstance(d0, sp.FiniteSet):
        pts |= set(d0)
    pts = sorted(pts, key=float)
    if not pts:
        return
    sol.add("Нули и точки разрыва: " + ", ".join(txt(p) for p in pts),
            r"\text{Нули и разрывы: } " + ",\\ ".join(tex(p) for p in pts))
    bounds = [-sp.oo] + pts + [sp.oo]
    signs = []
    for lo, hi in zip(bounds, bounds[1:]):
        if lo == -sp.oo and hi == sp.oo:
            tpt = 0
        elif lo == -sp.oo:
            tpt = hi - 1
        elif hi == sp.oo:
            tpt = lo + 1
        else:
            tpt = (lo + hi) / 2
        v = f.subs(var, tpt)
        signs.append("+" if v.is_positive else "−" if v.is_negative else "?")
    sol.add("Знаки на промежутках: " + "  ".join(signs))


def solve_inequality(lhs, op, rhs) -> Solution:
    sol = Solution(kind="inequality")
    f = sp.simplify(lhs - rhs)
    rel = {">": sp.Gt, "<": sp.Lt, ">=": sp.Ge, "<=": sp.Le}[op](lhs, rhs)
    sol.add(f"Перенесём всё влево: {txt(f)} {op.replace('>=', '≥').replace('<=', '≤')} 0",
            f"{tex(f)} {dict([('>', '>'), ('<', '<'), ('>=', r'\ge'), ('<=', r'\le')])[op]} 0")
    dom = _domain(f)
    if dom != sp.S.Reals:
        t, T = set_text(dom)
        sol.add(f"ОДЗ: {t}", rf"\text{{ОДЗ: }} {T}")
    try:
        _sign_table(sol, f)
    except Exception:
        pass
    S = sp.solveset(rel, x, sp.S.Reals)
    if isinstance(S, sp.ConditionSet):
        raise SolveError("Неравенство не решается точно")
    sol.answer = S
    sol.answer_text, sol.answer_tex = set_text(S)
    return sol


def solve_system(parts) -> Solution:
    sol = Solution(kind="system")
    eqs = []
    for p in parts:
        l, op, r = relation(p)
        eqs.append(sp.Eq(l, r))
    sol.add("Система: " + ", ".join(f"{txt(e.lhs)} = {txt(e.rhs)}" for e in eqs),
            r"\begin{cases}" + r"\\".join(f"{tex(e.lhs)} = {tex(e.rhs)}" for e in eqs) + r"\end{cases}")
    res = sp.solve(eqs, [x, y], dict=True)
    if not res:
        sol.answer = []
        sol.answer_text, sol.answer_tex = "нет решений", r"\varnothing"
        sol.add("Решений нет")
        return sol
    if all(e.lhs.is_polynomial(x, y) and sp.Poly(e.lhs - e.rhs, x, y).total_degree() == 1 for e in eqs):
        sol.add("Выразим одну переменную из первого уравнения и подставим во второе")
    for r in res:
        if not {x, y} <= set(r):
            sol.add("Уравнения пропорциональны — система имеет бесконечно много решений")
            sol.answer = res
            sol.answer_text = "бесконечно много решений: " + ", ".join(f"{k} = {txt(v)}" for k, v in r.items())
            sol.answer_tex = r"\text{бесконечно много: }" + r",\ ".join(f"{k} = {tex(v)}" for k, v in r.items())
            return sol
        for e in eqs:
            sol.add(f"Проверка: {txt(e.lhs.subs(r))} = {txt(e.rhs.subs(r))} ✓")
    sol.answer = res
    sol.answer_text = "; ".join(f"x = {txt(r[x])}, y = {txt(r[y])}" for r in res)
    sol.answer_tex = r";\ ".join(f"x = {tex(r[x])},\\ y = {tex(r[y])}" for r in res)
    return sol


# ---------------------------------------------------------------- параметры

def _count(eq_expr, a_val):
    """Сколько различных действительных x решают уравнение при a = a_val (None — не удалось)."""
    S = sp.solveset(sp.simplify(eq_expr.subs(a, a_val)), x, sp.S.Reals)
    if isinstance(S, sp.FiniteSet):
        return len(S)
    if S == sp.S.EmptySet:
        return 0
    if isinstance(S, (sp.Interval, sp.Union)):
        return sp.oo
    return None


def _critical_values(E):
    """Значения a, при которых может меняться число решений уравнения E(x, a) = 0."""
    cands, how = set(), ""
    if sp.degree(sp.expand(E), a) == 1 if E.is_polynomial(a) else False:
        A = sp.expand(E).coeff(a, 1)
        B = sp.simplify(E - A * a)
        F = sp.simplify(-B / A)
        how = (f"Выразим параметр: a = {txt(F)}. Число решений — число точек пересечения прямой y = a с графиком",
               rf"\text{{Выразим параметр: }} a = {tex(F)}\quad \text{{(ищем пересечения прямой }} y=a \text{{ с графиком)}}")
        pts = set()
        Fp = sp.piecewise_fold(F.rewrite(sp.Piecewise))
        pieces = Fp.args if isinstance(Fp, sp.Piecewise) else [(F, True)]
        taken = sp.false
        for g, c in pieces:  # условие куска — его собственное и не выполнены предыдущие
            eff = sp.And(c, sp.Not(taken))
            taken = sp.Or(taken, c)
            st = sp.solveset(sp.diff(g, x), x, sp.S.Reals)
            if isinstance(st, sp.FiniteSet):
                pts |= {s for s in st if bool(eff.subs(x, s))}
        for ab in F.atoms(sp.Abs):
            z = sp.solveset(ab.args[0], x, sp.S.Reals)
            if isinstance(z, sp.FiniteSet):
                pts |= set(z)
        z = sp.solveset(A, x, sp.S.Reals)
        special = set(z) if isinstance(z, sp.FiniteSet) else set()
        dom = _domain(F)
        if isinstance(dom.boundary, sp.FiniteSet):
            special |= set(dom.boundary)
        for p in pts:
            v = sp.simplify(F.subs(x, p))
            if v.is_finite and v.is_real:
                cands.add(v)
        for p in special:
            for d in ("+", "-"):
                v = sp.limit(F, x, p, d)
                if v.is_finite and v.is_real:
                    cands.add(v)
        for inf in (sp.oo, -sp.oo):
            v = sp.limit(F, x, inf)
            if v.is_finite and v.is_real:
                cands.add(v)
    elif E.is_polynomial(x):
        P = sp.Poly(sp.expand(E), x)
        how = ("Число корней многочлена меняется только там, где обращается в ноль дискриминант или старший коэффициент", "")
        for q in [P.LC()] + ([sp.discriminant(P)] if P.degree() >= 2 else []):
            z = sp.solveset(q, a, sp.S.Reals)
            if isinstance(z, sp.FiniteSet):
                cands |= set(z)
            elif z != sp.S.EmptySet:
                raise SolveError("Не удалось найти критические значения параметра")
    else:
        raise SolveError("Такую задачу с параметром решать не умею")
    return sorted(cands, key=float), how


def solve_param(body: str, cond: str) -> Solution:
    sol = Solution(kind="param", var="a")
    l, op, r = relation(body)
    if op != "=":
        raise SolveError("С параметром поддерживаются только уравнения")
    E = sp.simplify(l - r)
    if m := re.fullmatch(r"x=(-?[\d./]+)", cond):
        x0 = sp.nsimplify(m[1])
        eq = sp.simplify(E.subs(x, x0))
        sol.add(f"Подставим x = {txt(x0)}: {txt(eq)} = 0", rf"x={tex(x0)}:\ {tex(eq)} = 0")
        S = sp.solveset(eq, a, sp.S.Reals)
        sol.answer = S
        sol.answer_text, sol.answer_tex = set_text(S, "a")
        return sol
    if cond in COUNT_WORDS:
        target = COUNT_WORDS[cond]
    elif m := re.fullmatch(r"ровно (\d+)", cond):
        target = int(m[1])
    elif cond == "есть корни":
        target = "≥1"
    else:
        raise SolveError("Не понял условие задачи с параметром")

    cands, how = _critical_values(E)
    sol.add(*how)
    if cands:
        sol.add("Особые значения параметра: a = " + ", ".join(txt(c) for c in cands),
                r"\text{Особые значения: } a \in \{" + ",\\ ".join(tex(c) for c in cands) + r"\}")
    bounds = [-sp.oo] + cands + [sp.oo]
    regions = []
    for lo, hi in zip(bounds, bounds[1:]):
        if lo == -sp.oo and hi == sp.oo:
            t = sp.Integer(0)
        elif lo == -sp.oo:
            t = sp.floor(hi) - 1
        elif hi == sp.oo:
            t = sp.ceiling(lo) + 1
        else:
            t = (lo + hi) / 2
        regions.append((sp.Interval.open(lo, hi), _count(E, t)))
    for c in cands:
        regions.append((sp.FiniteSet(c), _count(E, c)))
    if any(n is None for _, n in regions):
        raise SolveError("Не удалось посчитать число решений")
    regions.sort(key=lambda rn: (float(rn[0].inf) if rn[0].inf.is_finite else -1e18, isinstance(rn[0], sp.Interval)))
    table, rows = [], []
    for R, n in regions:
        t, T = set_text(R, "a")
        cnt = "бесконечно много" if n == sp.oo else str(n)
        table.append(f"{t}: {cnt}")
        rows.append(rf"{T} & {cnt if n != sp.oo else r'\infty'}")
    sol.add("Число решений: " + ";  ".join(table),
            r"\begin{array}{l|c} \text{параметр} & \text{решений} \\ \hline " + r" \\ ".join(rows) + r" \end{array}")
    ok = [R for R, n in regions if (n >= 1 if target == "≥1" else n == target)]
    S = sp.Union(*ok) if ok else sp.S.EmptySet
    sol.answer = S
    sol.answer_text, sol.answer_tex = set_text(S, "a")
    if S == sp.S.EmptySet:
        sol.answer_text = "таких a нет"
    return sol
