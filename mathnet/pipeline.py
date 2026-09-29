"""Сеть + проверка + символьный решатель.

1. Символьный решатель (SymPy, в отдельном процессе с таймаутом) даёт эталонный ответ.
2. MathNet пишет решение: сначала жадно, потом несколько вариантов сэмплированием.
   Вариант принимается, только если каждое числовое равенство в шагах верно
   и итоговый ответ совпадает с эталоном (или, если эталона нет, проходит подстановку).
3. Если ни один вариант MathNet не прошёл проверку — показываем решение SymPy.
"""
import concurrent.futures as cf
import multiprocessing as mp
import re

import json
from pathlib import Path

import numpy as np
import sympy as sp

from . import solver
from .data import BLOCK_SIZE, STOI
from .solver import Solution, SolveError, a, x, y

SIGNATURES = Path(__file__).resolve().parent.parent / "weights" / "mathnet_signatures.json"
SOLVER_TIMEOUT = 90
NET_SAMPLES = 6
NET_TEMPERATURE = 0.7


# ---------------------------------------------------------------- решатель в отдельном процессе

def _solve_worker(problem: str):
    try:
        return solver.solve(problem), None
    except SolveError as e:
        return None, str(e)
    except Exception as e:  # SymPy иногда падает на экзотике — это не повод ронять сайт
        return None, f"Решатель не справился ({type(e).__name__})"


class SolverPool:
    """Один процесс для SymPy: зависший расчёт убиваем по таймауту и поднимаем процесс заново."""

    def __init__(self):
        self._pool = None

    def _get(self):
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        return self._pool

    def solve(self, problem: str, timeout=SOLVER_TIMEOUT) -> tuple[Solution | None, str | None]:
        fut = self._get().submit(_solve_worker, problem)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError:
            for p in list(self._pool._processes.values()):
                p.kill()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            return None, "Решатель не уложился во время"
        except cf.process.BrokenProcessPool:
            self._pool = None
            return None, "Решатель упал"


# ---------------------------------------------------------------- проверка решения MathNet

NUMERIC = re.compile(r"^[\d+\-*/^().√ ]+$")


def _num(s: str):
    return sp.nsimplify(solver.expr(s))


def arithmetic_ok(steps: list[str]) -> bool:
    """В каждом шаге все чисто числовые части цепочки «…=…=…» должны быть равны."""
    for st in steps:
        for chain in re.split(r"[<>]=?", st):
            nums = [p for p in chain.split("=") if p and NUMERIC.match(p)]
            try:
                vals = [_num(p) for p in nums]
            except Exception:
                return False
            if any(sp.simplify(v - vals[0]) != 0 for v in vals[1:]):
                return False
    return True


SIMPLE_REL = re.compile(r"^(-?[\d/]+)?(<=|>=|<|>)?([axy])(<=|>=|<|>|=)(-?[\d/()]+)$")


def _rel_set(line: str, var):
    """'a>16', '-8<a<8', 'x<=-9', 'a=6' -> множество значений переменной."""
    m = SIMPLE_REL.match(line)
    if not m or (m[1] is None) != (m[2] is None):
        return None
    ops = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}
    v = _num(m[5])
    S = sp.solveset(sp.Eq(var, v) if m[4] == "=" else ops[m[4]](var, v), var, sp.S.Reals)
    if m[1] is not None:  # двойное неравенство -8<a<8
        S = S & sp.solveset(ops[m[2]](_num(m[1]), var), var, sp.S.Reals)
    return S


def net_answer(steps: list[str], kind: str):
    """Итоговый ответ из шагов MathNet в виде, сравнимом с ответом решателя."""
    if steps and steps[-1] == "нет корней":
        return sp.S.EmptySet
    if kind == "expr":
        last = steps[-1].split("=")[-1]
        return solver.expr(last)
    if kind == "system":
        vals = {}
        for st in steps:
            if m := re.match(r"^([xy])=.*=(-?\d+)$", st):
                vals[{"x": x, "y": y}[m[1]]] = sp.Integer(m[2])
        return [vals] if len(vals) == 2 else None
    var = a if kind == "param" else x
    if kind in ("param", "inequality"):
        acc = sp.S.EmptySet
        for st in reversed(steps):
            S = _rel_set(st.replace(" ", ""), var)
            if S is None:
                break
            acc = acc | S
        return acc if acc != sp.S.EmptySet else None
    # уравнение: строки вида x=..., иначе x1=..., x2=...
    numeric = lambda st: all(NUMERIC.match(p) for p in st.split("=")[1:])  # 'x=...' где справа только числа
    plain = [st for st in steps if st.startswith("x=") and numeric(st)]
    lines = plain or [st for st in steps if re.match(r"^x[12]=", st) and numeric(st)]
    try:
        return sp.FiniteSet(*[_num(st.split("=")[-1]) for st in lines]) if lines else None
    except Exception:
        return None


def same_answer(net, ref) -> bool:
    try:
        if isinstance(ref, list) or isinstance(net, list):
            return net is not None and len(net) == len(ref) and all(
                all(sp.simplify(n[k] - r[k]) == 0 for k in r) for n, r in zip(net, ref))
        if isinstance(ref, sp.Set):
            return isinstance(net, sp.Set) and (net == ref or sp.simplify(sp.SymmetricDifference(net, ref)) == sp.S.EmptySet)
        return sp.simplify(net - ref) == 0
    except Exception:
        return False


def substitution_ok(problem: str, ans) -> bool:
    """Если эталона нет: корни уравнения должны обращать его в верное равенство."""
    try:
        l, op, r = solver.relation(problem.split(";")[0])
        if op != "=" or not isinstance(ans, sp.FiniteSet) or not ans:
            return False
        return all(sp.simplify((l - r).subs(x, v)) == 0 for v in ans)
    except Exception:
        return False


def guess_kind(problem: str) -> str:
    body, _, cond = problem.partition(";")
    if cond:
        return "param"
    if re.fullmatch(r"\(.*\)'", body) or "=" not in body and not re.search(r"[<>]", body):
        return "expr"
    if len(solver.split_top(body, ",")) == 2:
        return "system"
    return "inequality" if re.search(r"[<>]", body) else "equation"


# ---------------------------------------------------------------- TeX для шагов MathNet

def step_tex(st: str) -> str:
    if re.search(r"[а-я]", st):
        return rf"\text{{{st}}}"
    parts = re.split(r"(<=|>=|<|>|=)", st)
    out = []
    for p in parts:
        if p in ("<=", ">=", "<", ">", "="):
            out.append({"<=": r"\le", ">=": r"\ge"}.get(p, p))
        elif m := re.fullmatch(r"(x|D)([12xy])", p):
            out.append(f"{m[1]}_{{{m[2]}}}")
        else:
            try:
                # log2(u) -> функция log_2(u), чтобы в TeX осталось основание
                q = re.sub(r"log(\d+)\(", r"log_\1(", p)
                local = dict(solver.LOCALS, **{f"log_{b}": sp.Function(f"log_{b}") for b in re.findall(r"log_(\d+)", q)})
                q = q.replace("√", "sqrt") if "√(" in q else solver.to_sympy_syntax(q)
                e = sp.parse_expr(q, local_dict=local, transformations=solver.TRANSFORMS, evaluate=False)
                out.append(sp.latex(e))
            except Exception:
                out.append(rf"\text{{{p}}}")
    return " ".join(out)


# ---------------------------------------------------------------- типы задач, знакомые сети

def signature(problem: str) -> str:
    """Форма задачи без конкретных чисел: 'x^2-5x+6=0' -> 'x^N-Nx+N=0'."""
    return re.sub(r"\d+", "N", problem)


_signatures = None


def known_signatures() -> set[str]:
    """Формы задач из обучающих генераторов (строится build_signatures в train.py)."""
    global _signatures
    if _signatures is None:
        _signatures = set(json.loads(SIGNATURES.read_text())) if SIGNATURES.exists() else set()
    return _signatures


def build_signatures(n=400_000) -> set[str]:
    from .data import sample
    return {signature(sample()[0]) for _ in range(n)}


# ---------------------------------------------------------------- главный вход

def solve_problem(problem: str, net, pool: SolverPool) -> dict:
    ref, why = pool.solve(problem)
    kind = ref.kind if ref else guess_kind(problem)
    result = {"problem": problem, "kind": kind}

    attempts, first = [], None
    net_ok = (all(c in STOI and c not in "?._" for c in problem) and len(problem) <= BLOCK_SIZE // 2
              and signature(problem) in known_signatures())
    if net_ok:
        rng = np.random.default_rng(0)
        for i in range(1 + NET_SAMPLES):
            text, conf = net.solve(problem, temperature=0 if i == 0 else NET_TEMPERATURE, rng=rng)
            steps = [s for s in text.split(";") if s]
            if first is None:
                first = steps
            if text in attempts:
                continue
            attempts.append(text)
            if not steps or not arithmetic_ok(steps):
                continue
            try:
                ans = net_answer(steps, kind)
            except Exception:
                ans = None
            if ans is None:
                continue
            ok = same_answer(ans, ref.answer) if ref else (kind == "equation" and substitution_ok(problem, ans))
            if ok:
                result.update(source="mathnet", verified=True, attempts=len(attempts),
                              steps=[{"text": s, "tex": step_tex(s)} for s in steps],
                              answer={"text": ref.answer_text if ref else steps[-1],
                                      "tex": ref.answer_tex if ref else step_tex(steps[-1])})
                return result

    if ref is not None:
        result.update(source="solver", verified=True, attempts=len(attempts),
                      steps=[{"text": s.text, "tex": s.tex} for s in ref.steps],
                      answer={"text": ref.answer_text, "tex": ref.answer_tex},
                      note=("Нейросеть не нашла проверенного решения — показано точное решение символьного решателя."
                            if net_ok else "Такой тип задачи нейросеть не знает — решил символьный решатель."))
        if first:
            result["net_attempt"] = [{"text": s, "tex": step_tex(s)} for s in first]
        return result

    result.update(source="none", verified=False, attempts=len(attempts), error=why or "Не удалось решить задачу",
                  steps=[{"text": s, "tex": step_tex(s)} for s in (first or [])])
    return result
