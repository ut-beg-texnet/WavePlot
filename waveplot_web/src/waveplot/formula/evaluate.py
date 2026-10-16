"""Evaluation of formula syntax trees against a dataset's histories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..science.compat import Mode, Provenance
from ..science.signal import DEFAULT_FOURIER, FourierSettings
from .expr import parse_expr
from .legacy import Diagnostic, parse_legacy
from .nodes import Call, Hist, Node, Num, Self, Undefined
from .ops import OPS, FormulaError, OpSpec, Param, op
from .printer import to_expr
from .trace import Trace

V407_MAX_SAMPLES = 4096


@op("unsupported", "Disabled v407 function", "signal", 2, params=(Param("name", "", "str"),),
    legacy="V W Z A B", doc="Phase/group-velocity functions whose code is commented out in v407.")
def _unsupported(a, p, ctx):
    raise FormulaError(f"v407 function '{p.get('name')}' (phase/group velocity) is disabled and has no result")


@dataclass
class EvalContext:
    load: Callable[[int], Trace]
    self_ref: int | None = None
    mode: Mode = "standard"
    settings: FourierSettings = DEFAULT_FOURIER
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: Callable[[int], tuple[int, int, int]] = lambda h: (0, 0, 0)
    provenance: Provenance = field(default_factory=Provenance)
    warnings: list[str] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


def evaluate(node: Node, ctx: EvalContext) -> Trace:
    key = node
    if key in ctx._cache:
        return ctx._cache[key]
    out = _eval(node, ctx)
    ctx._cache[key] = out
    return out


def _eval(node: Node, ctx: EvalContext) -> Trace:
    if isinstance(node, Hist):
        return ctx.load(node.number)
    if isinstance(node, Self):
        if not ctx.self_ref:
            raise FormulaError("the formula refers to the trace's own history ('self'), but none is set")
        return ctx.load(ctx.self_ref)
    if isinstance(node, Num):
        return Trace(np.array([node.value]), meta={"scalar": True}, label=str(node.value))
    if isinstance(node, Undefined):
        raise FormulaError("an operand has no data (v407: binary operator followed by no [n])")
    if not isinstance(node, Call):
        raise FormulaError(f"cannot evaluate {node!r}")
    spec: OpSpec | None = OPS.get(node.op)
    if spec is None:
        raise FormulaError(f"unknown operation '{node.op}'")
    args: list[Trace] = []
    for k, a in enumerate(node.args):
        if isinstance(a, Undefined) and ctx.mode == "v407" and k > 0 and args:
            ctx.warn("right operand has no history; v407 used an uninitialised buffer (zeros used here)")
            args.append(args[0].with_y(np.zeros(args[0].n)))
            continue
        args.append(evaluate(a, ctx))
    if spec.arity == 1 and args and args[0].meta.get("scalar"):
        raise FormulaError(f"{node.op} needs a trace, not a number")
    params = {**spec.defaults(), **node.param_dict}
    return spec.impl(args, params, ctx)


def make_loader(ds, mode: Mode, provenance: Provenance | None = None) -> Callable[[int], Trace]:
    def load(n: int) -> Trace:
        if n < 1 or n > len(ds.histories):
            raise FormulaError(f"history {n} does not exist (1–{len(ds.histories)})")
        rec = ds.history_record(n)
        y = ds.history_data(n).astype(float)
        t0, dt = rec.tstart, rec.dt
        if mode == "v407" and len(y) > V407_MAX_SAMPLES:
            y = y[:V407_MAX_SAMPLES]
            if provenance is not None:
                provenance.add("hist_truncate")
        pos = "global" if rec.is_global else f"({rec.position[0]},{rec.position[1]},{rec.position[2]})"
        return Trace(y, t0, dt, rec.x_unit, rec.units, f"h{n} {rec.name} {pos}", meta={"history": n})
    return load


def make_context(ds, self_ref: int | None = None, mode: Mode = "standard",
                 settings: FourierSettings | None = None) -> EvalContext:
    prov = Provenance(mode=mode)
    sp = ds.spacing

    def position(h: int) -> tuple[int, int, int]:
        return ds.history_record(h).position

    return EvalContext(load=make_loader(ds, mode, prov), self_ref=self_ref, mode=mode,
                       settings=settings or DEFAULT_FOURIER, spacing=(sp["i"], sp["j"], sp["k"]),
                       position=position, provenance=prov)


@dataclass
class EvalResult:
    trace: Trace | None
    node: Node | None
    syntax: str                       # "modern" | "legacy"
    modern: str                       # expression in modern syntax
    diagnostics: list[str]
    provenance: Provenance
    error: str | None = None


def parse_any(text: str, prefer: str = "auto") -> tuple[Node, str, list[Diagnostic]]:
    """Parse modern syntax, falling back to the legacy v407 language (or force one with ``prefer``)."""
    text = text.strip()
    if prefer in ("auto", "modern"):
        try:
            return parse_expr(text), "modern", []
        except FormulaError:
            if prefer == "modern":
                raise
    node, diags = parse_legacy(text)
    return node, "legacy", diags


def evaluate_text(ds, text: str, self_ref: int | None = None, mode: Mode = "standard",
                  settings: FourierSettings | None = None, prefer: str = "auto") -> EvalResult:
    ctx = make_context(ds, self_ref, mode, settings)
    try:
        node, syntax, diags = parse_any(text, prefer)
    except FormulaError as exc:
        return EvalResult(None, None, "modern", "", [], ctx.provenance, str(exc))
    msgs = [str(d) for d in diags]
    modern = to_expr(node)
    if any(d.severity == "error" for d in diags) and syntax == "legacy":
        pass  # still try to evaluate what could be parsed
    try:
        tr = evaluate(node, ctx)
    except (FormulaError, ValueError, IndexError, ZeroDivisionError) as exc:
        return EvalResult(None, node, syntax, modern, msgs + ctx.warnings, ctx.provenance, str(exc))
    tr = tr.replace(label=modern if len(modern) < 80 else modern[:77] + "…")
    return EvalResult(tr, node, syntax, modern, msgs + ctx.warnings, ctx.provenance)


_ = Undefined
