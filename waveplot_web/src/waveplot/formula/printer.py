"""Printing syntax trees as modern expressions (pipe syntax for single-input operations)."""

from __future__ import annotations

from .nodes import Call, Hist, Node, Num, Self, Undefined
from .ops import OPS

_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "pow": 4}
PIPE_PREC = 0
ATOM = 10


def fmt_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"
    if isinstance(v, str):
        return repr(v)
    return str(v)


def _params(node: Call) -> str:
    return ", ".join(f"{k}={fmt_value(v)}" for k, v in node.params)


def _expr(node: Node) -> tuple[str, int]:
    if isinstance(node, Hist):
        return f"h{node.number}", ATOM
    if isinstance(node, Self):
        return "self", ATOM
    if isinstance(node, Num):
        s = fmt_value(node.value)
        return s, (ATOM if not s.startswith("-") else 3)
    if isinstance(node, Undefined):
        return "undefined", ATOM
    if not isinstance(node, Call):
        return "?", ATOM
    spec = OPS.get(node.op)
    if node.op == "neg" and len(node.args) == 1:
        s, p = _expr(node.args[0])
        return f"-{s if p >= 3 else f'({s})'}", 3
    if spec is not None and spec.infix and len(node.args) == 2 and not node.params:
        prec = _PREC[node.op]
        ls, lp = _expr(node.args[0])
        rs, rp = _expr(node.args[1])
        if lp < prec or (node.op == "pow" and lp == prec):
            ls = f"({ls})"
        if rp < prec or (rp == prec and node.op in ("sub", "div")) or (rp == prec and node.op == "add"
                                                                       and _is_sub(node.args[1])):
            rs = f"({rs})"
        return f"{ls} {spec.infix} {rs}", prec
    arity = spec.arity if spec else len(node.args)
    if arity == 1 and len(node.args) == 1:
        s, p = _expr(node.args[0])
        if p < ATOM and p != PIPE_PREC:
            s = f"({s})"
        args = _params(node)
        return f"{s} | {node.op}" + (f"({args})" if args else ""), PIPE_PREC
    parts = []
    for a in node.args:
        s, _p = _expr(a)
        parts.append(s)
    if node.params:
        parts.append(_params(node))
    return f"{node.op}({', '.join(parts)})", ATOM


def _is_sub(node: Node) -> bool:
    return isinstance(node, Call) and node.op == "sub"


def to_expr(node: Node) -> str:
    return _expr(node)[0]
