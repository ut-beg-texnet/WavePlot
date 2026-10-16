"""Modern expression syntax (safe: parsed with ``ast`` against a whitelist, never ``eval``).

Examples::

    h12                          history 12            self      the trace slot's history
    h2 | integrate | bandpass(0.5, 10)                 pipe syntax for single-input operations
    (h3 + h4) / 2                normal precedence      rotate(h5, h6, theta=30)
    vector_magnitude(h=4)        operations reading several histories
"""

from __future__ import annotations

import ast
import math
import re

from .nodes import Call, Hist, Node, Num, Self, Undefined
from .ops import OPS, FormulaError

MAX_LEN = 2000
MAX_NODES = 500
_HNAME = re.compile(r"^h(\d+)$")
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_CONST = {"pi": math.pi, "e": math.e}


def parse_expr(text: str) -> Node:
    text = text.strip()
    if not text:
        raise FormulaError("empty expression")
    if len(text) > MAX_LEN:
        raise FormulaError(f"expression longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"syntax error at column {exc.offset}: {exc.msg}") from exc
    count = sum(1 for _ in ast.walk(tree))
    if count > MAX_NODES:
        raise FormulaError("expression too complex")
    return _conv(tree.body)


def _number(node: ast.AST) -> float | str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _number(node.operand)
        if isinstance(v, int | float):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Name) and node.id in _CONST:
        return _CONST[node.id]
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    return "__notnum__"  # sentinel


def _conv(node: ast.AST) -> Node:
    if isinstance(node, ast.Name):
        if node.id == "self":
            return Self()
        if node.id == "undefined":
            return Undefined()
        m = _HNAME.match(node.id)
        if m:
            return Hist(int(m.group(1)))
        if node.id in _CONST:
            return Num(_CONST[node.id])
        if node.id in OPS:
            raise FormulaError(f"'{node.id}' is an operation: apply it with '| {node.id}' or '{node.id}(…)'")
        raise FormulaError(f"unknown name '{node.id}' (use h<number>, self, or an operation)")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return Num(float(node.value))
        raise FormulaError(f"unexpected constant {node.value!r}")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "h":
            v = _number(node.slice)
            if isinstance(v, int | float) and float(v).is_integer():
                return Hist(int(v))
        raise FormulaError("only h[<number>] subscripts are allowed")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _conv(node.operand)
        if isinstance(node.op, ast.USub):
            inner = _conv(node.operand)
            if isinstance(inner, Num):
                return Num(-inner.value)
            return Call("neg", (inner,))
        raise FormulaError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitOr):
            return _pipe(_conv(node.left), node.right)
        opname = _BIN.get(type(node.op))
        if opname is None:
            raise FormulaError(f"unsupported operator {type(node.op).__name__}")
        return Call(opname, (_conv(node.left), _conv(node.right)))
    if isinstance(node, ast.Call):
        return _call(node, None)
    raise FormulaError(f"unsupported syntax: {type(node).__name__}")


def _pipe(left: Node, right: ast.AST) -> Node:
    if isinstance(right, ast.Name):
        if right.id not in OPS:
            raise FormulaError(f"unknown operation '{right.id}'")
        spec = OPS[right.id]
        if spec.arity != 1:
            raise FormulaError(f"'{right.id}' takes {spec.arity} inputs and cannot be used in a pipe")
        return Call(right.id, (left,))
    if isinstance(right, ast.Call):
        return _call(right, left)
    raise FormulaError("the right side of '|' must be an operation")


def _call(node: ast.Call, piped: Node | None) -> Node:
    if not isinstance(node.func, ast.Name):
        raise FormulaError("only plain operation names can be called")
    name = node.func.id
    spec = OPS.get(name)
    if spec is None:
        raise FormulaError(f"unknown operation '{name}'")
    args: list[Node] = [piped] if piped is not None else []
    params: dict[str, object] = {}
    pnames = [p.name for p in spec.params]
    pos_params: list = []
    for a in node.args:
        if len(args) < spec.arity:
            args.append(_conv(a))
        else:
            v = _number(a)
            if v == "__notnum__":
                raise FormulaError(f"{name}: parameters must be numbers or quoted words")
            pos_params.append(v)
    if len(pos_params) > len(pnames):
        raise FormulaError(f"{name}: too many parameters (expects {', '.join(pnames) or 'none'})")
    for k, v in zip(pnames, pos_params, strict=False):
        params[k] = v
    for kw in node.keywords:
        if kw.arg not in pnames:
            raise FormulaError(f"{name}: unknown parameter '{kw.arg}' (expects {', '.join(pnames) or 'none'})")
        v = _number(kw.value)
        if v == "__notnum__":
            raise FormulaError(f"{name}: parameter '{kw.arg}' must be a number or a quoted word")
        params[kw.arg] = v
    if len(args) != spec.arity:
        raise FormulaError(f"{name} takes {spec.arity} trace input(s), got {len(args)}")
    for p in spec.params:
        if p.kind == "choice" and p.name in params and params[p.name] is not None \
                and params[p.name] not in p.choices:
            raise FormulaError(f"{name}: {p.name} must be one of {', '.join(map(str, p.choices))}")
    return Call(name, tuple(args), tuple((k, v) for k, v in params.items() if v is not None))
