"""Parser for the WavePlot v407 history formula language (``DSPFORM.PAS`` ``Parse``).

Semantics reproduced exactly:

* the accumulator starts as the trace slot's own history (``self``); ``[n]`` replaces it;
* unary operators apply to the accumulator; ``op,p1,p2(…)`` first evaluates the bracket group;
* a binary operator (``+ - * / C R r``, ``$…``) takes **the whole rest of the formula** as its
  right operand, which starts from an empty buffer — so ``[1]*[2]+[3]`` means ``h1*(h2+h3)``;
* parameters are ``,value`` lists directly after the operator (at most 5); missing ones are 0
  unless the operator has its own default.

Unlike v407, unknown characters, malformed numbers and unbalanced brackets produce diagnostics
(with column numbers) instead of being skipped silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..science.compat import pas_round, pas_trunc
from .nodes import Call, Hist, Node, Self, Undefined, call

MAX_PARAMS = 5
NUM_CHARS = set("0123456789.-+eE")
FUNC1 = set("IiDdFfPpTtMmSsXxEe#LlJj")
FUNC2 = set("+-*/CcRr")
FUNC2B = set("GgVvWwZzAaBb")
AMP = set("FfXxDdRrWwpPQqVvTt")
DOLLAR = set("GgHh*")


@dataclass
class Diagnostic:
    message: str
    column: int | None = None
    severity: str = "warning"      # "warning" | "error"

    def __str__(self) -> str:
        where = f" (column {self.column + 1})" if self.column is not None else ""
        return f"{self.severity}{where}: {self.message}"


class _State:
    def __init__(self, text: str):
        self.text = text
        self.diags: list[Diagnostic] = []

    def warn(self, msg: str, col: int | None = None, severity: str = "warning") -> None:
        self.diags.append(Diagnostic(msg, col, severity))


def parse_legacy(text: str) -> tuple[Node, list[Diagnostic]]:
    st = _State(text)
    if len(text) > 255:
        st.warn("formula longer than 255 characters (v407 truncated it)")
    node = _parse(st, text, Self(), 0)
    return node, st.diags


def _skip_spaces(s: str, pos: int) -> int:
    while pos < len(s) and s[pos] == " ":
        pos += 1
    return pos


def _params(st: _State, s: str, pos: int, base: int) -> tuple[list[float], int]:
    vals: list[float] = []
    pos = _skip_spaces(s, pos)
    while pos < len(s) and s[pos] == ",":
        end = pos + 1
        while end < len(s) and s[end] in NUM_CHARS:
            end += 1
        txt = s[pos + 1:end]
        try:
            v = float(txt) if txt else 0.0
            if not txt:
                st.warn("empty parameter read as 0", base + pos)
        except ValueError:
            v = 0.0
            st.warn(f"cannot read number {txt!r} (read as 0)", base + pos + 1, "error")
        if len(vals) >= MAX_PARAMS:
            st.warn("more than 5 parameters (extra ones ignored)", base + pos)
        else:
            vals.append(v)
        pos = _skip_spaces(s, end)
    return vals, pos


def _bracket(st: _State, s: str, pos: int, base: int) -> tuple[str, int, int]:
    """``s[pos] == '('`` → (inner text, inner offset, position after ')')."""
    depth = 0
    for k in range(pos, len(s)):
        if s[k] == "(":
            depth += 1
        elif s[k] == ")":
            depth -= 1
            if depth == 0:
                return s[pos + 1:k], base + pos + 1, k + 1
    st.warn("unbalanced '(' — the rest of the formula is taken as the group", base + pos, "error")
    return s[pos + 1:], base + pos + 1, len(s)


def _p(vals: list[float], k: int, default=None):
    """Parameter k (1-based) if given, else ``default``."""
    return vals[k - 1] if len(vals) >= k else default


def _p0(vals: list[float], k: int) -> float:
    """Parameter k (1-based) with Pascal's zero fill."""
    return vals[k - 1] if len(vals) >= k else 0.0


def _parse(st: _State, s: str, acc: Node, base: int) -> Node:
    pos = 0
    while pos < len(s):
        c = s[pos]
        if c == "(":
            inner, ibase, pos = _bracket(st, s, pos, base)
            acc = _parse(st, inner, acc, ibase)
        elif c == "[":
            end = pos + 1
            while end < len(s) and s[end].isdigit():
                end += 1
            if end == pos + 1:
                st.warn("'[' without a history number", base + pos, "error")
            else:
                acc = Hist(int(s[pos + 1:end]))
            if end < len(s) and s[end] == "]":
                end += 1
            elif end < len(s):
                st.warn("expected ']' after history number", base + end)
                end += 1  # v407 deletes one character after the digits
            pos = end
        elif c in FUNC1:
            vals, pos = _params(st, s, pos + 1, base)
            if pos < len(s) and s[pos] == "(":
                inner, ibase, pos = _bracket(st, s, pos, base)
                acc = _parse(st, inner, acc, ibase)
            acc = _func1(st, c, vals, acc, base + pos)
        elif c in FUNC2:
            vals, pos = _params(st, s, pos + 1, base)
            right = _parse(st, s[pos:], Undefined(), base + pos)
            return _func2(c, vals, acc, right)
        elif c in FUNC2B:
            vals, pos = _params(st, s, pos + 1, base)
            if pos >= len(s) or s[pos] != "(":
                st.warn(f"'{c}' expects two bracket groups, e.g. {c}([1])([2])", base + pos, "error")
                return acc
            inner, ibase, pos = _bracket(st, s, pos, base)
            g1 = _parse(st, inner, acc, ibase)
            g2: Node = Undefined()
            if pos < len(s) and s[pos] == "(":
                inner, ibase, pos = _bracket(st, s, pos, base)
                g2 = _parse(st, inner, Undefined(), ibase)
            if c in "Gg":
                return call("from_spectrum", g1, g2)
            st.warn(f"'{c}' (phase/group velocity) is disabled in WavePlot v407 and has no effect", base, "error")
            return Call("unsupported", (g1, g2), (("name", c),))
        elif c == "&":
            pos += 1
            if pos < len(s) and s[pos] in AMP:
                f = s[pos]
                vals, pos = _params(st, s, pos + 1, base)
                if pos < len(s) and s[pos] == "(":
                    inner, ibase, pos = _bracket(st, s, pos, base)
                    acc = _parse(st, inner, acc, ibase)
                acc = _func1b(st, f, vals, acc, base + pos)
            else:
                st.warn(f"unknown extended function '&{s[pos] if pos < len(s) else ''}'", base + pos, "error")
                pos += 1
        elif c == "$":
            pos += 1
            if pos < len(s) and s[pos] in DOLLAR:
                f = s[pos]
                vals, pos = _params(st, s, pos + 1, base)
                right = _parse(st, s[pos:], Undefined(), base + pos)
                return _func2c(st, f, vals, acc, right, base + pos)
            st.warn(f"unknown function '${s[pos] if pos < len(s) else ''}'", base + pos, "error")
            pos += 1
        elif c == " ":
            pos += 1
        else:
            st.warn(f"unknown character {c!r} ignored", base + pos)
            pos += 1
    return acc


_SPEC = {"P": ("phase", "none"), "p": ("phase", "stop"), "F": ("amp", "none"), "f": ("amp", "stop"),
         "T": ("inverse", "none"), "t": ("inverse", "stop")}


def _spec_call(acc: Node, kind: str, filt: str, vals: list[float], units: str) -> Call:
    return call("spectrum", acc, kind=kind, filter=filt, f1=_p(vals, 1), f2=_p(vals, 2), t1=_p(vals, 3),
                t2=_p(vals, 4), taper=_p(vals, 5), units=units if units != "percent" else None)


def _window_call(acc: Node, vals: list[float]) -> Call:
    units = "absolute" if (len(vals) >= 1 and pas_trunc(vals[0]) == 1) else None
    return call("window", acc, units=units, t1=_p(vals, 2), t2=_p(vals, 3), taper=_p(vals, 4),
                decimate=int(vals[4]) if len(vals) >= 5 else None)


def _func1(st: _State, c: str, vals: list[float], acc: Node, col: int) -> Node:
    u = c.upper()
    if u == "X":
        return call("reverse", acc)
    if u == "M":
        return call("scale", acc, c=_p(vals, 1, 1.0))
    if c == "#":
        return call("offset", acc, c=_p(vals, 1, 0.0))
    if u == "I":
        return call("integrate", acc)
    if u == "D":
        return call("differentiate", acc)
    if c == "s":
        return call("shift_time", acc, tau=_p(vals, 1, 0.0))
    if c == "S":
        return call("shift_samples", acc, n=pas_trunc(_p(vals, 1, 0.0)))
    if u == "E":
        return call("power", acc, p=_p(vals, 1, 0.5))
    if u == "J":
        return _window_call(acc, vals)
    if u == "L":
        hi, lo = _p0(vals, 1), _p0(vals, 2)
        if hi <= lo:
            st.warn("L needs hi > lo (v407 would loop forever)", col, "error")
        return call("unwrap", acc, hi=hi, lo=lo)
    if c in _SPEC:
        kind, filt = _SPEC[c]
        return _spec_call(acc, kind, filt, vals, "percent")
    return acc


def _func2(c: str, vals: list[float], left: Node, right: Node) -> Node:
    if c == "+":
        return call("add", left, right)
    if c == "-":
        return call("sub", left, right)
    if c == "*":
        return call("mul", left, right)
    if c == "/":
        return call("div", left, right)
    if c in "Cc":
        return call("rotate", left, right, theta=_p0(vals, 1))
    if c == "R":
        return call("hypot", left, right)
    return call("angle", left, right)  # 'r'


def _func1b(st: _State, f: str, vals: list[float], acc: Node, col: int) -> Node:
    if f in "xX":
        mirror = pas_trunc(vals[2]) if len(vals) >= 3 else None
        return call("tp_window", acc, start=_p(vals, 1), window=_p(vals, 2),
                    mirror=min(mirror, 1) if mirror is not None else None)
    if f in "Dd":
        return call("decimate", acc, q=max(pas_trunc(_p(vals, 1, 1.0)), 1))
    if f in "Ww":
        return _window_call(acc, vals)
    if f == "F":
        return _spec_call(acc, "inverse", "pass", vals, "absolute")
    if f == "P":
        return _spec_call(acc, "amp", "pass", vals, "absolute")
    if f == "p":
        return _spec_call(acc, "phase", "pass", vals, "absolute")
    if f == "f":
        if len(vals) < 2:
            st.warn("&f without both frequencies: v407 uses 0 Hz, which removes the whole signal", col)
        return call("bandpass", acc, f_lo=_p(vals, 1, 0.0), f_hi=_p(vals, 2, 0.0),
                    sections=pas_trunc(_p(vals, 3, 4.0)), zero_phase=pas_trunc(_p(vals, 4, 1.0)))
    if f == "q":
        return call("attenuate", acc, Q=_p(vals, 1, 1e6), c=_p(vals, 2, 6000.0), L=_p(vals, 3, 1.0))
    if f == "Q":
        return call("attenuated_magnitude", Q=_p(vals, 1, 1e6), c=_p(vals, 2, 6000.0), L=_p(vals, 3, 1.0),
                    h=pas_trunc(vals[3]) if len(vals) >= 4 else None)
    if f in "Vv":
        return call("vector_magnitude", h=pas_trunc(vals[0]) if len(vals) >= 1 else None)
    if f in "Rr":
        h = pas_trunc(vals[1]) if len(vals) >= 2 else None
        return call("rotate_to_source", component=pas_trunc(_p(vals, 1, 0.0)), h=h or None,
                    i=_p(vals, 3, 0.0), j=_p(vals, 4, 0.0), k=_p(vals, 5, 0.0))
    if f in "tT":
        kind = pas_trunc(_p(vals, 1, 0.0))
        if kind > 2:
            kind = 0
        return call("tau_p" if f == "t" else "tau_p_damped", acc, kind=kind, ds=_p(vals, 2),
                    window=_p(vals, 3, 10.0), output=pas_trunc(_p(vals, 4, 0.0)))
    return acc


def _func2c(st: _State, f: str, vals: list[float], left: Node, right: Node, col: int) -> Node:
    if f == "*":
        return call("convolve", left, right)
    if f == "g":
        return call("reflect_g", left, right, p=int(pas_round(_p0(vals, 1))), s=int(pas_round(_p0(vals, 2))))
    if f == "h":
        return call("reflect_h", left, right, p=int(pas_round(_p0(vals, 1))), s=int(pas_round(_p0(vals, 2))))
    if f == "G":
        return call("reflect_G", left, right, p=int(pas_round(_p0(vals, 1))))
    st.warn(f"'${f}' does nothing in v407 (the rest of the formula is consumed)", col)
    return left
