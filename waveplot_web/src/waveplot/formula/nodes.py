"""Formula syntax tree shared by the legacy v407 parser, the modern expression parser and the
pipeline editor. Nodes are immutable and hashable (results are cached by node)."""

from __future__ import annotations

from dataclasses import dataclass, field


class Node:
    """Base class."""


@dataclass(frozen=True)
class Hist(Node):
    number: int


@dataclass(frozen=True)
class Self(Node):
    """The history of the trace slot (v407: the accumulator starts with the slot's history)."""


@dataclass(frozen=True)
class Num(Node):
    value: float


@dataclass(frozen=True)
class Undefined(Node):
    """v407 right operand without any ``[n]`` (an uninitialised buffer)."""

    reason: str = "no history loaded"


@dataclass(frozen=True)
class Call(Node):
    op: str
    args: tuple[Node, ...] = ()
    params: tuple[tuple[str, object], ...] = field(default_factory=tuple)

    def param(self, name: str, default=None):
        for k, v in self.params:
            if k == name:
                return v
        return default

    @property
    def param_dict(self) -> dict[str, object]:
        return dict(self.params)


def call(op: str, *args: Node, **params) -> Call:
    return Call(op, tuple(args), tuple((k, v) for k, v in params.items() if v is not None))


def walk(node: Node):
    yield node
    if isinstance(node, Call):
        for a in node.args:
            yield from walk(a)


def histories_used(node: Node) -> set[int]:
    out = {n.number for n in walk(node) if isinstance(n, Hist)}
    for n in walk(node):
        if isinstance(n, Call) and n.op in ("vector_magnitude", "attenuated_magnitude", "rotate_to_source"):
            h = n.param("h")
            if isinstance(h, int | float) and h:
                out |= {int(h), int(h) + 1, int(h) + 2}
    return out
