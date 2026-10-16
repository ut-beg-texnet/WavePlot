from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from waveplot.formula.evaluate import evaluate_text, parse_any
from waveplot.formula.expr import parse_expr
from waveplot.formula.legacy import parse_legacy
from waveplot.formula.nodes import Call, Hist, Self, Undefined
from waveplot.formula.ops import OPS, FormulaError
from waveplot.formula.printer import to_expr


def test_legacy_right_recursion_and_accumulator():
    node, d = parse_legacy("[1]*[2]+[3]")
    assert d == []
    assert to_expr(node) == "h1 * (h2 + h3)"
    node, _ = parse_legacy("I")
    assert node == Call("integrate", (Self(),))
    node, _ = parse_legacy("[2]I&f,1,10")
    assert to_expr(node) == "h2 | integrate | bandpass(f_lo=1, f_hi=10, sections=4, zero_phase=1)"
    node, _ = parse_legacy("I([2])")
    assert to_expr(node) == "h2 | integrate"
    node, _ = parse_legacy("[2]+")
    assert node.args[1] == Undefined()


def test_legacy_parameters_and_diagnostics():
    node, d = parse_legacy("[2]M,2E")
    assert any("cannot read number" in str(x) for x in d)
    node, d = parse_legacy("[2]?I")
    assert any("unknown character" in str(x) for x in d)
    assert to_expr(node) == "h2 | integrate"
    _node, d = parse_legacy("([2]I")
    assert any("unbalanced" in str(x) for x in d)
    node, _ = parse_legacy("[3]J,1,0.5,2,0.1")
    assert node.param_dict == {"units": "absolute", "t1": 0.5, "t2": 2.0, "taper": 0.1}
    node, _ = parse_legacy("&R,1,4,250,50,250")
    assert node.op == "rotate_to_source" and node.param("h") == 4 and node.param("component") == 1


@pytest.mark.parametrize("text", ["h2 | integrate | bandpass(f_lo=1, f_hi=10)", "h1 * (h2 + h3)", "(h1 + h2) / 2",
                                  "rotate(h5, h6, theta=30)", "vector_magnitude(h=4)", "-h1 + 2 * h3",
                                  "(h1 - h2) - h3", "h1 - (h2 - h3)", "(h2 + h3) | integrate",
                                  "h2 | spectrum(kind='phase', filter='stop', f1=5)", "h1 ** 2"])
def test_modern_round_trip(text):
    node = parse_expr(text)
    assert parse_expr(to_expr(node)) == node


def test_modern_errors_and_safety():
    for bad in ["__import__('os')", "h1.real", "lambda: 1", "[1]", "foo(h1)", "integrate", "h1 | rotate",
                "bandpass(h1, nope=3)", "spectrum(h1, kind='bogus')", "open('x')", "h1[0]"]:
        with pytest.raises(FormulaError):
            parse_expr(bad)
    node, syntax, _ = parse_any("[1]+[2]")
    assert syntax == "legacy" and to_expr(node) == "h1 + h2"
    assert parse_any("h1+h2")[1] == "modern"


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=list("[]()0123456789,.-+eE IiDdFfPpTtMmSsXxEe#LlJj*/CcRrGgVv&$WwQqzZ "), max_size=40))
def test_legacy_parser_never_crashes(text):
    node, diags = parse_legacy(text)
    to_expr(node)
    assert all(d.column is None or 0 <= d.column <= len(text) for d in diags)


def test_every_op_is_documented():
    for spec in OPS.values():
        assert spec.title and spec.category


def test_evaluate_against_dataset(ds_a):
    r = evaluate_text(ds_a, "[2]I", mode="v407")
    h2 = ds_a.history_data(2).astype(float)
    dt = ds_a.history_record(2).dt
    np.testing.assert_allclose(r.trace.y, np.concatenate([[0], np.cumsum(h2[:-1]) * dt]))
    assert "integrate_left_rect" in r.provenance.quirks
    r = evaluate_text(ds_a, "self | differentiate", self_ref=3)
    assert r.error is None and r.trace.y_unit == "Pa/s"
    r = evaluate_text(ds_a, "h2 - h2")
    assert np.all(r.trace.y == 0)
    r = evaluate_text(ds_a, "[999]")
    assert r.error and "does not exist" in r.error
    r = evaluate_text(ds_a, "I")
    assert r.error and "own history" in r.error
    r = evaluate_text(ds_a, "[2]F", mode="standard")
    assert r.trace.x_unit == "Hz" and r.trace.scale_hint == "log"
    r = evaluate_text(ds_a, "[2]+", mode="v407")
    assert r.trace is not None and any("uninitialised" in d for d in r.diagnostics)
    r = evaluate_text(ds_a, "[2]+", mode="standard")
    assert r.error


def test_standard_binary_alignment(ds_a):
    r = evaluate_text(ds_a, "h2 + (h3 | decimate(q=2))")
    assert r.trace.n == 4001 and any("interpolated" in d for d in r.diagnostics)
    assert isinstance(parse_expr("h[7]"), Hist)
