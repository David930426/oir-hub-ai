"""Hand-checkable sanity tests for metrics.py. Run: python experiment/test_metrics.py"""

import math

from metrics import (bootstrap_ci, confusion, correct_vector, holm, kappa, macro_f1,
                     mcnemar_exact, paired_bootstrap_diff, target_accuracy)

U, N, R = "UPDATE", "NEW", "REDUNDANT"


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_macro_f1():
    assert close(macro_f1([U, N, R], [U, N, R]), 1.0)
    # Majority baseline on 2 NEW / 1 UPDATE / 1 REDUNDANT: NEW has P=0.5 R=1 F1=2/3,
    # the other two are 0, so macro-F1 = (2/3)/3.
    assert close(macro_f1([N, N, U, R], [N, N, N, N]), (2 / 3) / 3)


def test_confusion():
    m = confusion([U, U, N], [U, N, N])
    assert m.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_target_accuracy():
    r = target_accuracy([U, U, U, N], ["a", "b", "c", None], [U, U, N, N], ["a", "x", None, None])
    assert r["n_gold_update"] == 3
    assert close(r["conditional"], 1 / 2)  # 2 called UPDATE, 1 right target
    assert close(r["joint"], 1 / 3)
    # either edition of the predecessor is an acceptable target
    r = target_accuracy([U], ["zh|en"], [U], ["en"])
    assert close(r["joint"], 1.0)


def test_correct_vector():
    v = correct_vector([U, N], ["a", None], [U, N], ["wrong", "whatever"])
    assert v.tolist() == [0, 1]
    assert correct_vector([U], ["a"], [U], ["wrong"], with_target=False).tolist() == [1]


def test_mcnemar():
    # 10 discordant cases all favouring B: p = 2 * 0.5**10
    r = mcnemar_exact([0] * 10 + [1] * 5, [1] * 10 + [1] * 5)
    assert r["a_only"] == 0 and r["b_only"] == 10
    assert close(r["p"], 2 * 0.5 ** 10)
    assert mcnemar_exact([1, 0], [1, 0])["p"] == 1.0


def test_holm():
    # sorted p: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    assert [round(x, 10) for x in holm([0.04, 0.01, 0.02])] == [0.04, 0.03, 0.04]


def test_bootstrap():
    lo, hi = bootstrap_ci([U, N, R] * 10, [U, N, R] * 10, n=200)
    assert lo == hi == 1.0
    d = paired_bootstrap_diff([U, N, R] * 10, [N] * 30, [U, N, R] * 10, n=200)
    assert d["diff"] > 0 and d["ci"][0] > 0


def test_kappa():
    assert close(kappa([U, N, R, N], [U, N, R, N]), 1.0)
    assert math.isfinite(kappa([U, N, R, N], [N, N, R, U]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
