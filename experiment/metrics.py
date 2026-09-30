"""Evaluation metrics and significance tests for the supersession experiment.

Everything takes plain lists so it can be checked by hand on toy data (see test_metrics.py)
before a single real label exists.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["UPDATE", "NEW", "REDUNDANT"]


def macro_f1(y_true, y_pred):
    # labels= is fixed so a class the method never predicts still counts as F1 = 0,
    # instead of silently dropping out of the average and inflating the score.
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def per_class(y_true, y_pred):
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(n[i])}
            for i, c in enumerate(LABELS)}


def confusion(y_true, y_pred):
    """Rows are gold labels, columns are predictions, both in LABELS order."""
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def target_hit(pred, gold):
    """gold may be one doc_id or several acceptable ones ("oir-10|oir-11", or a set): a
    successor can legitimately replace either the zh or the en edition of its predecessor."""
    if isinstance(gold, str):
        gold = set(gold.split("|"))
    return pred is not None and pred in (gold or set())


def target_accuracy(gold_labels, gold_targets, pred_labels, pred_targets):
    """For gold-UPDATE cases only.

    conditional: of the cases the method correctly called UPDATE, how often it named the
                 right superseded document.
    joint:       of all gold-UPDATE cases, how often it got both the label and the target.
    """
    idx = [i for i, g in enumerate(gold_labels) if g == "UPDATE"]
    hit = [i for i in idx if pred_labels[i] == "UPDATE"]
    right = [i for i in hit if target_hit(pred_targets[i], gold_targets[i])]
    return {
        "n_gold_update": len(idx),
        "conditional": len(right) / len(hit) if hit else float("nan"),
        "joint": len(right) / len(idx) if idx else float("nan"),
    }


def correct_vector(gold_labels, gold_targets, pred_labels, pred_targets, with_target=True):
    """Per-case 0/1 correctness, the input McNemar needs. With with_target, an UPDATE is
    only correct if it also points at the right document - that is what the office needs."""
    out = []
    for gl, gt, pl, pt in zip(gold_labels, gold_targets, pred_labels, pred_targets):
        ok = gl == pl and (not with_target or gl != "UPDATE" or target_hit(pt, gt))
        out.append(int(ok))
    return np.array(out)


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) McNemar test on paired correctness. Only the discordant cases carry
    information: b = A right & B wrong, c = A wrong & B right. Exact rather than chi-square
    because with ~100 test cases b + c is often under 25."""
    a, b_ = np.asarray(correct_a), np.asarray(correct_b)
    b = int(((a == 1) & (b_ == 0)).sum())
    c = int(((a == 0) & (b_ == 1)).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only": b, "b_only": c, "p": p}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bootstrap_ci(y_true, y_pred, stat=macro_f1, n=10_000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric, resampling test cases with replacement."""
    rng = np.random.default_rng(seed)
    t, p = np.asarray(y_true), np.asarray(y_pred)
    idx = rng.integers(0, len(t), size=(n, len(t)))
    vals = np.array([stat(t[i], p[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y_true, pred_a, pred_b, stat=macro_f1, n=10_000, seed=0, alpha=0.05):
    """CI for stat(B) - stat(A) on the same resampled cases. McNemar compares accuracy; this is
    the matching test for macro-F1, which is the headline number."""
    rng = np.random.default_rng(seed)
    t, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    idx = rng.integers(0, len(t), size=(n, len(t)))
    d = np.array([stat(t[i], b[i]) - stat(t[i], a[i]) for i in idx])
    lo, hi = np.quantile(d, alpha / 2), np.quantile(d, 1 - alpha / 2)
    return {"diff": stat(t, b) - stat(t, a), "ci": (float(lo), float(hi)),
            "p_le_0": float((d <= 0).mean())}


def kappa(labels_a, labels_b):
    return cohen_kappa_score(labels_a, labels_b, labels=LABELS)
