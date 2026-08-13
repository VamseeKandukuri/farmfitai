"""
FarmFit AI - Phase 2 metrics, implemented exactly as frozen in
PHASE2B_IMPLEMENTATION_ADDENDUM.md sections 1-5.

Every function here has a hand-calculated unit test in tests/test_phase2.py.
"""
import numpy as np
from sklearn.metrics import f1_score

from . import p2config as K

LABELS = np.arange(K.N_CLASSES)


# ----------------------------------------------------------------- ordering
def rank_classes(proba):
    """Stable descending class ranking.

    Ties are broken by the frozen class order (ascending class index), because
    argsort on the negated probabilities with kind='stable' preserves the
    original ordering among equal values.
    """
    return np.argsort(-np.asarray(proba, dtype=float), axis=-1, kind="stable")


def top_k(proba, k=3):
    """Returns (indices, probabilities) for the top k classes, stable order."""
    proba = np.atleast_2d(np.asarray(proba, dtype=float))
    order = rank_classes(proba)[:, :k]
    rows = np.arange(len(proba))[:, None]
    return order, proba[rows, order]


# ----------------------------------------------------------------- metrics
def accuracy(y_true, proba):
    pred = rank_classes(proba)[:, 0]
    return float(np.mean(pred == np.asarray(y_true)))


def macro_f1(y_true, proba):
    """Complete class ordering, zero_division=0 (addendum section 4)."""
    pred = rank_classes(proba)[:, 0]
    return float(f1_score(y_true, pred, average="macro",
                          labels=LABELS, zero_division=0))


def top_k_accuracy(y_true, proba, k=3):
    order = rank_classes(proba)[:, :k]
    y_true = np.asarray(y_true)[:, None]
    return float(np.mean((order == y_true).any(axis=1)))


def p_true(y_true, proba):
    """Probability assigned to the true class, per observation."""
    proba = np.asarray(proba, dtype=float)
    return proba[np.arange(len(proba)), np.asarray(y_true)]


def log_loss_fixed(y_true, proba):
    """Log loss over the complete fixed 22-class ordering (addendum section 3)."""
    pt = np.clip(p_true(y_true, proba), K.LOGLOSS_EPS, 1.0)
    return float(-np.mean(np.log(pt)))


def brier_contributions(y_true, proba):
    """Per-observation sum over ALL classes of (p - onehot)^2.

    Not divided by the number of classes (addendum section 2).
    """
    proba = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(proba)), np.asarray(y_true)] = 1.0
    return ((proba - onehot) ** 2).sum(axis=1)


def brier(y_true, proba):
    return float(np.mean(brier_contributions(y_true, proba)))


def ece(y_true, proba, n_bins=K.ECE_BINS):
    """Expected calibration error, addendum section 1.

    Ten equal-width bins over [0,1]; bin index = min(floor(conf*n), n-1) so the
    final bin is closed on the right and includes confidence exactly 1.0.
    Each bin contributes (n_bin/N) * |mean_confidence - accuracy|.
    """
    proba = np.asarray(proba, dtype=float)
    order = rank_classes(proba)
    conf = proba[np.arange(len(proba)), order[:, 0]]
    correct = (order[:, 0] == np.asarray(y_true)).astype(float)

    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = len(conf)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        out += (n_b / total) * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def normalised_entropy(proba):
    """Shannon entropy of each row scaled to [0, 1] by log(n_classes)."""
    p = np.clip(np.atleast_2d(np.asarray(proba, dtype=float)), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def all_metrics(y_true, proba):
    """The standard metric block used by both tracks."""
    return {
        "accuracy": accuracy(y_true, proba),
        "macro_f1": macro_f1(y_true, proba),
        "top3_accuracy": top_k_accuracy(y_true, proba, 3),
        "log_loss": log_loss_fixed(y_true, proba),
        "brier": brier(y_true, proba),
        "ece": ece(y_true, proba),
    }


def confidence_block(proba):
    """Top-3 classes, probabilities, margin, cumulative mass and entropy."""
    idx, prob = top_k(proba, 3)
    return {
        "top1_class": idx[:, 0], "top2_class": idx[:, 1], "top3_class": idx[:, 2],
        "top1_prob": prob[:, 0], "top2_prob": prob[:, 1], "top3_prob": prob[:, 2],
        "margin_12": prob[:, 0] - prob[:, 1],
        "top3_cum_prob": prob.sum(axis=1),
        "entropy": normalised_entropy(proba),
    }


# ----------------------------------------------------------------- pooled from rows
def metrics_from_rows(y_true, y_pred, p_true_vals, brier_contribs,
                      correct, in_top3, top1_prob, n_bins=K.ECE_BINS):
    """Recompute the full metric block from stored compact prediction rows.

    Fold-level metrics are NOT averaged to obtain a repeat-level value. Macro F1
    and ECE are not linear in the observations, so the mean of five fold values
    is not the value computed over the pooled 2,200 rows. Every repeat-level
    number is therefore recomputed here from that repeat's complete prediction
    set, which is also what makes the summaries exactly reproducible from the
    published prediction files.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pt = np.clip(np.asarray(p_true_vals, dtype=float), K.LOGLOSS_EPS, 1.0)
    conf = np.asarray(top1_prob, dtype=float)
    corr = np.asarray(correct, dtype=float)

    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = len(conf)
    ece_val = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if n_b:
            ece_val += (n_b / total) * abs(conf[m].mean() - corr[m].mean())

    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=LABELS, zero_division=0)),
        "top3_accuracy": float(np.mean(np.asarray(in_top3, dtype=float))),
        "log_loss": float(-np.mean(np.log(pt))),
        "brier": float(np.mean(np.asarray(brier_contribs, dtype=float))),
        "ece": float(ece_val),
    }


def metrics_from_frame(df):
    """Convenience wrapper over a compact prediction DataFrame."""
    return metrics_from_rows(df["y_true"], df["y_pred"], df["p_true"],
                             df["brier_contribution"], df["correct"],
                             df["in_top3"], df["top1_prob"])
