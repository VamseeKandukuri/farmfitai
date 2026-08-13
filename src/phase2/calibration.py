"""
FarmFit AI - Phase 2 calibration.

Implements the addendum's requirement explicitly: the base estimator is fitted
ONCE on M, frozen, and never refitted. Only the calibration MAPPING is fitted,
and it is cross-fitted within P.

Why this is not `CalibratedClassifierCV(FrozenEstimator(base), cv=5)`:
that construction fits an internal calibrator per CV fold and then, at predict
time, AVERAGES the predictions of all five calibrators. Every one of those five
calibrators has seen four fifths of P, so the averaged prediction for a P row is
influenced by calibrators that were fitted on that row. It is an ensemble, not a
cross-fitted estimate, and it cannot produce the held-out P probabilities the
threshold sweep requires. The loop below keeps each P fold's prediction strictly
out-of-sample with respect to the calibration mapping.
"""
import hashlib
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import p2config as K


# ------------------------------------------------------------------ state hash
def estimator_state_hash(est):
    """SHA-256 of the fitted estimator's serialised state.

    Used to prove the base estimator's learned parameters did not change while
    the calibration mapping was being fitted.
    """
    return hashlib.sha256(pickle.dumps(est)).hexdigest()


# ------------------------------------------------------------------ mappings
class MulticlassCalibrationMapping:
    """One-vs-rest probability mapping over a frozen base model's outputs.

    method='sigmoid'  : Platt scaling — logistic regression on each class's
                        raw score, fitted on that class's binary indicator.
    method='isotonic' : isotonic regression per class, clipped out of range.

    Rows are renormalised to sum to one after mapping. A row whose mapped
    values all collapse to zero falls back to the uniform distribution, which
    is the maximum-entropy answer and routes the case to expert review rather
    than to a spurious confident recommendation.
    """

    def __init__(self, method, n_classes=K.N_CLASSES):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"unsupported calibration mapping: {method}")
        self.method = method
        self.n_classes = n_classes
        self.models_ = {}

    def fit(self, proba, y):
        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y)
        for k in range(self.n_classes):
            target = (y == k).astype(int)
            score = proba[:, k].reshape(-1, 1)
            if target.sum() == 0 or target.sum() == len(target):
                # class absent (or the only class) in this fitting fold:
                # fall back to the identity mapping for that class
                self.models_[k] = None
                continue
            if self.method == "sigmoid":
                m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
                m.fit(score, target)
            else:
                m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                m.fit(score.ravel(), target)
            self.models_[k] = m
        return self

    def transform(self, proba):
        proba = np.asarray(proba, dtype=float)
        out = np.empty_like(proba)
        for k in range(self.n_classes):
            m = self.models_.get(k)
            if m is None:
                out[:, k] = proba[:, k]
            elif self.method == "sigmoid":
                out[:, k] = m.predict_proba(proba[:, k].reshape(-1, 1))[:, 1]
            else:
                out[:, k] = m.predict(proba[:, k])
        out = np.clip(out, 0.0, None)
        s = out.sum(axis=1, keepdims=True)
        flat = (s.ravel() <= 0)
        out = np.divide(out, np.where(s <= 0, 1.0, s))
        if flat.any():
            out[flat, :] = 1.0 / self.n_classes
        return out


def apply_treatment(mapping, raw_proba):
    """`none` is the identity treatment on the raw base probabilities."""
    if mapping is None:
        return np.asarray(raw_proba, dtype=float)
    return mapping.transform(raw_proba)


# ------------------------------------------------------------------ cross-fitting
def crossfit_P(frozen_base, X_P, y_P, repeat, fold):
    """Cross-fitted calibrated probabilities for every P observation.

    Returns {treatment: (n_P, n_classes) array} where each row was produced by
    a calibration mapping fitted WITHOUT that row. The base estimator is only
    ever called for prediction.

    Also returns the per-fold provenance record used by the manual
    cross-fitting proof test: for each P row, which crossfit fold it was held
    out in, and which rows fitted the mapping that predicted it.
    """
    # X_P is kept as-is (a DataFrame) so the frozen estimator sees the same
    # feature names it was fitted with; only the label vector is coerced.
    y_P = np.asarray(y_P)
    n = len(y_P)

    raw_P = frozen_base.predict_proba(X_P)

    out = {"none": raw_P.copy()}
    for method in ("sigmoid", "isotonic"):
        out[method] = np.full_like(raw_P, np.nan)

    skf = StratifiedKFold(n_splits=K.P_CROSSFIT_FOLDS, shuffle=True,
                          random_state=K.seed_crossfit(repeat, fold))
    provenance = np.full(n, -1, dtype=int)
    fit_index_sets = []

    for cf, (fit_rows, held_rows) in enumerate(skf.split(X_P, y_P)):
        provenance[held_rows] = cf
        fit_index_sets.append(np.sort(fit_rows))
        for method in ("sigmoid", "isotonic"):
            mapping = MulticlassCalibrationMapping(method).fit(
                raw_P[fit_rows], y_P[fit_rows])
            out[method][held_rows] = mapping.transform(raw_P[held_rows])

    for method in ("sigmoid", "isotonic"):
        if np.isnan(out[method]).any():
            raise AssertionError(
                f"cross-fitted {method} probabilities incomplete: "
                f"{int(np.isnan(out[method]).any(axis=1).sum())} rows unfilled")

    return out, {"crossfit_fold_of_row": provenance,
                 "fit_index_sets": fit_index_sets}


def select_calibration(crossfitted, y_P, metrics_mod):
    """Lowest cross-fitted P log loss; ties within CAL_TIE_EPS broken by
    CAL_CANDIDATES order (none -> sigmoid -> isotonic)."""
    rows = []
    for method in K.CAL_CANDIDATES:
        pr = crossfitted[method]
        rows.append({
            "calibration": method,
            "crossfit_log_loss": metrics_mod.log_loss_fixed(y_P, pr),
            "crossfit_ece": metrics_mod.ece(y_P, pr),
            "crossfit_brier": metrics_mod.brier(y_P, pr),
            "crossfit_accuracy": metrics_mod.accuracy(y_P, pr),
        })
    best = min(r["crossfit_log_loss"] for r in rows)
    tied = [r["calibration"] for r in rows
            if r["crossfit_log_loss"] - best < K.CAL_TIE_EPS]
    selected = next(m for m in K.CAL_CANDIDATES if m in tied)
    return selected, rows, tied


def fit_final_mapping(method, raw_proba, y):
    """Refit the selected mapping on ALL of the development probabilities.

    See addendum section 8: the difference between this mapping and the
    cross-fitted ones is a calibration-threshold distribution mismatch whose
    direction of effect is not determined by this design.
    """
    if method == "none":
        return None
    return MulticlassCalibrationMapping(method).fit(raw_proba, y)
