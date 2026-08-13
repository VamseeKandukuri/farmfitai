"""
FarmFit AI - Phase 2 OOD reference and advisory policy.

OOD terminology, fixed by the Phase 2A.1 freeze: the quantity reported is the
"held-out reference flag rate under the dataset-as-reference assumption". The
dataset carries no ground-truth in/out-of-distribution labels, so no flag can be
known to be false and no rate has a target value. False-positive and false-flag
terminology is prohibited.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import p2config as K


class OODReference:
    """Class-conditional Mahalanobis reference fitted on M only.

    Distance is measured to the model's PREDICTED class, because the true class
    is unavailable at deployment. That couples the flag rate to classification
    correctness: a misclassified row is measured against the wrong reference.
    This is a property of the deployable design and is reported, not hidden.
    """

    def __init__(self):
        self.mu, self.prec, self.cond = {}, {}, {}
        self.lo = self.hi = self.cutoff = None
        self.n_features = None

    def fit(self, X_M, y_M):
        Xv = np.asarray(X_M, dtype=float)
        y_M = np.asarray(y_M)
        self.n_features = Xv.shape[1]
        for c in np.unique(y_M):
            Z = Xv[y_M == c]
            cov = np.cov(Z, rowvar=False) + K.OOD_COV_RIDGE * np.eye(Z.shape[1])
            self.mu[int(c)] = Z.mean(axis=0)
            self.prec[int(c)] = np.linalg.inv(cov)
            self.cond[int(c)] = float(np.linalg.cond(cov))
        rng = Xv.max(axis=0) - Xv.min(axis=0)
        self.lo = Xv.min(axis=0) - K.OOD_RANGE_MARGIN * rng
        self.hi = Xv.max(axis=0) + K.OOD_RANGE_MARGIN * rng
        self.cutoff = float(stats.chi2.ppf(K.OOD_CHI2_QUANTILE, self.n_features))
        return self

    def score(self, X, predicted_class):
        """Squared Mahalanobis distance to the PREDICTED class reference."""
        Xv = np.asarray(X, dtype=float)
        pred = np.asarray(predicted_class)
        d = np.empty(len(Xv))
        for i in range(len(Xv)):
            c = int(pred[i])
            if c not in self.mu:               # class unseen in M (cannot occur
                d[i] = np.inf                  # under the stratified design)
                continue
            diff = Xv[i] - self.mu[c]
            d[i] = float(diff @ self.prec[c] @ diff)
        multivariate = d > self.cutoff
        out_of_range = ((Xv < self.lo) | (Xv > self.hi)).any(axis=1)
        return pd.DataFrame({
            "mahalanobis_predicted_class": d,
            "flag_multivariate": multivariate,
            "flag_univariate_range": out_of_range,
            "flag_union": multivariate | out_of_range,
        })

    def diagnostics(self):
        vals = list(self.cond.values())
        return {"chi2_cutoff": self.cutoff,
                "n_features": int(self.n_features),
                "cov_condition_min": float(np.min(vals)),
                "cov_condition_median": float(np.median(vals)),
                "cov_condition_max": float(np.max(vals)),
                "per_class_condition": {int(k): v for k, v in self.cond.items()}}


# ------------------------------------------------------------------ thresholds
def sweep_thresholds(conf, correct, excluded):
    """Coverage / selective-accuracy sweep over the frozen grid.

    `excluded` is the UNION of the OOD flag and the entropy flag. Under the
    frozen expert-first precedence those observations can never reach STRONG,
    so counting them toward a cell's support, coverage or selective accuracy
    would optimise a category that does not exist as measured. Every cell here
    is therefore evaluated over exactly the observations that could actually be
    routed to STRONG, and the MIN_STRONG_SUPPORT floor refers to that actual
    category. See PHASE2B_PRE_FULL_CLARIFICATION.md.
    """
    eligible = ~np.asarray(excluded, dtype=bool)
    p1 = np.asarray(conf["top1_prob"])[eligible]
    mg = np.asarray(conf["margin_12"])[eligible]
    ok = np.asarray(correct)[eligible]
    n_total = len(conf)

    rows = []
    for p in K.P_STAR_GRID:
        for m in K.M_STAR_GRID:
            sel = (p1 >= p) & (mg >= m)
            n = int(sel.sum())
            if n < K.MIN_STRONG_SUPPORT:
                continue
            rows.append({"p_star": round(float(p), 4), "m_star": round(float(m), 4),
                         "n_strong": n, "coverage": n / n_total,
                         "selective_accuracy": float(ok[sel].mean())})
    return pd.DataFrame(rows, columns=["p_star", "m_star", "n_strong",
                                       "coverage", "selective_accuracy"])


def pick_operating_point(sweep, target):
    """Deterministic tie-breaking sequence, addendum section 7.3:
    max coverage -> max selective accuracy -> max p_star -> max m_star.

    Frozen edge case: if no cell clears the support floor once expert-review
    exclusions are applied, no Strong category exists for this fold and
    operating point. Thresholds are null, target_met is false, and the STRONG
    category stays empty. The floor is never lowered.
    """
    if not len(sweep):
        return {"p_star": None, "m_star": None, "target": float(target),
                "target_met": False, "strong_available": False,
                "dev_coverage": 0.0, "dev_selective_accuracy": None,
                "dev_n_strong": 0}
    eligible = sweep[sweep.selective_accuracy >= target]
    target_met = bool(len(eligible))
    pool = eligible if target_met else sweep
    ranked = pool.sort_values(
        ["coverage", "selective_accuracy", "p_star", "m_star"],
        ascending=[False, False, False, False], kind="stable")
    pick = ranked.iloc[0]
    return {"p_star": float(pick.p_star), "m_star": float(pick.m_star),
            "target": float(target), "target_met": target_met,
            "strong_available": True,
            "dev_coverage": float(pick.coverage),
            "dev_selective_accuracy": float(pick.selective_accuracy),
            "dev_n_strong": int(pick.n_strong)}


def pick_entropy_threshold(entropy, correct):
    """Smallest E whose tail is both large enough and genuinely risky.

    Fitted FIRST, before the P*/M* sweep, because the sweep must know which
    observations the entropy rule will divert to expert review.
    """
    entropy = np.asarray(entropy)
    correct = np.asarray(correct)
    for q in K.ENTROPY_QUANTILE_GRID:
        e = float(np.quantile(entropy, q))
        tail = entropy >= e
        if tail.sum() >= K.MIN_STRONG_SUPPORT and \
                correct[tail].mean() < K.ENTROPY_RISK_ACCURACY:
            return e, True
    return float(np.quantile(entropy, K.ENTROPY_FALLBACK_QUANTILE)), False


def apply_policy(conf, ood_flag, th):
    """Assign advisory categories under the FROZEN precedence order.

        1. EXPERT_REVIEW  if OOD is true OR entropy >= E*
        2. STRONG         if p1 >= P* AND margin >= M*
        3. ALTERNATIVES   otherwise

    Expert review is evaluated FIRST, so an entropy-triggered observation can
    never remain STRONG even when it clears both probability thresholds. When
    no supported Strong category exists (p_star is None), nothing is STRONG.

    Returns (category, flag_entropy, expert_trigger) where expert_trigger is one
    of 'ood_only', 'entropy_only', 'both', 'none', mutually exclusive by
    construction.
    """
    ood_flag = np.asarray(ood_flag, dtype=bool)
    p1 = np.asarray(conf["top1_prob"])
    mg = np.asarray(conf["margin_12"])
    ent = np.asarray(conf["entropy"])

    flag_entropy = ent >= th["entropy_star"]
    expert = ood_flag | flag_entropy                      # step 1, evaluated first

    if th.get("p_star") is None or th.get("m_star") is None:
        strong = np.zeros(len(p1), dtype=bool)
    else:
        strong = (p1 >= th["p_star"]) & (mg >= th["m_star"]) & (~expert)

    category = np.where(expert, "EXPERT_REVIEW",
                        np.where(strong, "STRONG", "ALTERNATIVES"))

    trigger = np.full(len(category), "none", dtype=object)
    trigger[ood_flag & flag_entropy] = "both"
    trigger[ood_flag & ~flag_entropy] = "ood_only"
    trigger[~ood_flag & flag_entropy] = "entropy_only"

    return category, flag_entropy, trigger


def realised_strong_stats(conf, ood_flag, correct, th):
    """Recompute actual STRONG support, coverage and selective accuracy by
    applying the FINAL policy. Used to assert that the stored threshold-selection
    record matches what the policy really produces."""
    cat, _, _ = apply_policy(conf, ood_flag, th)
    sel = cat == "STRONG"
    n = int(sel.sum())
    return {"n_strong": n, "coverage": n / len(cat),
            "selective_accuracy": float(np.asarray(correct)[sel].mean())
            if n else None}
