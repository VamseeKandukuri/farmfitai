"""
FarmFit AI - Phase 2 frozen constants.

Every value here is fixed by PHASE2B_IMPLEMENTATION_ADDENDUM.md and must not be
edited after the ten-repeat evidence run begins. run_phase2.py records a
SHA-256 of this file in every output so a change is detectable after the fact.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------- paths
DATA_CSV = ROOT / "data" / "Crop_recommendation.csv"
PHASE2_OUT = ROOT / "outputs_phase2"          # published Phase 2 evidence
SMOKE_OUT = ROOT / "smoke_outputs"            # smoke test only, never published
CHECKPOINTS = ROOT / ".phase2_checkpoints"    # working data, not evidence

# Phase 1 outputs are never written to by Phase 2.
PHASE1_OUT = ROOT / "outputs"

# ----------------------------------------------------------------- data contract
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
N_CLASSES = 22

# ----------------------------------------------------------------- resampling
MASTER_SEED = 42
N_REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 3
P_CROSSFIT_FOLDS = 5
MP_TRAIN_FRACTION = 0.75      # T (1760) -> M (1320) / P (440)

# exact expected partition sizes, asserted every fold
EXPECT_H = 440
EXPECT_T = 1760
EXPECT_M = 1320
EXPECT_P = 440
EXPECT_PER_CLASS = {"H": 20, "T": 80, "M": 60, "P": 20}


def seed_mp(repeat, fold):
    return MASTER_SEED + 500 * repeat + fold


def seed_inner(repeat, fold):
    return MASTER_SEED + 1000 * repeat + fold


def seed_crossfit(repeat, fold):
    return MASTER_SEED + 2000 * repeat + fold


def seed_noise(repeat, fold):
    return MASTER_SEED + 3000 * repeat + fold


# ----------------------------------------------------------------- metrics
ECE_BINS = 10                 # equal width over [0,1]; final bin includes 1.0
LOGLOSS_EPS = 1e-15

# ----------------------------------------------------------------- calibration
CAL_CANDIDATES = ("none", "sigmoid", "isotonic")
CAL_TIE_EPS = 1e-12           # ties broken in CAL_CANDIDATES order

# ----------------------------------------------------------------- thresholds
P_STAR_GRID = np.arange(0.30, 1.001, 0.02)
M_STAR_GRID = np.arange(0.00, 0.85, 0.05)
MIN_STRONG_SUPPORT = 30
TARGET_SELECTIVE_ACCURACY = 0.98      # declared primary operating point
EXPLORATORY_TARGET = 1.00             # post-hoc sensitivity analysis only
ENTROPY_QUANTILE_GRID = np.arange(0.50, 1.00, 0.005)
ENTROPY_RISK_ACCURACY = 0.90
ENTROPY_FALLBACK_QUANTILE = 0.99

# ----------------------------------------------------------------- selection
TOL_MACRO_F1 = 0.005          # practical selection tolerance, absolute
TOL_LOGLOSS_REL = 0.02        # practical selection tolerance, relative
SELECTION_ORDER = ["LogisticRegression", "DecisionTree", "KNN",
                   "RandomForest", "XGBoost", "LightGBM"]
BASELINE_NAME = "DummyBaseline"
REQUIRED_CANDIDATES = [BASELINE_NAME] + SELECTION_ORDER

# ----------------------------------------------------------------- OOD
OOD_CHI2_QUANTILE = 0.995
OOD_RANGE_MARGIN = 0.05
OOD_COV_RIDGE = 1e-6

# ----------------------------------------------------------------- explainability
PERM_IMPORTANCE_REPEATS = 10
PERM_IMPORTANCE_METRIC = "f1_macro"

# ----------------------------------------------------------------- evidence tags
EVIDENCE_REAL = "real_data"
EVIDENCE_SMOKE = "smoke_test"
