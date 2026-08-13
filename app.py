from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phase2 import ood_policy  # noqa: E402  (required before joblib unpickling)


MODEL_PATH = ROOT / "outputs_phase2" / "deployment_model_bundle.joblib"
RESULTS_PATH = ROOT / "outputs_phase2" / "website_data.json"

INPUTS = {
    "N": {"label": "Nitrogen (N) index", "default": 90.0, "step": 1.0, "help": "Dataset range: 0–140. Treat this as the source dataset's N index, not a universal field unit."},
    "P": {"label": "Phosphorus (P) index", "default": 42.0, "step": 1.0, "help": "Dataset range: 5–145. Treat this as the source dataset's P index."},
    "K": {"label": "Potassium (K) index", "default": 43.0, "step": 1.0, "help": "Dataset range: 5–205. Treat this as the source dataset's K index."},
    "temperature": {"label": "Temperature (°C)", "default": 20.88, "step": 0.1, "help": "Dataset range: approximately 8.8–43.7 °C."},
    "humidity": {"label": "Relative humidity (%)", "default": 82.0, "step": 0.1, "help": "Dataset range: approximately 14.3–100%."},
    "ph": {"label": "Soil pH", "default": 6.50, "step": 0.01, "help": "Dataset range: approximately 3.50–9.94."},
    "rainfall": {"label": "Rainfall (mm)", "default": 202.94, "step": 0.1, "help": "Dataset range: approximately 20.2–298.6 mm. The source does not document the measurement window."},
}


st.set_page_config(
    page_title="FarmFit AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 1.8rem; padding-bottom: 3rem;}
      .route-card {padding: 1rem 1.15rem; border-radius: .75rem; margin: .75rem 0 1.1rem 0;}
      .route-strong {background: #e8f5e9; border: 1px solid #2e7d32;}
      .route-review {background: #ffebee; border: 1px solid #c62828;}
      .route-alternatives {background: #fff8e1; border: 1px solid #f9a825;}
      .small-note {color: #5f6b76; font-size: .9rem;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the verified FarmFit model…")
def load_assets():
    bundle = joblib.load(MODEL_PATH)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if results.get("evidence_class") != "real_data" or results.get("is_smoke_test"):
        raise RuntimeError("Refusing to load non-evidence website data.")
    if results.get("analysis_run_id") != bundle.get("metadata", {}).get("analysis_run_id"):
        raise RuntimeError("Model and website-data identities do not match.")
    if list(bundle.get("feature_order", [])) != list(INPUTS):
        raise RuntimeError("Model feature order does not match the interface.")
    return bundle, results


def calibrated_probabilities(bundle, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(bundle["base_estimator"].predict_proba(frame), dtype=float)
    mapping = bundle.get("calibration_mapping")
    if mapping is None:
        return raw
    return np.asarray(mapping.transform(raw), dtype=float)


def confidence_frame(probabilities: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rows = np.arange(len(probabilities))[:, None]
    top = probabilities[rows, order[:, :3]]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / np.log(probabilities.shape[1])
    return pd.DataFrame(
        {
            "top1_class": order[:, 0],
            "top2_class": order[:, 1],
            "top3_class": order[:, 2],
            "top1_prob": top[:, 0],
            "top2_prob": top[:, 1],
            "top3_prob": top[:, 2],
            "margin_12": top[:, 0] - top[:, 1],
            "top3_cum_prob": top.sum(axis=1),
            "entropy": entropy,
        }
    ), order


def make_prediction(bundle, values: dict[str, float]):
    frame = pd.DataFrame([[values[name] for name in bundle["feature_order"]]], columns=bundle["feature_order"])
    probabilities = calibrated_probabilities(bundle, frame)
    confidence, order = confidence_frame(probabilities)
    predicted_class = confidence["top1_class"].to_numpy()
    reference = bundle["ood_reference"].score(frame, predicted_class)
    category, entropy_flag, trigger = ood_policy.apply_policy(
        confidence,
        reference["flag_union"].to_numpy(),
        bundle["thresholds_primary_98"],
    )
    return {
        "frame": frame,
        "probabilities": probabilities[0],
        "confidence": confidence.iloc[0],
        "order": order[0],
        "reference": reference.iloc[0],
        "category": str(category[0]),
        "entropy_flag": bool(entropy_flag[0]),
        "trigger": str(trigger[0]),
    }


def quantity(stats: list[dict], name: str) -> dict:
    return next(row for row in stats if row["quantity"] == name)


try:
    bundle, results = load_assets()
except Exception as exc:
    st.error("The verified model package could not be loaded.")
    st.exception(exc)
    st.stop()


st.title("🌱 FarmFit AI")
st.subheader("Confidence-aware crop-suitability decision support")
st.caption(
    "Educational prototype · 22 crop classes · Final evidence run "
    f"{results['analysis_run_id']}"
)

with st.sidebar:
    st.header("Important boundary")
    st.warning(
        "This is not production-ready agronomic advice. The model was evaluated on one "
        "structured educational dataset and has not been validated on independent farm data."
    )
    st.write("**Model:**", bundle["model_family"])
    st.write("**Calibration:**", bundle["calibration_method"])
    st.write("**Evidence:**", "10 repeated 5-fold evaluations")
    st.write("**Privacy:**", "Inputs are processed in the current session and are not deliberately stored by this app.")


predict_tab, evidence_tab, guide_tab = st.tabs(["Try the prototype", "Evidence dashboard", "How to read it"])

with predict_tab:
    st.markdown("### Enter one measurement profile")
    st.write("Use the same measurement definitions as the source dataset. Hover over each field for its documented range.")
    st.caption("The form is pre-filled with one source-dataset demonstration profile; change any value to explore the prototype.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        values: dict[str, float] = {}
        for i, (name, spec) in enumerate(INPUTS.items()):
            target = left if i < 4 else right
            with target:
                values[name] = st.number_input(
                    spec["label"],
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                    help=spec["help"],
                    format="%.2f",
                )
        submitted = st.form_submit_button("Generate recommendation", type="primary", width="stretch")

    if submitted:
        prediction = make_prediction(bundle, values)
        classes = bundle["class_order"]
        conf = prediction["confidence"]
        route = prediction["category"]
        top_class = classes[int(conf["top1_class"])]

        route_text = {
            "STRONG": ("route-strong", "Strong recommendation", "The leading crop cleared the confidence gates and no review flag fired."),
            "EXPERT_REVIEW": ("route-review", "Expert review required", "The leading model candidate is shown, but uncertainty or an unfamiliar-input flag overrides a Strong recommendation."),
            "ALTERNATIVES": ("route-alternatives", "Show alternatives", "No review flag fired, but the leading prediction did not clear both confidence gates."),
        }
        css, title, explanation = route_text[route]
        st.markdown(
            f'<div class="route-card {css}"><b>{title}</b><br>{explanation}</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Leading crop", top_class.title())
        m2.metric("Model confidence", f"{float(conf['top1_prob']):.1%}")
        m3.metric("Top-two margin", f"{float(conf['margin_12']):.1%}")
        m4.metric("Route", route.replace("_", " ").title())

        st.markdown("#### Top three model candidates")
        top_rows = []
        for rank, idx in enumerate(prediction["order"][:3], start=1):
            top_rows.append(
                {"Rank": rank, "Crop": classes[int(idx)].title(), "Probability": float(prediction["probabilities"][int(idx)])}
            )
        top_df = pd.DataFrame(top_rows).set_index("Crop")
        st.bar_chart(top_df["Probability"], horizontal=True, color="#2e8b57")
        st.dataframe(
            top_df.assign(Probability=top_df["Probability"].map(lambda x: f"{x:.2%}")),
            width="stretch",
        )

        flags = []
        if bool(prediction["reference"]["flag_multivariate"]):
            flags.append("multivariate reference flag")
        if bool(prediction["reference"]["flag_univariate_range"]):
            flags.append("input outside the reference range")
        if prediction["entropy_flag"]:
            flags.append("high prediction entropy")
        st.info("Review signals: " + (", ".join(flags) if flags else "none"))

        with st.expander("Technical decision details"):
            th = bundle["thresholds_primary_98"]
            st.json(
                {
                    "route": route,
                    "expert_trigger": prediction["trigger"],
                    "top_probability": round(float(conf["top1_prob"]), 6),
                    "top_two_margin": round(float(conf["margin_12"]), 6),
                    "normalised_entropy": round(float(conf["entropy"]), 6),
                    "primary_thresholds": {
                        "P_star": th["p_star"],
                        "M_star": th["m_star"],
                        "E_star": th["entropy_star"],
                    },
                    "reference_flags": {
                        "multivariate": bool(prediction["reference"]["flag_multivariate"]),
                        "univariate_range": bool(prediction["reference"]["flag_univariate_range"]),
                    },
                }
            )

with evidence_tab:
    st.markdown("### What the held-out evidence showed")
    repeats = pd.DataFrame(results["track_b_by_repeat"])
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Mean accuracy", f"{repeats['accuracy'].mean():.2%}")
    e2.metric("Mean macro F1", f"{repeats['macro_f1'].mean():.2%}")
    e3.metric("Mean top-3 accuracy", f"{repeats['top3_accuracy'].mean():.2%}")
    e4.metric("Mean ECE", f"{repeats['ece'].mean():.4f}")

    stats = results["advisory_policy"]["primary_operating_point"]["across_repeats"]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Strong coverage", f"{quantity(stats, 'STRONG_coverage')['mean']:.2%}")
    p2.metric("Strong accuracy", f"{quantity(stats, 'STRONG_selective_accuracy')['mean']:.2%}")
    p3.metric("Expert-review share", f"{quantity(stats, 'EXPERT_REVIEW_coverage')['mean']:.2%}")
    p4.metric("Errors captured", f"{quantity(stats, 'error_capture_rate')['mean']:.2%}")

    st.markdown("#### Model-family comparison")
    comparison = pd.DataFrame(results["track_a_model_comparison"])
    display = comparison[["model", "mean_macro_f1", "mean_log_loss", "mean_ece"]].copy()
    display.columns = ["Model", "Macro F1", "Log loss", "ECE"]
    st.dataframe(display.style.format({"Macro F1": "{:.4f}", "Log loss": "{:.4f}", "ECE": "{:.4f}"}), width="stretch")

    st.markdown("#### Global model signals")
    shap_global = pd.DataFrame(results["shap_deployment_only"]["global"])
    st.bar_chart(shap_global.set_index("feature")["mean_abs_shap"], horizontal=True, color="#2e8b57")
    st.caption("These SHAP values describe the fitted model, not agronomic causality and not a case-specific explanation.")

with guide_tab:
    st.markdown("### What the three routes mean")
    st.markdown(
        """
        - **Strong:** the model's leading crop clears both probability gates and neither uncertainty rule fires.
        - **Alternatives:** show the top three candidates because the leading crop is not strong enough.
        - **Expert Review:** uncertainty or an unfamiliar-input signal overrides a high-looking probability.

        **Precedence matters:** Expert Review is checked first, then Strong, then Alternatives.
        """
    )
    st.markdown("### What this website cannot claim")
    for limit in results["scope_limits"]:
        st.markdown(f"- {limit}")
    st.info("Use this page for classroom demonstration and controlled pilot design—not autonomous farm decisions.")

st.divider()
st.caption(
    "FarmFit AI educational prototype · Model prototype not independently evaluated · "
    f"Evidence run {results['analysis_run_id']}"
)
