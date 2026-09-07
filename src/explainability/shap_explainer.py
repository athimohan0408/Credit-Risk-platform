"""
src/explainability/shap_explainer.py
-------------------------------------
Computes SHAP values for the trained LightGBM model and provides helper
functions used by both the rule_derivation module and the Streamlit UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical human-readable feature names, shared with the rule-derivation module
# so the UI never shows two different labels for the same feature.
FRIENDLY_NAMES = {
    "EXT_SOURCE_1":           "External Credit Score #1",
    "EXT_SOURCE_2":           "External Credit Score #2",
    "EXT_SOURCE_3":           "External Credit Score #3",
    "ext_source_mean":        "Average External Credit Score",
    "ext_source_min":         "Lowest External Credit Score",
    # These are stored as negative day counts but rendered in years, so the
    # labels must not mention days.
    "DAYS_BIRTH":             "Applicant Age",
    "age_years":              "Applicant Age",
    "DAYS_EMPLOYED":          "Employment Duration",
    "employment_years":       "Employment Duration",
    "employment_age_ratio":   "Share of Life Spent Employed",
    "AMT_CREDIT":             "Loan Amount",
    "AMT_ANNUITY":            "Monthly Loan Annuity",
    "AMT_INCOME_TOTAL":       "Annual Income",
    "AMT_GOODS_PRICE":        "Price of Goods Financed",
    "credit_income_ratio":    "Loan-to-Income Ratio",
    "annuity_income_ratio":   "Annuity-to-Income Ratio",
    "credit_goods_ratio":     "Loan-to-Goods-Price Ratio",
    "bureau_loan_count":      "Number of Prior Bureau Loans",
    "bureau_debt_ratio":      "Bureau Debt-to-Credit Ratio",
    "bureau_active_count":    "Active Bureau Loans",
    "bureau_active_ratio":    "Share of Bureau Loans Still Active",
    "bureau_avg_days_credit": "Average Age of Bureau Credit History",
    "bureau_avg_amt_credit_sum": "Average Prior Bureau Loan Size",
    "bureau_max_amt_credit_sum": "Largest Prior Bureau Loan",
    "bureau_avg_credit_day_overdue": "Average Days Overdue on Bureau Loans",
    "DAYS_LAST_PHONE_CHANGE": "Time Since Last Phone Change",
    "DAYS_ID_PUBLISH":        "Time Since ID Document Was Issued",
    "DAYS_REGISTRATION":      "Time Since Registration Was Changed",
    "REGION_POPULATION_RELATIVE": "Region Population Density",
    "REGION_RATING_CLIENT":   "Region Risk Rating",
    "CODE_GENDER":            "Gender",
    "NAME_EDUCATION_TYPE":    "Education Level",
    "NAME_INCOME_TYPE":       "Income Type",
    "NAME_FAMILY_STATUS":     "Family Status",
    "NAME_HOUSING_TYPE":      "Housing Type",
    "NAME_CONTRACT_TYPE":     "Contract Type",
    "OCCUPATION_TYPE":        "Occupation",
    "ORGANIZATION_TYPE":      "Employer Industry",
    "FLAG_OWN_CAR":           "Owns a Car",
    "FLAG_OWN_REALTY":        "Owns Property",
    "CNT_CHILDREN":           "Number of Children",
}


def compute_shap_values(
    n_background: int = 2000,
    n_explain: int = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series, list[str]]:
    """
    Compute SHAP values for a sample of training data.

    Parameters
    ----------
    n_background : int — number of background samples for the TreeExplainer
    n_explain    : int — number of rows to explain
    random_state : int

    Returns
    -------
    shap_values  : np.ndarray  shape (n_explain, n_features)
    X_sample     : pd.DataFrame  the rows that were explained
    y_sample     : pd.Series     the true TARGET for those rows, so that derived
                                 rules can be checked against real default rates
    feature_cols : list[str]
    """
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.data.loader import load_train
    from src.data.preprocessor import preprocess_test
    from src.ml.predictor import get_artifacts, get_train_medians

    model, feature_cols, cat_encoders, _ = get_artifacts()

    logger.info("Loading data for SHAP computation ...")
    df_raw = load_train()

    # Sample the RAW rows first, then transform only those. Preprocessing all
    # 307k rows to keep a few thousand needs ~1.5 GB of transient float64
    # buffers and fails outright on a small machine.
    rng = np.random.RandomState(random_state)
    n = min(n_explain, len(df_raw))
    idx = rng.choice(len(df_raw), size=n, replace=False)
    df_sample = df_raw.iloc[idx].reset_index(drop=True)
    del df_raw

    y_sample = df_sample["TARGET"].astype(int).reset_index(drop=True)

    # Transform with the encoders and medians the model was actually fitted
    # with, rather than refitting on the sample — refitting could assign a
    # different integer to a category and silently invalidate every SHAP value.
    X_sample = preprocess_test(df_sample, cat_encoders, feature_cols, get_train_medians())
    if "SK_ID_CURR" in X_sample.columns:
        X_sample = X_sample.drop(columns=["SK_ID_CURR"])

    logger.info("Computing SHAP values for %d samples ...", n)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification LightGBM, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # use positive-class SHAP values

    logger.info("SHAP values shape: %s", np.array(shap_values).shape)
    return np.array(shap_values), X_sample, y_sample, feature_cols


def get_shap_summary(shap_values: np.ndarray, feature_cols: list[str], top_n: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame of mean |SHAP| per feature (global importance).

    Columns: feature, mean_abs_shap, direction ('+' = increases risk, '-' = decreases risk)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "feature":        feature_cols,
        "mean_abs_shap":  mean_abs,
        "mean_shap":      mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    df["direction"] = df["mean_shap"].apply(lambda v: "[+] risk" if v > 0 else "[-] risk")
    return df


def _prepare_single(applicant_data: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """Preprocess one applicant into the model's feature space."""
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.data.preprocessor import preprocess_test
    from src.ml.predictor import get_artifacts, get_train_medians

    _, feature_cols, cat_encoders, _ = get_artifacts()

    df = pd.DataFrame([applicant_data])
    X = preprocess_test(df, cat_encoders, feature_cols, get_train_medians())
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])
    return X, feature_cols, cat_encoders


def get_single_shap(
    applicant_data: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for a single applicant dict (used by the UI).
    """
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.data.preprocessor import preprocess_test
    from src.ml.predictor import get_artifacts, get_train_medians

    model, feature_cols, cat_encoders, _ = get_artifacts()

    df = pd.DataFrame([applicant_data])
    X = preprocess_test(df, cat_encoders, feature_cols, get_train_medians())
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.array(shap_values)[0], feature_cols


# ─── plain-language explanations ──────────────────────────────────────────────
# Features that encode the same underlying fact. Only the most influential
# member of each group is reported, so the reviewer does not read "age" twice.
_SYNONYM_GROUPS = [
    {"age_years", "DAYS_BIRTH"},
    {"employment_years", "DAYS_EMPLOYED"},
]

# Features whose raw units are not meaningful to a reader, with a converter to
# something that is. Applied to the model-space value before formatting.
# The label already names the quantity, so these templates supply only the
# value and its unit.
_VALUE_DISPLAY = {
    "DAYS_BIRTH":             (lambda v: -v / 365.25, "{:.0f} years"),
    "DAYS_EMPLOYED":          (lambda v: -v / 365.25, "{:.1f} years"),
    "DAYS_ID_PUBLISH":        (lambda v: -v / 365.25, "{:.1f} years"),
    "DAYS_REGISTRATION":      (lambda v: -v / 365.25, "{:.1f} years"),
    "DAYS_LAST_PHONE_CHANGE": (lambda v: -v / 365.25, "{:.1f} years"),
    "age_years":              (lambda v: v, "{:.0f} years"),
    "employment_years":       (lambda v: v, "{:.1f} years"),
    "AMT_INCOME_TOTAL":       (lambda v: v, "{:,.0f}"),
    "AMT_CREDIT":             (lambda v: v, "{:,.0f}"),
    "AMT_ANNUITY":            (lambda v: v, "{:,.0f}"),
    "AMT_GOODS_PRICE":        (lambda v: v, "{:,.0f}"),
}


def _display_value(feature: str, value: float) -> tuple[float, str]:
    """
    Convert a model-space value into display space.

    Returns both the converted number and its formatted string, because the
    comparison against the portfolio norm has to happen in the SAME space that
    is shown. DAYS_BIRTH is negative, so -8000 > -15750 in raw space while
    22 years < 43 years in display space — comparing raw and printing converted
    produces the self-contradictory "22 years, above the norm of 43 years".
    """
    conv, template = _VALUE_DISPLAY.get(feature, (None, None))
    if conv is not None:
        try:
            converted = float(conv(float(value)))
            return converted, template.format(converted)
        except (TypeError, ValueError):
            pass
    v = float(value)
    if abs(v) >= 1000:
        return v, f"{v:,.0f}"
    if abs(v) < 1 and v != 0:
        return v, f"{v:.3f}"
    return v, f"{v:,.2f}"


def explain_readable(applicant_data: dict, top_n: int = 5) -> dict:
    """
    Turn one applicant's SHAP values into plain-English reasons.

    SHAP output alone is not an explanation for a non-technical reviewer:
    "DAYS_ID_PUBLISH = +0.08" means nothing to a loan officer.

    Each sentence states two separately verifiable facts — the applicant's
    actual value against the portfolio norm, and the direction in which the
    model moved the score. It deliberately does NOT infer the value from the
    SHAP sign: a young applicant can still receive a risk-reducing age
    contribution because of interactions, and asserting "older than typical"
    on the strength of a negative SHAP value would simply be false.

    Returns
    -------
    dict with `increases` and `decreases` — each a list of
    {feature, label, value, shap_value, share_pct, sentence}.
    """
    from src.ml.predictor import get_train_medians

    X, feature_cols, cat_encoders = _prepare_single(applicant_data)
    shap_vals, _ = get_single_shap(applicant_data)
    medians = get_train_medians()
    row = X.iloc[0]

    pairs = sorted(
        zip(feature_cols, shap_vals.tolist()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    total = sum(abs(v) for _, v in pairs) or 1.0

    def _entry(feature: str, shap_value: float) -> dict:
        raises = shap_value > 0
        label = FRIENDLY_NAMES.get(feature, feature.replace("_", " ").title())
        value = row.get(feature)
        effect = "increases" if raises else "reduces"

        if feature in cat_encoders and value is not None:
            # Decode back to the category the applicant actually falls in.
            try:
                shown = str(cat_encoders[feature].inverse_transform([int(value)])[0])
            except Exception:
                _, shown = _display_value(feature, value)
            sentence = f'{label} is "{shown}" - this {effect} the predicted risk.'
        else:
            if value is None:
                shown, comparison = "unknown", ""
            else:
                shown_num, shown = _display_value(feature, value)
                comparison = ""
                if medians is not None and feature in medians.index:
                    med_num, med_str = _display_value(feature, float(medians[feature]))
                    if shown_num > med_num:
                        comparison = f", above the portfolio norm of {med_str}"
                    elif shown_num < med_num:
                        comparison = f", below the portfolio norm of {med_str}"
            sentence = f"{label} is {shown}{comparison} - this {effect} the predicted risk."

        return {
            "feature":    feature,
            "label":      label,
            "value":      None if value is None else round(float(value), 4),
            "shap_value": round(float(shap_value), 5),
            "share_pct":  round(100.0 * abs(shap_value) / total, 1),
            "sentence":   sentence,
        }

    def _take(want_positive: bool) -> list[dict]:
        picked: list[dict] = []
        used_groups: list[set] = []
        for feature, value in pairs:
            if (value > 0) != want_positive or value == 0:
                continue
            group = next((g for g in _SYNONYM_GROUPS if feature in g), None)
            if group is not None:
                if group in used_groups:
                    continue
                used_groups.append(group)
            picked.append(_entry(feature, value))
            if len(picked) >= top_n:
                break
        return picked

    return {"increases": _take(True), "decreases": _take(False)}
