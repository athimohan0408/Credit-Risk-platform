"""
src/explainability/rule_derivation.py
--------------------------------------
Derives human-readable underwriting rules from SHAP values, then validates each
one against the observed default rate in the training data.

Why the direction is inferred from a correlation, not from mean SHAP
-------------------------------------------------------------------
SHAP values are additive contributions measured against a base value, so for
any feature the *signed* mean SHAP over a representative sample is ~0 by
construction — the positive and negative contributions cancel. Taking the sign
of that near-zero number reads pure noise, which is how an earlier version of
this module produced rules that contradicted the EDA (e.g. "higher external
credit score -> higher risk").

The direction of a feature's effect is instead the relationship between the
feature's *value* and its SHAP contribution:

    corr(feature value, SHAP value) > 0  =>  higher value pushes risk UP
    corr(feature value, SHAP value) < 0  =>  higher value pushes risk DOWN

Spearman is used so the measure survives the monotone-but-non-linear responses
that gradient-boosted trees produce.

Every rule is then held to an empirical test: the segment it flags must default
more often than the portfolio baseline. A rule whose lift is <= 1.0 is reported
as REJECTED rather than being explained away.

Run:
    python -m src.explainability.rule_derivation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

MODELS_DIR = _PROJECT_ROOT / "models"

# A rule must lift the default rate by at least this much to be accepted.
MIN_LIFT = 1.05
# Below this |correlation| the direction is not trustworthy enough to act on.
MIN_ABS_CORR = 0.05
# A flagged segment smaller than this is too thin to underwrite against.
MIN_SEGMENT_FRAC = 0.02


# Single source of truth for display names and units, shared with the SHAP module.
from src.explainability.shap_explainer import (
    FRIENDLY_NAMES as FEATURE_DESCRIPTIONS,
    _SYNONYM_GROUPS,
    _display_value,
)


def _describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())


def _numeric_rule(
    feature: str,
    x: pd.Series,
    y: pd.Series,
    corr: float,
    baseline: float,
) -> dict:
    """Build and empirically validate a threshold rule for a numeric feature."""
    desc = _describe(feature)

    if corr > 0:
        threshold = float(x.quantile(0.75))
        mask = x > threshold
        comparator = ">"
        direction = "higher values increase default risk"
    else:
        threshold = float(x.quantile(0.25))
        mask = x < threshold
        comparator = "<"
        direction = "lower values increase default risk"

    # Render the threshold in the same units the UI shows elsewhere. The DAYS_*
    # columns are negative day counts, so converting them to years reverses the
    # ordering and the comparator has to flip with it.
    shown_threshold, shown_str = _display_value(feature, threshold)
    probe_lo, _ = _display_value(feature, threshold - 1.0)
    reversed_orientation = probe_lo > shown_threshold
    shown_comparator = comparator
    if reversed_orientation:
        shown_comparator = "<" if comparator == ">" else ">"

    condition = f"{desc} {shown_comparator} {shown_str}"

    seg_n = int(mask.sum())
    seg_rate = float(y[mask].mean()) if seg_n else 0.0
    lift = seg_rate / baseline if baseline else 0.0

    return {
        "condition": condition,
        "threshold": round(threshold, 4),
        "comparator": comparator,
        "threshold_display": shown_str,
        "comparator_display": shown_comparator,
        "direction": direction,
        "segment_size": seg_n,
        "segment_share_pct": round(100.0 * seg_n / len(x), 2),
        "segment_default_rate_pct": round(100.0 * seg_rate, 2),
        "lift_vs_baseline": round(lift, 2),
    }


def _categorical_rule(
    feature: str,
    x: pd.Series,
    y: pd.Series,
    encoder,
    baseline: float,
) -> dict:
    """
    Build a category-membership rule for a label-encoded feature.

    A ">" threshold on an arbitrary integer encoding is meaningless, so instead
    name the specific categories whose observed default rate beats the baseline.
    """
    desc = _describe(feature)

    stats = (
        pd.DataFrame({"code": x.values, "y": y.values})
        .groupby("code")["y"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Ignore categories too rare to underwrite against.
    stats = stats[stats["count"] >= max(MIN_SEGMENT_FRAC * len(x), 30)]
    risky = stats[stats["mean"] > baseline * MIN_LIFT].sort_values("mean", ascending=False)

    if risky.empty:
        return {
            "condition": f"{desc}: no category exceeds the baseline default rate",
            "direction": "no actionable split",
            "segment_size": 0,
            "segment_share_pct": 0.0,
            "segment_default_rate_pct": round(100.0 * baseline, 2),
            "lift_vs_baseline": 1.0,
            "categories": [],
        }

    def _label(code: int) -> str:
        try:
            return str(encoder.inverse_transform([int(code)])[0])
        except Exception:
            return str(code)

    top = risky.head(3)
    labels = [_label(c) for c in top["code"]]

    mask = x.isin(top["code"].tolist())
    seg_n = int(mask.sum())
    seg_rate = float(y[mask].mean()) if seg_n else 0.0

    return {
        "condition": f"{desc} IS ONE OF ({', '.join(labels)})",
        "direction": "these categories carry above-baseline default risk",
        "segment_size": seg_n,
        "segment_share_pct": round(100.0 * seg_n / len(x), 2),
        "segment_default_rate_pct": round(100.0 * seg_rate, 2),
        "lift_vs_baseline": round(seg_rate / baseline, 2) if baseline else 0.0,
        "categories": [
            {
                "value": lab,
                "default_rate_pct": round(100.0 * float(r["mean"]), 2),
                "n": int(r["count"]),
            }
            for lab, (_, r) in zip(labels, top.iterrows())
        ],
    }


def derive_rules(top_n: int = 15, n_explain: int = 5000) -> dict:
    """
    Compute SHAP values and derive validated underwriting rules.

    Returns a dict with `rules`, `rejected`, `baseline_default_rate_pct`
    and `validation` (a summary of how the rules were checked).
    """
    from src.explainability.shap_explainer import compute_shap_values
    from src.ml.predictor import get_artifacts

    _, _, cat_encoders, _ = get_artifacts()

    logger.info("Computing SHAP values for rule derivation ...")
    shap_values, X_sample, y_sample, feature_cols = compute_shap_values(n_explain=n_explain)

    baseline = float(y_sample.mean())
    logger.info("Baseline default rate in SHAP sample: %.2f%%", 100 * baseline)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    col_index = {f: i for i, f in enumerate(feature_cols)}
    accepted: list[dict] = []
    rejected: list[dict] = []
    used_groups: list[set] = []

    for _, row in ranked.iterrows():
        if len(accepted) >= top_n:
            break

        feat = row["feature"]
        if feat not in X_sample.columns:
            continue

        # age_years and DAYS_BIRTH encode the same fact; emitting both produces
        # two identically-worded rules. Keep only the higher-SHAP member.
        group = next((g for g in _SYNONYM_GROUPS if feat in g), None)
        if group is not None:
            if group in used_groups:
                continue
            used_groups.append(group)

        x = X_sample[feat]
        shap_col = pd.Series(shap_values[:, col_index[feat]])

        # Direction of effect: how the feature's value moves its own SHAP value.
        if x.nunique() < 2:
            continue
        corr, _ = spearmanr(x, shap_col)
        if not np.isfinite(corr):
            continue

        is_categorical = feat in cat_encoders
        if is_categorical:
            rule = _categorical_rule(feat, x, y_sample, cat_encoders[feat], baseline)
        else:
            if abs(corr) < MIN_ABS_CORR:
                rejected.append({
                    "feature": feat,
                    "reason": (
                        f"|corr(value, SHAP)| = {abs(corr):.3f} < {MIN_ABS_CORR} — "
                        "no consistent direction of effect, so no rule was emitted."
                    ),
                })
                continue
            rule = _numeric_rule(feat, x, y_sample, corr, baseline)

        entry = {
            "feature": feat,
            "description": _describe(feat),
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 4),
            "value_shap_corr": round(float(corr), 3),
            "type": "categorical" if is_categorical else "numeric",
            **rule,
        }

        # Empirical gate: the flagged segment must actually be riskier.
        if (
            entry["lift_vs_baseline"] >= MIN_LIFT
            and entry["segment_share_pct"] >= MIN_SEGMENT_FRAC * 100
        ):
            entry["rank"] = len(accepted) + 1
            entry["rule_text"] = (
                f"IF {entry['condition']} "
                f"THEN flag for enhanced underwriting "
                f"[{entry['segment_default_rate_pct']}% default vs "
                f"{round(100 * baseline, 2)}% baseline "
                f"= {entry['lift_vs_baseline']}x risk, "
                f"covers {entry['segment_share_pct']}% of applicants]"
            )
            accepted.append(entry)
        else:
            rejected.append({
                "feature": feat,
                "reason": (
                    f"Flagged segment defaults at {entry['segment_default_rate_pct']}% "
                    f"vs {round(100 * baseline, 2)}% baseline "
                    f"(lift {entry['lift_vs_baseline']}x, "
                    f"covers {entry['segment_share_pct']}%) — "
                    "below the acceptance threshold."
                ),
            })

    return {
        "baseline_default_rate_pct": round(100 * baseline, 2),
        "n_samples_used": int(len(X_sample)),
        "rules": accepted,
        "rejected": rejected,
        "validation": {
            "direction_method": "Spearman corr(feature value, feature SHAP value)",
            "acceptance_criteria": {
                "min_lift_vs_baseline": MIN_LIFT,
                "min_abs_value_shap_corr": MIN_ABS_CORR,
                "min_segment_share": MIN_SEGMENT_FRAC,
            },
            "note": (
                "Every accepted rule was checked against observed TARGET rates in "
                "the held sample; rules that did not raise the default rate were "
                "rejected rather than reported."
            ),
        },
    }


def sanity_check(payload: dict) -> list[str]:
    """
    Check accepted rules against credit-risk domain expectations.

    Unlike the previous version, a violation here is reported as a genuine
    failure — it is not rationalised away.
    """
    # feature -> expected sign of corr(value, SHAP). +1 = higher value is riskier.
    EXPECTED = {
        "ext_source_mean": -1, "ext_source_min": -1,
        "EXT_SOURCE_1": -1, "EXT_SOURCE_2": -1, "EXT_SOURCE_3": -1,
        "age_years": -1, "employment_years": -1, "employment_age_ratio": -1,
        "DAYS_BIRTH": +1,          # less negative = younger = riskier
        "credit_income_ratio": +1, "annuity_income_ratio": +1,
        "bureau_debt_ratio": +1, "bureau_active_ratio": +1,
    }

    out: list[str] = []
    for rule in payload["rules"]:
        feat = rule["feature"]
        if feat not in EXPECTED:
            continue
        expected = EXPECTED[feat]
        actual = np.sign(rule["value_shap_corr"])
        if actual == expected:
            out.append(
                f"[OK] {feat}: corr(value, SHAP) = {rule['value_shap_corr']:+.3f} "
                f"matches the expected direction, and the flagged segment defaults at "
                f"{rule['segment_default_rate_pct']}% "
                f"({rule['lift_vs_baseline']}x baseline)."
            )
        else:
            out.append(
                f"[FAIL] {feat}: corr(value, SHAP) = {rule['value_shap_corr']:+.3f} "
                f"contradicts the expected credit-risk direction. Investigate before "
                f"using this rule in underwriting."
            )

    if not out:
        out.append("[INFO] No rules covered a feature with a known expected direction.")
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 78)
    print("  UNDERWRITING RULE DERIVATION FROM SHAP VALUES")
    print("=" * 78)

    payload = derive_rules(top_n=15, n_explain=5000)
    baseline = payload["baseline_default_rate_pct"]

    print(f"\nBaseline default rate: {baseline}%  "
          f"(n = {payload['n_samples_used']:,} applicants)")

    print("\n" + "-" * 78)
    print(f"  {len(payload['rules'])} ACCEPTED RULES")
    print("-" * 78)
    for r in payload["rules"]:
        print(f"\n  Rule {r['rank']:02d} [{r['type']}]  mean|SHAP|={r['mean_abs_shap']:.4f}  "
              f"corr={r['value_shap_corr']:+.3f}")
        print(f"    {r['rule_text']}")

    if payload["rejected"]:
        print("\n" + "-" * 78)
        print(f"  {len(payload['rejected'])} REJECTED CANDIDATES")
        print("-" * 78)
        for r in payload["rejected"]:
            print(f"  - {r['feature']}: {r['reason']}")

    warnings = sanity_check(payload)
    payload["sanity_checks"] = warnings

    print("\n" + "-" * 78)
    print("  DIRECTIONAL SANITY CHECKS")
    print("-" * 78)
    for w in warnings:
        print(f"  {w}")

    failures = [w for w in warnings if w.startswith("[FAIL]")]
    print(f"\n  {len(warnings) - len(failures)}/{len(warnings)} directional checks passed.")

    rules_path = MODELS_DIR / "business_rules.json"
    with open(rules_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Rules saved to: {rules_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
