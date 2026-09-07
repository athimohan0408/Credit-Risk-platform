"""
src/ml/train.py
---------------
Trains a LightGBM classifier on the Home Credit Default Risk dataset using
5-fold stratified cross-validation.

Outputs saved to models/:
    lgbm_model.pkl        — the final model (fit on all training data)
    feature_cols.pkl      — ordered list of feature column names
    cat_encoders.pkl      — fitted LabelEncoders for categorical columns
    metrics.json          — ROC-AUC, PR-AUC, confusion matrix, feature importances
    submission.csv        — predicted probabilities for application_test.csv

Run:
    python -m src.ml.train
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── paths ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
MODELS_DIR = _PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.data.loader import load_train, load_test
from src.data.preprocessor import preprocess_train, preprocess_test

# ─── LightGBM hyperparameters ──────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 100,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "scale_pos_weight": 10,   # compensate for class imbalance (~10:1)
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

N_FOLDS = 5
SCALE_POS_WEIGHT = LGBM_PARAMS["scale_pos_weight"]


def calibrate(p: np.ndarray, weight: float = SCALE_POS_WEIGHT) -> np.ndarray:
    """
    Undo the probability inflation caused by `scale_pos_weight`.

    Training with scale_pos_weight=w tells LightGBM that every positive is worth
    w observations, so it fits the odds of a re-balanced population rather than
    the real one. Dividing the fitted odds by w recovers the original prior:

        odds_true = odds_model / w
        p_true    = p / (p + (1 - p) * w)

    Without this, a perfectly ordinary applicant scores ~0.54 and any threshold
    reads as alarming. Ranking is untouched — the map is strictly increasing —
    so ROC-AUC is unchanged while the numbers become real probabilities.
    """
    p = np.asarray(p, dtype=float)
    return p / (p + (1.0 - p) * weight)


def _calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed default rate per probability decile."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    out = []
    for b in sorted(pd.unique(bins)):
        m = bins == b
        out.append({
            "decile": int(b) + 1,
            "mean_predicted": round(float(p[m].mean()), 4),
            "actual_default_rate": round(float(y[m].mean()), 4),
            "n": int(m.sum()),
        })
    return out


def _pick_threshold(p: np.ndarray, y: np.ndarray) -> dict:
    """
    Choose the operating threshold that maximises F1 on out-of-fold predictions.

    F1 is the neutral choice absent a stated cost of a missed default versus a
    rejected good customer; the README explains how to retune it once a lender
    supplies those costs.
    """
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.arange(0.02, 0.61, 0.01):
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {
                "threshold": round(float(t), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
            }
    return best


def train() -> None:
    # ── 1. Load & preprocess training data ────────────────────────────────
    logger.info("Loading training data ...")
    df_train_raw = load_train()

    logger.info("Preprocessing training data ...")
    X, y, cat_encoders, feature_cols, train_medians = preprocess_train(df_train_raw)

    logger.info("Feature matrix shape: %s | Default rate: %.2f%%",
                X.shape, y.mean() * 100)

    # ── 2. Cross-validation ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(200)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        fold_auc = roc_auc_score(y_val, preds)
        fold_aucs.append(fold_auc)
        logger.info("  Fold %d/%d — AUC: %.4f  best_iter: %d",
                    fold, N_FOLDS, fold_auc, model.best_iteration_)

    cv_auc = roc_auc_score(y, oof_preds)
    cv_prauc = average_precision_score(y, oof_preds)
    logger.info("\nCV ROC-AUC  : %.4f  (std %.4f)", cv_auc, np.std(fold_aucs))
    logger.info("CV PR-AUC   : %.4f", cv_prauc)

    # ── 2b. Calibration + operating point, both fitted on out-of-fold preds ──
    oof_cal = calibrate(oof_preds)
    brier_raw = brier_score_loss(y, oof_preds)
    brier_cal = brier_score_loss(y, oof_cal)
    logger.info("OOF Brier — raw: %.4f  calibrated: %.4f", brier_raw, brier_cal)
    logger.info("Mean calibrated probability: %.4f  (actual default rate: %.4f)",
                oof_cal.mean(), y.mean())

    operating_point = _pick_threshold(oof_cal, y.values)
    logger.info("Chosen operating threshold: %.3f (F1=%.4f, precision=%.4f, recall=%.4f)",
                operating_point["threshold"], operating_point["f1"],
                operating_point["precision"], operating_point["recall"])

    preds_binary = (oof_cal >= operating_point["threshold"]).astype(int)
    cm = confusion_matrix(y, preds_binary)
    logger.info("Confusion matrix at threshold %.3f:\n%s", operating_point["threshold"], cm)

    np.save(MODELS_DIR / "oof_predictions.npy", oof_cal)

    # ── 3. Final model on all training data ──────────────────────────────
    logger.info("Training final model on full dataset ...")
    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X, y)

    # ── 4. Save artifacts ─────────────────────────────────────────────────
    joblib.dump(final_model,   MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(feature_cols,  MODELS_DIR / "feature_cols.pkl")
    joblib.dump(cat_encoders,  MODELS_DIR / "cat_encoders.pkl")
    joblib.dump(train_medians, MODELS_DIR / "train_medians.pkl")

    # Feature importances
    importance = pd.Series(
        final_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    metrics = {
        "cv_roc_auc":  round(float(cv_auc), 4),
        "cv_pr_auc":   round(float(cv_prauc), 4),
        "fold_aucs":   [round(float(a), 4) for a in fold_aucs],
        "n_folds":     N_FOLDS,
        "n_features":  int(X.shape[1]),
        "n_samples":   int(X.shape[0]),
        "default_rate": round(float(y.mean()), 4),
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "confusion_matrix": {
            "threshold": operating_point["threshold"],
            "threshold_space": "calibrated probability",
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
        "operating_point": operating_point,
        "calibration": {
            "method": "prior correction: p / (p + (1-p) * scale_pos_weight)",
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "oof_brier_raw": round(float(brier_raw), 4),
            "oof_brier_calibrated": round(float(brier_cal), 4),
            "mean_calibrated_probability": round(float(oof_cal.mean()), 4),
            "actual_default_rate": round(float(y.mean()), 4),
            "deciles": _calibration_table(oof_cal, y.values),
        },
        "top_features": importance.head(20).to_dict(),
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ── 5. Generate submission ────────────────────────────────────────────
    logger.info("Generating submission predictions on test set ...")
    df_test_raw = load_test()
    X_test = preprocess_test(df_test_raw, cat_encoders, feature_cols, train_medians)

    # Extract SK_ID_CURR before passing to model
    sk_ids = X_test.pop("SK_ID_CURR") if "SK_ID_CURR" in X_test.columns else df_test_raw["SK_ID_CURR"]

    # The Kaggle metric is ROC-AUC, which only depends on ranking, so the raw
    # scores are submitted as-is; calibration would not change the leaderboard.
    test_preds = final_model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"SK_ID_CURR": sk_ids.values, "TARGET": test_preds})
    sub_path = MODELS_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    logger.info("Submission saved to %s  (%d rows)", sub_path, len(submission))

    # ── 6. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  CV ROC-AUC   : {cv_auc:.4f}")
    print(f"  CV PR-AUC    : {cv_prauc:.4f}")
    print(f"  Fold AUCs    : {[round(a, 4) for a in fold_aucs]}")
    print(f"\n  Calibration (out-of-fold):")
    print(f"    Brier raw        : {brier_raw:.4f}")
    print(f"    Brier calibrated : {brier_cal:.4f}")
    print(f"    Mean predicted   : {oof_cal.mean():.4f}  vs actual {y.mean():.4f}")
    print(f"\n  Confusion Matrix (calibrated threshold={operating_point['threshold']}):")
    print(f"    TN={cm[0,0]:6,}  FP={cm[0,1]:6,}")
    print(f"    FN={cm[1,0]:6,}  TP={cm[1,1]:6,}")
    print(f"\n  Top 10 features by importance:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:<45} {imp:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
