"""
Explainability endpoints — return business rules and global feature importance
from pre-computed model artifacts.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR   = _PROJECT_ROOT / "models"


@router.get("/rules")
def get_rules():
    rp = _MODELS_DIR / "business_rules.json"
    if rp.exists():
        with open(rp) as f:
            return json.load(f)
    return {"rules": [], "sanity_warnings": []}


@router.get("/feature_importance")
def get_feature_importance():
    mp = _MODELS_DIR / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            metrics = json.load(f)
            return {
                "top_features": metrics.get("top_features", {}),
                "cv_roc_auc":   metrics.get("cv_roc_auc"),
                "cv_pr_auc":    metrics.get("cv_pr_auc"),
                "n_features":   metrics.get("n_features"),
                "n_samples":    metrics.get("n_samples"),
                "fold_aucs":    metrics.get("fold_aucs", []),
                "confusion_matrix": metrics.get("confusion_matrix", {}),
                "operating_point":  metrics.get("operating_point", {}),
                "calibration":      metrics.get("calibration", {}),
            }
    return {"top_features": {}}


class ExplainRequest(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = 150000.0
    AMT_CREDIT: Optional[float] = 500000.0
    AMT_ANNUITY: Optional[float] = 25000.0
    AMT_GOODS_PRICE: Optional[float] = 450000.0
    DAYS_BIRTH: Optional[float] = -14600.0
    DAYS_EMPLOYED: Optional[float] = -2000.0
    EXT_SOURCE_1: Optional[float] = 0.5
    EXT_SOURCE_2: Optional[float] = 0.5
    EXT_SOURCE_3: Optional[float] = 0.5
    NAME_CONTRACT_TYPE: Optional[str] = "Cash loans"
    CODE_GENDER: Optional[str] = "M"
    FLAG_OWN_CAR: Optional[str] = "N"
    FLAG_OWN_REALTY: Optional[str] = "Y"
    NAME_INCOME_TYPE: Optional[str] = "Working"
    NAME_EDUCATION_TYPE: Optional[str] = "Secondary / secondary special"
    NAME_FAMILY_STATUS: Optional[str] = "Married"
    NAME_HOUSING_TYPE: Optional[str] = "House / apartment"
    REGION_RATING_CLIENT: Optional[int] = 2
    CNT_CHILDREN: Optional[int] = 0


@router.post("/explain")
def explain_applicant(request: ExplainRequest):
    """Per-applicant SHAP explanation endpoint."""
    try:
        from src.explainability.shap_explainer import get_single_shap
        import numpy as np

        applicant_data = request.model_dump()
        shap_vals, feature_cols = get_single_shap(applicant_data)

        shap_pairs = sorted(
            zip(feature_cols, shap_vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_shap = [
            {"feature": f, "shap_value": round(v, 5)}
            for f, v in shap_pairs[:15]
        ]
        return {
            "shap_values": top_shap,
            "base_value": round(float(np.mean(shap_vals)), 5),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
