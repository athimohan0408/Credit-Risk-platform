"""
Predict endpoint — accepts a raw applicant data dict and returns default probability + SHAP explanation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from src.ml.predictor import predict as ml_predict

router = APIRouter()


class PredictRequest(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = 150000.0
    AMT_CREDIT: Optional[float] = 500000.0
    AMT_ANNUITY: Optional[float] = 25000.0
    AMT_GOODS_PRICE: Optional[float] = 450000.0
    DAYS_BIRTH: Optional[float] = -14600.0     # ~40 years old
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


@router.post("/predict")
def predict_risk(request: PredictRequest):
    try:
        applicant_data = request.model_dump()
        result = ml_predict(applicant_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
def explain_prediction(request: PredictRequest):
    """
    Return per-applicant SHAP values for the top contributing features.
    Used by the frontend to show a waterfall-style explanation after prediction.
    """
    try:
        from src.explainability.shap_explainer import (
            FRIENDLY_NAMES,
            explain_readable,
            get_single_shap,
        )
        import numpy as np

        applicant_data = request.model_dump()
        shap_vals, feature_cols = get_single_shap(applicant_data)

        # Build sorted list of (feature, shap_value) by abs importance
        shap_pairs = sorted(
            zip(feature_cols, shap_vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Return top 15 features
        top_shap = [
            {
                "feature": f,
                "label": FRIENDLY_NAMES.get(f, f.replace("_", " ").title()),
                "shap_value": round(v, 5),
            }
            for f, v in shap_pairs[:15]
        ]

        return {
            "shap_values": top_shap,
            "base_value": round(float(np.mean(shap_vals)), 5),
            # Plain-English drivers so a loan officer can read the decision
            # without knowing what a SHAP value is.
            "reasons": explain_readable(applicant_data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
