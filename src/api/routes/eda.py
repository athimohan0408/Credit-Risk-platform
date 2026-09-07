"""
EDA summary endpoint — returns pre-computed statistics from models/metrics.json
plus the hardcoded EDA facts derived from the full 307,511-row dataset.
"""
from fastapi import APIRouter
import json
from pathlib import Path

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR   = _PROJECT_ROOT / "models"


@router.get("/summary")
def get_eda_summary():
    metrics: dict = {}
    mp = _MODELS_DIR / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            metrics = json.load(f)

    return {
        "total_applicants":  307511,
        "default_rate":      8.07,
        "class_balance":     {"default": 8.07, "no_default": 91.93},
        "metrics":           metrics,
        "insights": [
            "91.93% of applicants did not default — severe class imbalance (11.4:1) handled via scale_pos_weight in LightGBM.",
            "EXT_SOURCE_3, EXT_SOURCE_2, and EXT_SOURCE_1 are the top 3 predictors of default by SHAP importance.",
            "Defaulters are on average 3.4 years younger — non-default mean age = 44.2 yrs vs 40.8 yrs for defaulters.",
            "DAYS_EMPLOYED=365243 sentinel (retirees, 18% of applicants) has a lower default rate of 5.4% vs 8.7% for workers.",
            "Credit-to-goods ratio > 1.2 raises default risk from 5.7% to 11.3% — indicates cash-out or misrepresentation.",
            "Renting applicants default at 12.3% vs 7.8% for property owners — housing stability is a key risk signal.",
            "Maternity leave (40%) and Unemployed (36%) income types have the highest default rates vs 8.07% baseline.",
            "Active bureau loan fraction is 49.5% for defaulters vs 40.6% for non-defaulters — open credit lines signal risk.",
            "Employment-to-age ratio > 21.4% of lifetime: 5.8% default rate vs 10.8% for those < 5.5% — stability signal.",
            "Annuity burden is monotonically risky: default rates rise from 7.3% (Q1) to 8.6% (Q4) of annuity/income ratio.",
        ],
        "income_type_defaults": {
            "Maternity leave":      40.0,
            "Unemployed":           36.0,
            "Working":               9.5,
            "Commercial associate":  7.5,
            "State servant":         5.3,
            "Pensioner":             5.6,
            "Businessman":           5.6,
            "Student":               0.0,
        },
        "housing_type_defaults": {
            "Rented apartment":   12.3,
            "With parents":       11.7,
            "Municipal apartment": 10.3,
            "Co-op apartment":     8.6,
            "House / apartment":   7.8,
            "Office apartment":    6.6,
        },
        "gender_defaults": {
            "Male": 10.2,
            "Female": 6.9,
        },
        "contract_type_defaults": {
            "Cash loans":      8.35,
            "Revolving loans": 5.48,
        },
    }
