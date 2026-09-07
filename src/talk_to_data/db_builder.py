"""
src/talk_to_data/db_builder.py
-------------------------------
Builds a SQLite database from the Home Credit CSVs.
The DB is stored at models/credit_risk.db.

Run standalone to (re)build the DB:
    python -m src.talk_to_data.db_builder
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

DB_PATH = _PROJECT_ROOT / "models" / "credit_risk.db"


def build_db(force_rebuild: bool = False) -> Path:
    """
    Build SQLite DB with the following tables:
        applications  — from application_train.csv (all rows)
        bureau        — from bureau.csv (key columns only, for performance)

    Parameters
    ----------
    force_rebuild : if True, drop and recreate existing tables.

    Returns
    -------
    Path to the SQLite DB file.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not force_rebuild:
        logger.info("DB already exists at %s — skipping rebuild. "
                    "Pass force_rebuild=True to regenerate.", DB_PATH)
        return DB_PATH

    from src.data.loader import _data_path

    conn = sqlite3.connect(DB_PATH)

    # ── applications table ───────────────────────────────────────────────
    logger.info("Loading application_train.csv into SQLite ...")
    train_path = _data_path("application_train.csv")
    train = pd.read_csv(train_path)

    # Keep a cleaned subset of columns (avoid loading 120+ columns into chat)
    KEY_APP_COLS = [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
        "FLAG_PHONE", "FLAG_EMAIL",
    ]
    avail_cols = [c for c in KEY_APP_COLS if c in train.columns]
    train_sub = train[avail_cols]
    train_sub.to_sql("applications", conn, if_exists="replace", index=False)
    logger.info("applications table: %d rows, %d columns", len(train_sub), len(train_sub.columns))

    # ── bureau table ─────────────────────────────────────────────────────
    logger.info("Loading bureau.csv into SQLite ...")
    bureau_path = _data_path("bureau.csv")
    bureau = pd.read_csv(bureau_path)

    KEY_BUREAU_COLS = [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY",
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE", "DAYS_CREDIT_UPDATE",
    ]
    avail_bureau_cols = [c for c in KEY_BUREAU_COLS if c in bureau.columns]
    bureau_sub = bureau[avail_bureau_cols]
    bureau_sub.to_sql("bureau", conn, if_exists="replace", index=False)
    logger.info("bureau table: %d rows, %d columns", len(bureau_sub), len(bureau_sub.columns))

    # ── indexes ──────────────────────────────────────────────────────────
    conn.execute("CREATE INDEX IF NOT EXISTS idx_app_id ON applications(SK_ID_CURR)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bureau_id ON bureau(SK_ID_CURR)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_app_target ON applications(TARGET)")
    conn.commit()
    conn.close()

    logger.info("SQLite DB built at %s", DB_PATH)
    return DB_PATH


def get_schema() -> str:
    """Return the DB schema as a string (for LLM prompting)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        schema_parts.append(f"Table: {table}({col_defs})")
    conn.close()
    return "\n".join(schema_parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    path = build_db(force_rebuild=True)
    print(f"\nDB built at: {path}")
    print("\nSchema:")
    print(get_schema())
