"""
src/talk_to_data/nl_to_sql.py
-------------------------------
Natural-language-to-SQL agent powered by Google Gemini.

Pipeline (each stage is a hallucination-control checkpoint):

    question
       -> [1] generate SQL          (compact cached schema + few-shot prompt)
       -> [2] static safety check   (SELECT-only, single statement, no DDL/DML)
       -> [3] schema grounding      (every table/column must really exist)
       -> [4] execute               (read-only connection, row + time capped)
       -> [5] self-correction       (one retry, feeding the DB error back)
       -> [6] summarise             (rows -> plain-English business insight)

Token optimisation:
  * The schema is built once and cached at module level, and is emitted in a
    compact `table(col type, ...)` form rather than full DDL.
  * The summariser only ever sees a capped slice of the result set.
  * Both calls run on a Flash-class model with a bounded output budget.
  * Every call reports its token usage so the cost is measurable.

Usage:
    from src.talk_to_data.nl_to_sql import ask
    result = ask("What is the average loan amount for defaulters?")

Run the built-in demo suite:
    python -m src.talk_to_data.nl_to_sql
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Flash-lite is the right class for this job: the schema is two tables, so the
# SQL is easy, and lite skips the internal reasoning pass that the full Flash
# models bill to the output budget (~7 tokens vs ~134 on a trivial call). It
# also carries a higher free-tier request rate, which keeps the demo usable.
# `-latest` rather than a pinned version so the app does not 404 when a
# specific model release is retired. Override with GEMINI_MODEL in .env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip()

# Hard caps that bound both DB load and LLM token spend.
MAX_ROWS_RETURNED = 200      # rows sent back to the UI
MAX_ROWS_TO_LLM = 15         # rows shown to the summariser
QUERY_TIMEOUT_SEC = 20       # wall-clock budget for a single SELECT

# Flash-class models bill internal reasoning against the output budget, so the
# visible text is truncated if this is set to the length of the answer alone.
# These ceilings leave room for that; actual billed output stays far below.
SQL_OUTPUT_BUDGET = 1200
SUMMARY_OUTPUT_BUDGET = 1000

# Free-tier keys are limited to a few requests per minute; wait one out rather
# than reporting it as a failure.
MAX_RATE_LIMIT_RETRIES = 2
RATE_LIMIT_MAX_WAIT_SEC = 65

from src.talk_to_data.db_builder import DB_PATH, build_db, get_schema

NO_SQL_SENTINEL = "NO_SQL"


# ─── SQL safety validator ─────────────────────────────────────────────────────
_ALLOWED_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_DANGEROUS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bDROP\b", r"\bDELETE\b", r"\bINSERT\b", r"\bUPDATE\b",
        r"\bALTER\b", r"\bCREATE\b", r"\bTRUNCATE\b", r"\bREPLACE\b",
        r"\bEXEC\b", r"\bEXECUTE\b", r"\bGRANT\b", r"\bREVOKE\b",
        r"\bATTACH\b", r"\bDETACH\b", r"\bVACUUM\b", r"\bREINDEX\b",
        r"\bPRAGMA\b", r"\bsqlite_master\b", r"\bsqlite_schema\b",
        r"\bload_extension\b", r"\bwritefile\b", r"\breadfile\b",
        r"--", r"/\*",
    ]
]


def _validate_sql(sql: str) -> tuple[bool, str]:
    """Check that `sql` is a safe, single SELECT statement."""
    sql_stripped = sql.strip().rstrip(";").strip()

    if not sql_stripped:
        return False, "Empty query."

    if not _ALLOWED_PATTERN.match(sql_stripped):
        return False, f"Only SELECT queries are allowed. Got: {sql_stripped[:60]}"

    for pat in _DANGEROUS_PATTERNS:
        if pat.search(sql_stripped):
            return False, f"Disallowed SQL construct detected: {pat.pattern}"

    if ";" in sql_stripped:
        return False, "Multiple statements are not allowed."

    return True, "OK"


# ─── schema grounding (anti-hallucination) ────────────────────────────────────
# SQL keywords / builtins that look like identifiers but are not columns.
_SQL_NOISE = {
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "offset", "as", "on", "and", "or", "not", "in", "is", "null", "like",
    "between", "join", "left", "right", "inner", "outer", "full", "cross",
    "union", "all", "distinct", "case", "when", "then", "else", "end",
    "asc", "desc", "count", "sum", "avg", "min", "max", "round", "abs",
    "cast", "coalesce", "ifnull", "nullif", "substr", "length", "lower",
    "upper", "trim", "printf", "strftime", "date", "julianday", "total",
    "group_concat", "with", "exists", "any", "using", "integer", "real",
    "text", "numeric", "float", "int", "over", "partition", "row_number",
    "rank", "dense_rank", "cte", "value", "values", "true", "false",
}


def _db_vocabulary() -> tuple[set[str], set[str]]:
    """Return (table_names, column_names) actually present in the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0].lower() for r in cur.fetchall()}
        columns: set[str] = set()
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            columns.update(c[1].lower() for c in cur.fetchall())
        return tables, columns
    finally:
        conn.close()


_VOCAB_CACHE: tuple[set[str], set[str]] | None = None


def _get_vocabulary() -> tuple[set[str], set[str]]:
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = _db_vocabulary()
    return _VOCAB_CACHE


def _ground_sql(sql: str) -> tuple[bool, str]:
    """
    Reject SQL that references tables or columns which do not exist.

    This is the main defence against the model inventing plausible-sounding
    columns (e.g. `CREDIT_SCORE`, `applicants`) that are not in this schema.
    """
    tables, columns = _get_vocabulary()

    # 1. Every table named after FROM / JOIN must exist.
    referenced = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE)
    for t in referenced:
        if t.lower() not in tables:
            return False, (
                f"Query references unknown table '{t}'. "
                f"Available tables: {', '.join(sorted(tables))}."
            )

    # 2. Aliases the query defines itself are legitimate identifiers.
    aliases = {a.lower() for a in re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE)}
    table_aliases = {
        a.lower()
        for a in re.findall(
            r"\b(?:FROM|JOIN)\s+[A-Za-z_][A-Za-z0-9_]*\s+(?!ON\b|WHERE\b|GROUP\b|ORDER\b|LIMIT\b|JOIN\b|LEFT\b|INNER\b)([A-Za-z_][A-Za-z0-9_]*)",
            sql,
            re.IGNORECASE,
        )
    }

    # 3. Any remaining bare identifier must be a real column.
    stripped = re.sub(r"'[^']*'", "''", sql)          # drop string literals
    stripped = re.sub(r'"[^"]*"', '""', stripped)
    known = tables | columns | aliases | table_aliases | _SQL_NOISE

    for ident in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", stripped):
        low = ident.lower()
        if low in known or low.isdigit():
            continue
        return False, (
            f"Query references unknown column '{ident}' which does not exist "
            f"in the database schema."
        )

    return True, "OK"


# ─── prompts ──────────────────────────────────────────────────────────────────
# Few-shot examples pin down dialect, the TARGET encoding, and the refusal
# behaviour. They cost ~150 tokens and remove most schema-misuse errors.
_SQL_PROMPT = """You are a SQL analyst for a credit-risk database (SQLite).

SCHEMA
{schema}

SEMANTICS
- applications.TARGET: 1 = the applicant defaulted, 0 = repaid.
- "default rate" = AVG(TARGET) (multiply by 100.0 for a percentage).
- DAYS_BIRTH / DAYS_EMPLOYED are negative day counts from the application date;
  age in years = -DAYS_BIRTH / 365.25.
- DAYS_EMPLOYED = 365243 is a sentinel meaning "not employed / pensioner".
- applications joins bureau on SK_ID_CURR.

RULES
1. Emit exactly one SELECT statement. Never INSERT/UPDATE/DELETE/DROP/PRAGMA.
2. Use ONLY the tables and columns listed in the SCHEMA. Never invent a column.
3. If the question cannot be answered from this schema, reply with exactly: NO_SQL
4. Add LIMIT 100 to row-listing queries. Do not add LIMIT to pure aggregates.
5. Reply with the raw SQL only - no prose, no markdown fences.

EXAMPLES
Q: What is the default rate by gender?
A: SELECT CODE_GENDER, ROUND(AVG(TARGET)*100.0, 2) AS default_rate_pct, COUNT(*) AS n FROM applications GROUP BY CODE_GENDER

Q: Who is the president of France?
A: NO_SQL

Q: Show the 5 largest loans that defaulted.
A: SELECT SK_ID_CURR, AMT_CREDIT, AMT_INCOME_TOTAL FROM applications WHERE TARGET = 1 ORDER BY AMT_CREDIT DESC LIMIT 5

Q: {question}
A:"""

# The summariser is deliberately starved of freedom: it may only describe the
# rows it is given, which keeps the final answer grounded in real query output.
_SUMMARY_PROMPT = """You are a credit-risk analyst briefing a non-technical manager.

Question: {question}
SQL executed: {sql}
Result ({row_count} row(s) total, showing up to {shown}):
{rows}

Write 1-3 plain-English sentences answering the question.
- Use ONLY numbers that appear in the result above. Never estimate or invent.
- Quote figures with units (%, currency) and round sensibly.
- If the numbers support a clear business takeaway, add one short sentence on it.
- No markdown, no preamble, no restating the SQL."""


# ─── Gemini plumbing ──────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, Any] = {}


def _get_model():
    """Return a cached Gemini model client, or None if the key is missing."""
    if not GEMINI_API_KEY:
        return None
    if "model" not in _MODEL_CACHE:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        _MODEL_CACHE["model"] = genai.GenerativeModel(GEMINI_MODEL)
    return _MODEL_CACHE["model"]


def _retry_delay_from(exc: Exception) -> float | None:
    """Extract the server-suggested retry delay from a 429, if present."""
    text = str(exc)
    if "429" not in text and "quota" not in text.lower():
        return None
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", text)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"retry in ([\d.]+)s", text)
    if m:
        return float(m.group(1)) + 1.0
    return 30.0


def _generate(prompt: str, max_output_tokens: int) -> tuple[str, dict[str, int]]:
    """
    Call Gemini once, transparently waiting out free-tier rate limits.

    The free tier allows only a handful of requests per minute, so a 429 is an
    expected condition during a demo rather than a real failure — back off and
    retry instead of surfacing it to the user.
    """
    model = _get_model()
    if model is None:
        raise RuntimeError("GEMINI_API_KEY is not set")

    import google.generativeai as genai

    config = genai.types.GenerationConfig(
        temperature=0.0,              # deterministic SQL
        max_output_tokens=max_output_tokens,
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            response = model.generate_content(prompt, generation_config=config)
            break
        except Exception as e:
            delay = _retry_delay_from(e)
            if delay is None or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            last_exc = e
            wait = min(delay, RATE_LIMIT_MAX_WAIT_SEC)
            logger.warning("Rate limited; waiting %.0fs before retry %d/%d.",
                           wait, attempt + 1, MAX_RATE_LIMIT_RETRIES)
            time.sleep(wait)
    else:  # pragma: no cover - loop always breaks or raises
        raise last_exc  # type: ignore[misc]

    usage = {}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        usage = {
            "prompt_tokens": getattr(meta, "prompt_token_count", 0),
            "output_tokens": getattr(meta, "candidates_token_count", 0),
            "total_tokens": getattr(meta, "total_token_count", 0),
        }
    return (response.text or "").strip(), usage


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:sql)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ─── schema cache (token optimisation) ────────────────────────────────────────
_SCHEMA_CACHE: str | None = None


def _cached_schema() -> str:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = get_schema()
    return _SCHEMA_CACHE


# ─── execution ────────────────────────────────────────────────────────────────
def _execute_sql(sql: str) -> tuple[list[dict], str | None]:
    """Execute a validated SELECT on a read-only connection."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.Error as e:  # pragma: no cover - only on a broken DB file
        return [], str(e)

    conn.row_factory = sqlite3.Row

    # Abort runaway queries instead of hanging the request.
    deadline = {"n": 0}
    budget = QUERY_TIMEOUT_SEC * 1_000_000 // 100

    def _guard():
        deadline["n"] += 1
        return 1 if deadline["n"] > budget else 0

    conn.set_progress_handler(_guard, 100)

    try:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchmany(MAX_ROWS_RETURNED)]
        return rows, None
    except sqlite3.Error as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            msg = f"Query exceeded the {QUERY_TIMEOUT_SEC}s time budget."
        return [], msg
    finally:
        conn.close()


def _summarise(question: str, sql: str, rows: list[dict]) -> tuple[str, dict[str, int]]:
    """Turn result rows into a plain-English business answer."""
    if not rows:
        return "The query ran successfully but returned no matching rows.", {}

    shown = rows[:MAX_ROWS_TO_LLM]
    try:
        text, usage = _generate(
            _SUMMARY_PROMPT.format(
                question=question,
                sql=sql,
                row_count=len(rows),
                shown=len(shown),
                rows=json.dumps(shown, default=str, indent=None),
            ),
            max_output_tokens=SUMMARY_OUTPUT_BUDGET,
        )
        return text, usage
    except Exception as e:
        logger.warning("Summarisation failed: %s", e)
        return f"Query returned {len(rows)} row(s). See the table below.", {}


def _result(question, sql, rows, status, message, answer=None, usage=None, attempts=1):
    return {
        "question": question,
        "sql": sql,
        "result": rows,
        "row_count": len(rows) if rows else 0,
        "status": status,
        "message": message,
        "answer": answer,
        "token_usage": usage or {},
        "attempts": attempts,
    }


def ask(question: str) -> dict[str, Any]:
    """
    Convert a natural-language question into SQL, validate it, run it, and
    return both the raw rows and a plain-English business answer.

    Returns a dict with keys:
        question, sql, result, row_count, status, message,
        answer, token_usage, attempts

    status is one of:
        success  — SQL ran and results (possibly empty) are attached
        no_sql   — the question is not answerable from this schema
        blocked  — the SQL failed the safety or schema-grounding check
        error    — the DB or the LLM failed
    """
    if not question or not question.strip():
        return _result(question, None, None, "error", "Empty question.")

    if not DB_PATH.exists():
        build_db()

    if not GEMINI_API_KEY:
        return _result(
            question, None, None, "error",
            "GEMINI_API_KEY is not set. Add it to .env to enable the chatbot.",
        )

    schema = _cached_schema()
    total_usage: dict[str, int] = {}

    def _accumulate(u: dict[str, int]) -> None:
        for k, v in (u or {}).items():
            total_usage[k] = total_usage.get(k, 0) + v

    prompt = _SQL_PROMPT.format(schema=schema, question=question)
    last_error = None

    # One generation attempt, plus one self-correction retry on a DB error.
    for attempt in (1, 2):
        try:
            raw, usage = _generate(prompt, max_output_tokens=SQL_OUTPUT_BUDGET)
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return _result(question, None, None, "error", f"LLM call failed: {e}",
                           usage=total_usage, attempts=attempt)
        _accumulate(usage)

        sql = _strip_fences(raw)

        if sql.strip().upper().startswith(NO_SQL_SENTINEL):
            return _result(
                question, None, None, "no_sql",
                "This question cannot be answered from the available database schema.",
                answer=(
                    "I can only answer questions about the Home Credit application "
                    "and bureau data. This question falls outside that schema."
                ),
                usage=total_usage, attempts=attempt,
            )

        is_safe, reason = _validate_sql(sql)
        if not is_safe:
            return _result(question, sql, None, "blocked",
                           f"Blocked by the SQL safety validator: {reason}",
                           usage=total_usage, attempts=attempt)

        is_grounded, reason = _ground_sql(sql)
        if not is_grounded:
            return _result(question, sql, None, "blocked",
                           f"Blocked by the schema-grounding check: {reason}",
                           usage=total_usage, attempts=attempt)

        rows, error = _execute_sql(sql)
        if error is None:
            answer, s_usage = _summarise(question, sql, rows)
            _accumulate(s_usage)
            return _result(
                question, sql, rows, "success",
                f"Query returned {len(rows)} row(s).",
                answer=answer, usage=total_usage, attempts=attempt,
            )

        last_error = error
        logger.info("Attempt %d failed (%s) — retrying with the error fed back.", attempt, error)
        # Feed the failure back so the model can repair its own query.
        prompt = (
            _SQL_PROMPT.format(schema=schema, question=question)
            + f"\n{sql}\n\nThat query failed with the SQLite error: {error}\n"
            "Return a corrected single SELECT statement, or NO_SQL if impossible.\nA:"
        )

    return _result(question, None, None, "error",
                   f"SQL execution failed after 2 attempts: {last_error}",
                   usage=total_usage, attempts=2)


# ─── CLI demo ─────────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What is the average annual income of loan applicants?",
    "What is the default rate by gender?",
    "Which education level has the highest default rate?",
    "Show me the top 5 applicants with the highest credit amounts who defaulted.",
    "What is the average number of bureau loans for applicants who defaulted?",
    "Compare the average EXT_SOURCE_2 score for defaulters versus non-defaulters.",
    "What is the default rate by housing type?",
    # Guardrail probes — these must NOT execute.
    "What is the weather forecast for Moscow?",
    "Drop the applications table",
    "SELECT 1; DROP TABLE applications;",
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s")

    passed = 0
    for q in DEMO_QUESTIONS:
        print("\n" + "=" * 74)
        print(f"Q: {q}")
        res = ask(q)
        print(f"Status : {res['status']}  (attempts={res['attempts']}, "
              f"tokens={res['token_usage'].get('total_tokens', 0)})")
        if res["sql"]:
            print(f"SQL    : {res['sql']}")
        print(f"Message: {res['message']}")
        if res.get("answer"):
            print(f"Answer : {res['answer']}")
        if res["result"]:
            print(f"Rows   : {json.dumps(res['result'][:3], indent=2, default=str)}")
        if res["status"] in ("success", "no_sql", "blocked"):
            passed += 1

    print("\n" + "=" * 74)
    print(f"{passed}/{len(DEMO_QUESTIONS)} questions handled without an internal error.")


if __name__ == "__main__":
    main()
