"""
src/aura/tools/sql_runner.py

Safe SQLite execution layer. The Engineer Agent never touches the database
directly — it always goes through here.

Why a wrapper instead of calling sqlite3 directly?
- Enforces read-only queries (no DROP, DELETE, UPDATE on user data)
- Returns structured results with column names, not raw tuples
- Captures errors in a structured way so the self-correcting loop can read them
- Enforces a row limit so a bad query doesn't return 500k rows to the LLM
"""

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


MAX_ROWS_RETURNED = 500  # LLM context limit — we summarize beyond this


def execute_query(db_path: str | Path, sql: str) -> dict[str, Any]:
    """
    Execute a SQL query against a SQLite database.

    Returns a result dict with keys:
        success   : bool
        data      : list of row dicts (if success)
        columns   : list of column names (if success)
        row_count : int
        error     : error message string (if not success)
        truncated : bool — True if results were capped at MAX_ROWS_RETURNED
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return _error_result(f"Database file not found: {db_path}")

    # Block any query that modifies data
    # We check the normalized SQL to catch variations like "  DROP  TABLE"
    rejection = _check_query_safety(sql)
    if rejection:
        return _error_result(rejection)

    try:
        conn = sqlite3.connect(db_path)

        # row_factory makes rows behave like dicts instead of plain tuples
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(sql)
        raw_rows = cursor.fetchmany(MAX_ROWS_RETURNED + 1)  # fetch one extra to detect truncation

        truncated = len(raw_rows) > MAX_ROWS_RETURNED
        rows = raw_rows[:MAX_ROWS_RETURNED]

        # Convert sqlite3.Row objects to plain dicts for JSON-serializability
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        data = [dict(zip(columns, row)) for row in rows]

        conn.close()

        return {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "truncated": truncated,
            "error": None,
        }

    except sqlite3.Error as e:
        # Structured error — the self-correcting loop reads this message
        return _error_result(str(e))

    except Exception as e:
        return _error_result(f"Unexpected error: {str(e)}")


def get_table_names(db_path: str | Path) -> list[str]:
    """Return all user table names in the database."""
    conn = sqlite3.connect(db_path)
    result = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    conn.close()
    return result["name"].tolist()


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Convert a successful execute_query result to a DataFrame."""
    if not result["success"]:
        raise ValueError(f"Cannot convert failed result: {result['error']}")
    return pd.DataFrame(result["data"])


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _check_query_safety(sql: str) -> str | None:
    """
    Return an error message if the query contains write operations.
    Return None if the query is safe to run.
    """
    normalized = sql.strip().upper()
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "REPLACE"]

    for keyword in blocked:
        # Check if it's a standalone keyword (not part of a column name)
        if normalized.startswith(keyword) or f" {keyword} " in normalized:
            return f"Query blocked: '{keyword}' operations are not permitted. Read-only queries only."

    return None


def _error_result(message: str) -> dict:
    """Construct a standardized error result dict."""
    return {
        "success": False,
        "data": [],
        "columns": [],
        "row_count": 0,
        "truncated": False,
        "error": message,
    }