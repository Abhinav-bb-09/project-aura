"""
src/aura/agents/engineer_agent.py

The self-correcting SQL Engineer Agent.

Core loop:
  1. Receive natural language question + schema profile
  2. Write a SQL query
  3. Execute it
  4. If it fails — read the error, reason about it, rewrite, retry
  5. Max 3 attempts. Each attempt is logged.

This loop is Differentiator #2. The logging of each attempt's
reasoning is what makes it demonstrable in an interview.
"""

import json
from typing import Any

from aura.tools.sql_runner import execute_query, result_to_dataframe
from aura.core.llm import call_llm, get_session_cost
from aura.config import MAX_SQL_RETRIES


# ── System instruction ────────────────────────────────────────
ENGINEER_SYSTEM = """You are a precise SQL engineer. You write SQLite-compatible SQL queries.
Rules:
- Return ONLY the raw SQL query, no explanation, no markdown, no backticks
- Use exact table and column names from the schema provided
- Always use SELECT — never DROP, DELETE, UPDATE, or INSERT
- For aggregations, always include a GROUP BY
- Prefer readable column aliases (e.g. AS total_revenue)
- When asked about top N items, use ORDER BY ... DESC LIMIT N"""

CORRECTION_SYSTEM = """You are a SQL debugger. A query failed and you must fix it.
Rules:
- Return ONLY the corrected SQL query, nothing else
- Read the error message carefully — it tells you exactly what's wrong
- Common SQLite issues: wrong column names, missing quotes around strings,
  incorrect aggregation, ambiguous column references
- Do not change the intent of the query, only fix the syntax or logic error"""


def run_engineer_agent(
    question: str,
    db_path: str,
    schema_profile: dict,
) -> dict[str, Any]:
    """
    Translate a natural language question into SQL, execute it,
    and self-correct on failure.

    Parameters
    ----------
    question       : natural language question from the user
    db_path        : path to the SQLite database file
    schema_profile : the profile dict from run_schema_agent()

    Returns
    -------
    dict with keys:
        success        : bool
        question       : original question
        final_sql      : the SQL that ultimately worked (or last attempt)
        result         : execute_query result dict
        dataframe      : pandas DataFrame (if success)
        attempts       : list of attempt logs (for transparency UI)
        total_attempts : int
        cost           : session cost dict
    """
    schema_context = _build_schema_context(schema_profile)
    attempts = []

    # ── Initial query generation ──────────────────────────────
    sql = _generate_initial_sql(question, schema_context)
    print(f"\nAttempt 1 SQL:\n{sql}\n")

    # ── Self-correcting execution loop ────────────────────────
    for attempt_num in range(1, MAX_SQL_RETRIES + 1):
        result = execute_query(db_path, sql)

        attempt_log = {
            "attempt": attempt_num,
            "sql": sql,
            "success": result["success"],
            "error": result.get("error"),
            "row_count": result.get("row_count", 0),
        }
        attempts.append(attempt_log)

        if result["success"]:
            print(f"✓ Query succeeded on attempt {attempt_num} ({result['row_count']} rows)")
            df = result_to_dataframe(result)
            return {
                "success": True,
                "question": question,
                "final_sql": sql,
                "result": result,
                "dataframe": df,
                "attempts": attempts,
                "total_attempts": attempt_num,
                "cost": get_session_cost(),
            }

        # ── Query failed — attempt correction ─────────────────
        print(f"✗ Attempt {attempt_num} failed: {result['error']}")

        if attempt_num < MAX_SQL_RETRIES:
            print(f"  Requesting correction (attempt {attempt_num + 1}/{MAX_SQL_RETRIES})...")
            sql = _correct_sql(
                original_question=question,
                failed_sql=sql,
                error_message=result["error"],
                schema_context=schema_context,
            )
            print(f"\nAttempt {attempt_num + 1} SQL:\n{sql}\n")

    # ── All attempts exhausted ────────────────────────────────
    print(f"✗ All {MAX_SQL_RETRIES} attempts failed.")
    return {
        "success": False,
        "question": question,
        "final_sql": sql,
        "result": result,
        "dataframe": None,
        "attempts": attempts,
        "total_attempts": MAX_SQL_RETRIES,
        "cost": get_session_cost(),
    }


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def _build_schema_context(profile: dict) -> str:
    """
    Build a compact schema string the LLM can reference when writing SQL.
    Format: table_name(col1 type, col2 type, ...)
    This is the standard way to give an LLM schema context efficiently.
    """
    lines = []
    for table_name, table in profile["tables"].items():
        cols = []
        for col_name, col in table["columns"].items():
            # Map our semantic types back to SQL-friendly type hints
            type_hint = _semantic_to_sql_type(col["semantic_type"])
            cols.append(f"{col_name} {type_hint}")
        lines.append(f"{table_name}({', '.join(cols)})")

    if profile.get("relationships"):
        lines.append("\nRelationships:")
        for r in profile["relationships"]:
            lines.append(f"  {r['table_a']}.{r['column_a']} = {r['table_b']}.{r['column_b']}")

    return "\n".join(lines)


def _generate_initial_sql(question: str, schema_context: str) -> str:
    """Ask the LLM to write the first SQL attempt."""
    prompt = f"""Schema:
{schema_context}

Question: {question}

Write a SQLite query to answer this question."""

    response = call_llm(prompt, system_instruction=ENGINEER_SYSTEM)
    return _clean_sql(response)


def _correct_sql(
    original_question: str,
    failed_sql: str,
    error_message: str,
    schema_context: str,
) -> str:
    """Ask the LLM to fix a failed SQL query given the error message."""
    prompt = f"""Schema:
{schema_context}

Original question: {original_question}

Failed SQL:
{failed_sql}

Error message:
{error_message}

Fix the SQL query."""

    response = call_llm(prompt, system_instruction=CORRECTION_SYSTEM)
    return _clean_sql(response)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    """
    Strip markdown fences and whitespace from an LLM SQL response.
    LLMs sometimes return ```sql ... ``` even when told not to.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    return cleaned.strip()


def _semantic_to_sql_type(semantic_type: str) -> str:
    """Map our profiler's semantic types to SQL type hints for the prompt."""
    mapping = {
        "numeric": "NUMERIC",
        "categorical": "TEXT",
        "text": "TEXT",
        "id_or_code": "TEXT",
        "datetime": "DATETIME",
        "datetime_string": "TEXT",
        "boolean": "INTEGER",
    }
    return mapping.get(semantic_type, "TEXT")