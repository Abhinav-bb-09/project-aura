"""
src/aura/agents/schema_agent.py

Layer B: LLM-powered semantic enrichment of the deterministic profile.
Takes the profile dict from profiler.py and asks Gemini to interpret it.

This agent answers questions that pandas cannot:
- What does this column mean in business terms?
- Which columns are most analytically interesting?
- What questions would a human analyst ask of this dataset?
"""

import json
from typing import Any

from aura.tools.profiler import profile_dataset
from aura.core.llm import call_llm, get_session_cost


# ── System instruction (the agent's persona) ─────────────────
SCHEMA_AGENT_SYSTEM = """You are a senior data analyst performing schema discovery.
You receive statistical profiles of datasets and provide:
1. Business-language descriptions of what each column means
2. Analytical quality assessment (what's usable, what's problematic)
3. The 5 most valuable questions this dataset can answer
4. Columns to watch out for (high nulls, suspicious distributions)

Be specific and concise. No generic filler. Respond only in valid JSON."""


def run_schema_agent(source: str) -> dict[str, Any]:
    """
    Full pipeline: profile the data, then enrich with LLM interpretation.

    Parameters
    ----------
    source : path to CSV, Excel, or SQLite file

    Returns
    -------
    dict with keys:
        profile     : raw deterministic profile (from Layer A)
        enrichment  : LLM semantic interpretation
        cost        : token usage and dollar cost for this run
    """
    # ── Layer A: deterministic profiling ─────────────────────
    print("Running deterministic profiler...")
    profile = profile_dataset(source)

    # ── Prepare the prompt ───────────────────────────────────
    # We summarize the profile rather than dumping the full dict.
    # This keeps the prompt focused and saves tokens.
    prompt = _build_enrichment_prompt(profile)

    # ── Layer B: LLM enrichment ──────────────────────────────
    print(f"Sending profile to Gemini ({profile['summary']['total_columns']} columns)...")
    raw_response = call_llm(prompt, system_instruction=SCHEMA_AGENT_SYSTEM)

    # ── Parse the JSON response ──────────────────────────────
    enrichment = _parse_enrichment_response(raw_response)

    cost = get_session_cost()
    print(f"Done. Cost: ${cost['total_usd']:.6f} | Tokens: {cost['input_tokens']} in / {cost['output_tokens']} out")

    return {
        "profile": profile,
        "enrichment": enrichment,
        "cost": cost,
    }


def _build_enrichment_prompt(profile: dict) -> str:
    """
    Build a focused prompt from the profile.
    We deliberately summarize — not dump the raw dict — to save tokens.
    """
    lines = []
    lines.append(f"Dataset: {profile['source_path']}")
    lines.append(f"Format: {profile['file_type'].upper()}")
    lines.append(f"Size: {profile['summary']['total_rows']:,} rows, {profile['summary']['total_columns']} columns\n")

    # Cap at 10 tables for large databases like Northwind
    table_items = list(profile["tables"].items())[:10]
    for table_name, table in table_items:
        lines.append(f"Table: {table_name}")
        lines.append(f"Candidate primary keys: {table['candidate_keys']}")
        lines.append("Columns:")

        # Cap at 8 columns per table and 10 tables total for large databases
        col_items = list(table["columns"].items())[:8]
        for col_name, col in col_items:
            line = f"  - {col_name}: {col['semantic_type']}, {col['distinct_count']} distinct values, {col['null_pct']}% null"

            # Add numeric stats if available — gives LLM more to reason about
            if col["semantic_type"] == "numeric":
                line += f", range [{col.get('min', '?')} – {col.get('max', '?')}]"
                if col.get("skewness") and abs(col["skewness"]) > 1:
                    line += f", skewed ({col['skewness']})"

            # Show top values for categoricals
            if "top_values" in col:
                top = list(col["top_values"].keys())[:3]
                line += f", top values: {top}"

            lines.append(line)

        lines.append("")  # blank line between tables

    if profile["relationships"]:
        lines.append("Detected cross-table relationships:")
        for r in profile["relationships"]:
            lines.append(f"  {r['table_a']}.{r['column_a']} ↔ {r['table_b']}.{r['column_b']}")

    lines.append("""
Respond with a JSON object with this exact structure:
{
  "dataset_summary": "2-3 sentence plain-English description of what this dataset is",
  "column_descriptions": {
    "<column_name>": "plain-English description of what this column means"
  },
  "data_quality_issues": [
    {"column": "<name>", "issue": "<description>", "severity": "high|medium|low"}
  ],
  "analytical_questions": [
    "question 1",
    "question 2",
    "question 3",
    "question 4",
    "question 5"
  ],
  "recommended_target_variable": "<column_name or null>",
  "overall_quality_score": <integer 0-100>
}""")

    return "\n".join(lines)


def _parse_enrichment_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response safely.
    Strip markdown code fences if Gemini wraps the JSON in them.
    """
    # Gemini sometimes wraps JSON in ```json ... ``` fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        cleaned = "\n".join(lines[1:-1])

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # If parsing fails, return the raw text so we don't lose information
        return {
            "parse_error": str(e),
            "raw_response": raw,
        }