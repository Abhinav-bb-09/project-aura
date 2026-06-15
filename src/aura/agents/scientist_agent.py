"""
src/aura/agents/scientist_agent.py

The Scientist Agent interprets statistical findings in business language.

Layer A (stats.py) runs the actual math — p-values, IQR, correlations.
Layer B (this file) asks the LLM to explain what those numbers mean
to a business stakeholder who doesn't care about p-values.
"""

import json
from typing import Any

import pandas as pd

from aura.tools.stats import analyze_dataframe
from aura.core.llm import call_llm, get_session_cost


SCIENTIST_SYSTEM = """You are a senior data scientist explaining findings to a business audience.
You receive statistical analysis results and produce clear, actionable interpretations.
Rules:
- Explain statistical concepts in plain English — no jargon
- Always connect findings to business impact
- Be specific with numbers
- Flag anything that should concern a decision-maker
- Respond only in valid JSON"""


def run_scientist_agent(
    dataframe: pd.DataFrame,
    question: str,
    sql: str,
) -> dict[str, Any]:
    """
    Run statistical analysis and LLM interpretation on query results.

    Parameters
    ----------
    dataframe : the result DataFrame from the Engineer Agent
    question  : the original user question
    sql       : the SQL that produced this data (for context)

    Returns
    -------
    dict with keys:
        stats        : raw statistical analysis (Layer A)
        interpretation: LLM business interpretation (Layer B)
        confidence_score: 0-100
        confidence_label: green/yellow/red
        cost         : session cost
    """
    # ── Layer A: run the stats ────────────────────────────────
    print("Running statistical analysis...")
    stats = analyze_dataframe(dataframe, question)

    if "error" in stats:
        return {"error": stats["error"]}

    # ── Layer B: LLM interpretation ───────────────────────────
    print(f"Interpreting findings (confidence: {stats['confidence_score']}/100)...")
    prompt = _build_interpretation_prompt(stats, question, sql)
    raw_response = call_llm(prompt, system_instruction=SCIENTIST_SYSTEM)
    interpretation = _parse_interpretation(raw_response)

    confidence_score = stats["confidence_score"]

    return {
        "stats"            : stats,
        "interpretation"   : interpretation,
        "confidence_score" : confidence_score,
        "confidence_label" : _confidence_label(confidence_score),
        "cost"             : get_session_cost(),
    }


def _build_interpretation_prompt(stats: dict, question: str, sql: str) -> str:
    """Build a focused prompt from statistical results."""
    lines = [
        f"Original question: {question}",
        f"SQL used: {sql}",
        f"Result shape: {stats['shape'][0]} rows × {stats['shape'][1]} columns",
        f"Confidence score: {stats['confidence_score']}/100",
        "",
    ]

    # Numeric summaries
    if stats["numeric_stats"]:
        lines.append("Numeric column statistics:")
        for col, s in stats["numeric_stats"].items():
            lines.append(
                f"  {col}: mean={s['mean']}, median={s['median']}, "
                f"std={s['std']}, skewness={s['skewness']}"
            )

    # Strong correlations
    strong_pairs = stats["correlations"].get("strong_pairs", [])
    if strong_pairs:
        lines.append("\nStrong correlations found:")
        for p in strong_pairs[:3]:
            lines.append(f"  {p['col_a']} ↔ {p['col_b']}: r={p['r']} ({p['strength']})")

    # Auto-detected findings
    if stats["findings"]:
        lines.append("\nAuto-detected patterns:")
        for f in stats["findings"]:
            lines.append(f"  [{f['severity'].upper()}] {f['description']}")

    # Confidence factors
    if stats["confidence_factors"]:
        lines.append("\nConfidence score factors:")
        for f in stats["confidence_factors"]:
            lines.append(f"  {f}")

    lines.append("""
Respond with a JSON object with this exact structure:
{
  "summary": "2-3 sentence plain-English summary of what the data shows",
  "key_findings": [
    {
      "finding": "one clear observation",
      "business_impact": "why this matters to the business",
      "confidence": "high|medium|low"
    }
  ],
  "anomalies": ["any concerning patterns worth flagging"],
  "recommended_actions": ["specific action 1", "specific action 2"],
  "caveats": ["limitation 1", "limitation 2"]
}""")

    return "\n".join(lines)


def _parse_interpretation(raw: str) -> dict:
    """Parse LLM JSON response, stripping markdown fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw_response": raw}


def _confidence_label(score: int) -> str:
    """Map score to green/yellow/red for UI display."""
    if score >= 75:
        return "green"
    elif score >= 40:
        return "yellow"
    else:
        return "red"