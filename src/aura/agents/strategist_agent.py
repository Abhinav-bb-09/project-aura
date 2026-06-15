"""
src/aura/agents/strategist_agent.py

The Strategist Agent translates statistical findings into business
recommendations that a non-technical stakeholder can act on.

This is a pure LLM agent — no deterministic layer.
The math is already done by the Scientist. The Strategist's job
is to answer: "So what? What should we actually DO about this?"
"""

import json
from typing import Any

from aura.core.llm import call_llm, get_session_cost


STRATEGIST_SYSTEM = """You are a senior business strategist and management consultant.
You receive data analysis results and produce concrete, prioritized recommendations.
Rules:
- Every recommendation must be specific and actionable — no vague advice
- Prioritize by business impact (revenue, cost, risk)
- Acknowledge data limitations honestly
- Think like a McKinsey consultant: structure, clarity, impact
- Respond only in valid JSON"""


def run_strategist_agent(
    question: str,
    scientist_result: dict,
    engineer_result: dict,
) -> dict[str, Any]:
    """
    Generate business strategy recommendations from analysis results.

    Parameters
    ----------
    question          : original user question
    scientist_result  : output from run_scientist_agent()
    engineer_result   : output from run_engineer_agent()

    Returns
    -------
    dict with keys:
        recommendations : prioritized list of business actions
        risks           : risks of acting OR not acting
        metrics         : KPIs to track if recommendations are followed
        executive_summary: one paragraph for a C-suite audience
        cost            : session cost
    """
    print("Generating business strategy recommendations...")

    prompt = _build_strategy_prompt(question, scientist_result, engineer_result)
    raw_response = call_llm(prompt, system_instruction=STRATEGIST_SYSTEM)
    strategy = _parse_strategy(raw_response)

    return {
        "recommendations" : strategy,
        "cost"            : get_session_cost(),
    }


def _build_strategy_prompt(
    question: str,
    scientist_result: dict,
    engineer_result: dict,
) -> str:
    """Assemble context from previous agents into a strategy prompt."""
    stats        = scientist_result.get("stats", {})
    interp       = scientist_result.get("interpretation", {})
    confidence   = scientist_result.get("confidence_score", 0)
    conf_label   = scientist_result.get("confidence_label", "unknown")
    df           = engineer_result.get("dataframe")

    lines = [
        f"Business Question: {question}",
        f"Data Confidence: {confidence}/100 ({conf_label})",
        f"Sample Size: {stats.get('shape', ('?', '?'))[0]} rows",
        "",
    ]

    # Add the scientist's interpretation
    if interp.get("summary"):
        lines.append(f"Analysis Summary: {interp['summary']}")

    # Key findings
    if interp.get("key_findings"):
        lines.append("\nKey Findings:")
        for kf in interp["key_findings"]:
            lines.append(f"  - {kf['finding']} (confidence: {kf['confidence']})")
            lines.append(f"    Business impact: {kf['business_impact']}")

    # Auto-detected patterns
    if stats.get("findings"):
        lines.append("\nStatistical Patterns Detected:")
        for f in stats["findings"]:
            lines.append(f"  [{f['severity'].upper()}] {f['description']}")

    # Data snapshot
    if df is not None and not df.empty:
        lines.append(f"\nData Snapshot (first 5 rows):")
        lines.append(df.head(5).to_string(index=False))

    # Analyst caveats
    if interp.get("caveats"):
        lines.append("\nData Caveats:")
        for c in interp["caveats"]:
            lines.append(f"  ⚠ {c}")

    lines.append(f"""
Based on this analysis, provide strategic business recommendations.
Respond with a JSON object with this exact structure:
{{
  "executive_summary": "One paragraph (3-4 sentences) suitable for a CEO or board meeting",
  "recommendations": [
    {{
      "priority": 1,
      "action": "specific action to take",
      "rationale": "why this action, grounded in the data",
      "expected_impact": "what outcome to expect",
      "timeframe": "immediate|short-term|long-term"
    }}
  ],
  "risks": [
    {{
      "risk": "description of risk",
      "likelihood": "high|medium|low",
      "mitigation": "how to address it"
    }}
  ],
  "kpis_to_track": [
    {{
      "metric": "metric name",
      "target": "what good looks like",
      "frequency": "how often to measure"
    }}
  ],
  "confidence_caveat": "honest statement about data reliability and what additional data would improve confidence"
}}""")

    return "\n".join(lines)


def _parse_strategy(raw: str) -> dict:
    """Parse LLM JSON response, stripping markdown fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw_response": raw}