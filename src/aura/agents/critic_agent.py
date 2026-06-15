"""
src/aura/agents/critic_agent.py

The Critic Agent reviews the full pipeline output before the user sees it.
This is a control-flow agent — it can APPROVE or REVISE.

What makes this the "secret weapon":
- It's not appending commentary. It's a gatekeeper.
- In the LangGraph orchestrator (Stage 6), a REVISE verdict
  routes the workflow back to the Engineer/Scientist to retry.
- This creates a quality floor that generic multi-agent demos lack.

The Critic checks for:
1. Answer relevance — does the output actually answer the question?
2. Logical consistency — do the stats support the recommendations?
3. Confidence alignment — are strong claims made on weak data?
4. Completeness — is anything important missing?
"""

import json
from typing import Any

from aura.core.llm import call_llm, get_session_cost


CRITIC_SYSTEM = """You are a ruthless but fair quality reviewer for data analytics outputs.
Your job is to catch problems BEFORE the user sees the output.
You review the full pipeline: SQL results, statistical analysis, and business recommendations.
Be specific about what's wrong. Don't approve mediocre work.
Respond only in valid JSON."""


# Verdict constants — used by the orchestrator for routing
APPROVE = "APPROVE"
REVISE  = "REVISE"


def run_critic_agent(
    question: str,
    engineer_result: dict,
    scientist_result: dict,
    strategist_result: dict,
) -> dict[str, Any]:
    """
    Review the full pipeline output and return a verdict.

    Parameters
    ----------
    question           : original user question
    engineer_result    : output from run_engineer_agent()
    scientist_result   : output from run_scientist_agent()
    strategist_result  : output from run_strategist_agent()

    Returns
    -------
    dict with keys:
        verdict         : "APPROVE" or "REVISE"
        overall_score   : 0-100 quality score
        issues          : list of specific problems found
        approved_sections: which parts passed review
        revision_guidance: what to fix if verdict is REVISE
        cost            : session cost
    """
    print("Critic Agent reviewing pipeline output...")

    prompt = _build_critic_prompt(
        question, engineer_result, scientist_result, strategist_result
    )
    raw_response = call_llm(prompt, system_instruction=CRITIC_SYSTEM)
    review = _parse_review(raw_response)

    verdict = review.get("verdict", REVISE)
    score   = review.get("overall_score", 0)

    print(f"Critic verdict: {verdict} (score: {score}/100)")
    if verdict == REVISE:
        print("Issues found:")
        for issue in review.get("issues", []):
            print(f"  [{issue.get('severity', '?').upper()}] {issue.get('description', '')}")

    return {
        "verdict"           : verdict,
        "overall_score"     : score,
        "review"            : review,
        "cost"              : get_session_cost(),
    }


def _build_critic_prompt(
    question: str,
    engineer_result: dict,
    scientist_result: dict,
    strategist_result: dict,
) -> str:
    """Build a comprehensive review prompt covering all pipeline stages."""
    df           = engineer_result.get("dataframe")
    sql          = engineer_result.get("final_sql", "")
    attempts     = engineer_result.get("total_attempts", 1)
    confidence   = scientist_result.get("confidence_score", 0)
    interp       = scientist_result.get("interpretation", {})
    strategy     = strategist_result.get("recommendations", {})

    lines = [
        "You are reviewing a data analytics pipeline output. Be critical.",
        "",
        f"ORIGINAL QUESTION: {question}",
        "",
        "═" * 50,
        "STAGE 1 — SQL GENERATION",
        "═" * 50,
        f"SQL attempts needed: {attempts}",
        f"Final SQL:\n{sql}",
        f"Rows returned: {df.shape[0] if df is not None else 0}",
    ]

    if df is not None and not df.empty:
        lines.append(f"Result preview:\n{df.head(3).to_string(index=False)}")

    lines += [
        "",
        "═" * 50,
        "STAGE 2 — STATISTICAL ANALYSIS",
        "═" * 50,
        f"Confidence score: {confidence}/100",
        f"Analysis summary: {interp.get('summary', 'N/A')}",
    ]

    if interp.get("key_findings"):
        lines.append("Key findings:")
        for kf in interp["key_findings"]:
            lines.append(f"  - {kf['finding']} [{kf['confidence']} confidence]")

    if interp.get("caveats"):
        lines.append("Analyst caveats:")
        for c in interp["caveats"]:
            lines.append(f"  ⚠ {c}")

    lines += [
        "",
        "═" * 50,
        "STAGE 3 — BUSINESS RECOMMENDATIONS",
        "═" * 50,
        f"Executive summary: {strategy.get('executive_summary', 'N/A')}",
    ]

    if strategy.get("recommendations"):
        lines.append("Recommendations:")
        for rec in strategy["recommendations"][:3]:
            lines.append(f"  #{rec.get('priority', '?')}: {rec.get('action', '')}")

    lines.append(f"""
═══════════════════════════════════════════════════════
YOUR REVIEW CRITERIA
═══════════════════════════════════════════════════════
Check each of the following:

1. RELEVANCE: Does the SQL actually answer the question asked?
2. ACCURACY: Are the statistical findings correctly interpreted?
3. CONSISTENCY: Do recommendations align with confidence level?
   (Low confidence data should not support strong recommendations)
4. COMPLETENESS: Is anything important missing from the analysis?
5. HONESTY: Are data limitations clearly communicated?

Respond with a JSON object with this exact structure:
{{
  "verdict": "APPROVE or REVISE",
  "overall_score": <integer 0-100>,
  "issues": [
    {{
      "stage": "sql|statistics|strategy",
      "severity": "critical|major|minor",
      "description": "specific problem found"
    }}
  ],
  "approved_sections": ["list of sections that passed review"],
  "revision_guidance": "specific instructions for what to fix (empty string if APPROVE)",
  "praise": "what was done well (be specific)"
}}

APPROVE if overall_score >= 70 and no critical issues.
REVISE if overall_score < 70 or any critical issues exist.""")

    return "\n".join(lines)


def _parse_review(raw: str) -> dict:
    """Parse LLM JSON response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "verdict"       : REVISE,
            "overall_score" : 0,
            "issues"        : [{"severity": "critical", "description": f"Critic parse error: {e}"}],
            "parse_error"   : str(e),
        }