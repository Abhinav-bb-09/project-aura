"""
src/aura/core/state.py

The shared state object that flows through the LangGraph graph.
Every agent reads from this and writes back to it.

Using Pydantic for validation means we catch data contract violations
at agent boundaries — not silently downstream.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class AuraState(BaseModel):
    """
    Complete state for one user query through the Aura pipeline.
    LangGraph passes this between nodes, each agent updates its section.
    """

    # ── Input ─────────────────────────────────────────────────
    question   : str = ""
    db_path    : str = ""
    source_path: str = ""

    # ── Stage 1: Schema Agent ─────────────────────────────────
    schema_profile    : dict[str, Any] = Field(default_factory=dict)
    schema_enrichment : dict[str, Any] = Field(default_factory=dict)

    # ── Stage 2: Engineer Agent ───────────────────────────────
    final_sql         : str = ""
    sql_result        : dict[str, Any] = Field(default_factory=dict)
    sql_attempts      : int = 0
    engineer_success  : bool = False

    # ── Stage 3: Scientist Agent ──────────────────────────────
    statistical_analysis : dict[str, Any] = Field(default_factory=dict)
    interpretation       : dict[str, Any] = Field(default_factory=dict)
    confidence_score     : int = 0
    confidence_label     : str = ""

    # ── Stage 4: Strategist Agent ─────────────────────────────
    strategy : dict[str, Any] = Field(default_factory=dict)

    # ── Stage 5: Critic Agent ─────────────────────────────────
    critic_verdict  : str = ""   # "APPROVE" or "REVISE"
    critic_score    : int = 0
    critic_review   : dict[str, Any] = Field(default_factory=dict)
    revision_count  : int = 0    # how many times we've been through the loop
    max_revisions   : int = 2    # safety limit — never loop more than twice

    # ── Output ────────────────────────────────────────────────
    final_output      : dict[str, Any] = Field(default_factory=dict)
    error             : str = ""
    total_cost_usd    : float = 0.0

    class Config:
        # Allow arbitrary types (pandas DataFrame stored in sql_result)
        arbitrary_types_allowed = True