"""
src/aura/graph/orchestrator.py

The LangGraph orchestrator that wires all agents into a stateful graph.

Graph structure:
  schema_node → engineer_node → scientist_node → strategist_node → critic_node
                                                                        ↓
                                              (APPROVE) → output_node
                                              (REVISE)  → engineer_node (retry)
                                              (max_revisions hit) → output_node

The Critic's APPROVE/REVISE decision is the only conditional edge.
Everything else is a straight pipeline.
"""

import pandas as pd
from typing import Literal

from langgraph.graph import StateGraph, END

from aura.core.state import AuraState
from aura.core.llm import reset_session_cost, get_session_cost
from aura.agents.schema_agent import run_schema_agent
from aura.agents.engineer_agent import run_engineer_agent
from aura.agents.scientist_agent import run_scientist_agent
from aura.agents.strategist_agent import run_strategist_agent
from aura.agents.critic_agent import run_critic_agent, APPROVE


# ─────────────────────────────────────────────
# NODE FUNCTIONS
# Each node takes AuraState, returns a dict of updates.
# LangGraph merges the returned dict back into the state.
# ─────────────────────────────────────────────

def schema_node(state: AuraState) -> dict:
    """Run the Schema Discovery Agent."""
    print(f"\n{'='*50}")
    print("NODE: Schema Discovery")
    print(f"{'='*50}")
    try:
        result = run_schema_agent(state.source_path)
        return {
            "schema_profile"   : result["profile"],
            "schema_enrichment": result["enrichment"],
        }
    except Exception as e:
        return {"error": f"Schema node failed: {e}"}


def engineer_node(state: AuraState) -> dict:
    """Run the Engineer Agent (self-correcting SQL loop)."""
    print(f"\n{'='*50}")
    print(f"NODE: Engineer (revision #{state.revision_count})")
    print(f"{'='*50}")
    try:
        result = run_engineer_agent(
            question=state.question,
            db_path=state.db_path,
            schema_profile=state.schema_profile,
        )
        return {
            "final_sql"      : result["final_sql"],
            "sql_result"     : result["result"],
            "sql_attempts"   : result["total_attempts"],
            "engineer_success": result["success"],
            # Store dataframe inside sql_result for downstream agents
            "_dataframe"     : result["dataframe"],
        }
    except Exception as e:
        return {"error": f"Engineer node failed: {e}", "engineer_success": False}


def scientist_node(state: AuraState) -> dict:
    """Run the Scientist Agent."""
    print(f"\n{'='*50}")
    print("NODE: Scientist")
    print(f"{'='*50}")
    try:
        # Reconstruct DataFrame from sql_result
        df = pd.DataFrame(state.sql_result.get("data", []))
        if df.empty:
            return {"error": "No data to analyze"}

        result = run_scientist_agent(
            dataframe=df,
            question=state.question,
            sql=state.final_sql,
        )
        return {
            "statistical_analysis": result["stats"],
            "interpretation"      : result["interpretation"],
            "confidence_score"    : result["confidence_score"],
            "confidence_label"    : result["confidence_label"],
        }
    except Exception as e:
        return {"error": f"Scientist node failed: {e}"}


def strategist_node(state: AuraState) -> dict:
    """Run the Strategist Agent."""
    print(f"\n{'='*50}")
    print("NODE: Strategist")
    print(f"{'='*50}")
    try:
        df = pd.DataFrame(state.sql_result.get("data", []))

        # Reconstruct the dicts that run_strategist_agent expects
        scientist_result = {
            "stats"           : state.statistical_analysis,
            "interpretation"  : state.interpretation,
            "confidence_score": state.confidence_score,
            "confidence_label": state.confidence_label,
        }
        engineer_result = {
            "dataframe" : df,
            "final_sql" : state.final_sql,
        }

        result = run_strategist_agent(
            question=state.question,
            scientist_result=scientist_result,
            engineer_result=engineer_result,
        )
        return {"strategy": result["recommendations"]}
    except Exception as e:
        return {"error": f"Strategist node failed: {e}"}


def critic_node(state: AuraState) -> dict:
    """Run the Critic Agent."""
    print(f"\n{'='*50}")
    print("NODE: Critic")
    print(f"{'='*50}")
    try:
        df = pd.DataFrame(state.sql_result.get("data", []))

        engineer_result  = {"dataframe": df, "final_sql": state.final_sql,
                            "total_attempts": state.sql_attempts}
        scientist_result = {"confidence_score": state.confidence_score,
                            "confidence_label": state.confidence_label,
                            "interpretation": state.interpretation,
                            "stats": state.statistical_analysis}
        strategist_result = {"recommendations": state.strategy}

        result = run_critic_agent(
            question=state.question,
            engineer_result=engineer_result,
            scientist_result=scientist_result,
            strategist_result=strategist_result,
        )
        return {
            "critic_verdict": result["verdict"],
            "critic_score"  : result["overall_score"],
            "critic_review" : result["review"],
        }
    except Exception as e:
        return {"error": f"Critic node failed: {e}", "critic_verdict": "APPROVE"}


def output_node(state: AuraState) -> dict:
    """Assemble the final output dict."""
    print(f"\n{'='*50}")
    print("NODE: Output Assembly")
    print(f"{'='*50}")
    cost = get_session_cost()
    final = {
        "question"        : state.question,
        "sql"             : state.final_sql,
        "confidence_score": state.confidence_score,
        "confidence_label": state.confidence_label,
        "interpretation"  : state.interpretation,
        "strategy"        : state.strategy,
        "critic_verdict"  : state.critic_verdict,
        "critic_score"    : state.critic_score,
        "revision_count"  : state.revision_count,
        "total_cost_usd"  : cost["total_usd"],
    }
    return {"final_output": final, "total_cost_usd": cost["total_usd"]}


# ─────────────────────────────────────────────
# ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_critic(state: AuraState) -> Literal["engineer_node", "output_node"]:
    """
    The only conditional edge in the graph.
    APPROVE → output_node (done)
    REVISE  → engineer_node (retry), unless max_revisions hit
    """
    if state.critic_verdict == APPROVE:
        print("\n✓ Critic APPROVED — sending to output")
        return "output_node"

    if state.revision_count >= state.max_revisions:
        print(f"\n⚠ Max revisions ({state.max_revisions}) reached — forcing output")
        return "output_node"

    print(f"\n↩ Critic said REVISE — routing back (revision {state.revision_count + 1})")
    return "engineer_node"


def increment_revision(state: AuraState) -> dict:
    """Increment revision counter when looping back."""
    return {"revision_count": state.revision_count + 1}


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine.
    Call this once and reuse the compiled graph.
    """
    graph = StateGraph(AuraState)

    # ── Register nodes ────────────────────────────────────────
    graph.add_node("schema_node",     schema_node)
    graph.add_node("engineer_node",   engineer_node)
    graph.add_node("scientist_node",  scientist_node)
    graph.add_node("strategist_node", strategist_node)
    graph.add_node("critic_node",     critic_node)
    graph.add_node("output_node",     output_node)
    graph.add_node("increment_revision", increment_revision)

    # ── Entry point ───────────────────────────────────────────
    graph.set_entry_point("schema_node")

    # ── Straight pipeline edges ───────────────────────────────
    graph.add_edge("schema_node",     "engineer_node")
    graph.add_edge("engineer_node",   "scientist_node")
    graph.add_edge("scientist_node",  "strategist_node")
    graph.add_edge("strategist_node", "critic_node")
    graph.add_edge("output_node",     END)

    # ── Conditional edge: Critic's decision ───────────────────
    graph.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "engineer_node": "increment_revision",
            "output_node"  : "output_node",
        }
    )
    graph.add_edge("increment_revision", "engineer_node")

    return graph.compile()


# ─────────────────────────────────────────────
# PUBLIC RUN FUNCTION
# ─────────────────────────────────────────────

def run_aura(question: str, source_path: str, db_path: str) -> dict:
    """
    Run the full Aura pipeline for a user question.

    Parameters
    ----------
    question    : natural language question
    source_path : path to dataset file (for schema profiling)
    db_path     : path to SQLite database (for SQL execution)

    Returns
    -------
    The final_output dict from the output_node
    """
    reset_session_cost()

    graph = build_graph()

    initial_state = AuraState(
        question=question,
        db_path=db_path,
        source_path=source_path,
    )

    final_state = graph.invoke(initial_state)
    return final_state["final_output"]