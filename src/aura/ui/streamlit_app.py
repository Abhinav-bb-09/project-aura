"""
src/aura/ui/streamlit_app.py

Project Aura — Streamlit UI
Run with: streamlit run src/aura/ui/streamlit_app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aura.graph.orchestrator import run_aura, build_graph
from aura.agents.schema_agent import run_schema_agent
from aura.tools.profiler import profile_dataset
from aura.core.llm import reset_session_cost

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Project Aura",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Confidence badges */
    .badge-green {
        background: #0d3320; color: #22c55e;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600;
        border: 1px solid #22c55e;
    }
    .badge-yellow {
        background: #2d2200; color: #eab308;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600;
        border: 1px solid #eab308;
    }
    .badge-red {
        background: #2d0a0a; color: #ef4444;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600;
        border: 1px solid #ef4444;
    }

    /* Metric cards */
    .metric-card {
        background: #0f1117;
        border: 1px solid #1e2030;
        border-radius: 8px;
        padding: 16px 20px;
    }

    /* SQL block */
    .sql-block {
        background: #0d0d0d;
        border: 1px solid #1e2030;
        border-left: 3px solid #6366f1;
        border-radius: 4px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #a5b4fc;
        white-space: pre-wrap;
    }

    /* Section headers */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 8px;
    }

    /* Verdict banner */
    .verdict-approve {
        background: #0d3320;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px 20px;
        color: #22c55e;
        font-weight: 600;
    }
    .verdict-revise {
        background: #2d0a0a;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 20px;
        color: #ef4444;
        font-weight: 600;
    }

    /* Finding cards */
    .finding-high {
        border-left: 3px solid #ef4444;
        padding: 8px 12px;
        margin: 4px 0;
        background: #1a0a0a;
        border-radius: 0 4px 4px 0;
        font-size: 14px;
    }
    .finding-medium {
        border-left: 3px solid #eab308;
        padding: 8px 12px;
        margin: 4px 0;
        background: #1a1400;
        border-radius: 0 4px 4px 0;
        font-size: 14px;
    }
    .finding-low {
        border-left: 3px solid #6366f1;
        padding: 8px 12px;
        margin: 4px 0;
        background: #0d0d1a;
        border-radius: 0 4px 4px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⬡ Project Aura")
    st.markdown("*Multi-agent analytics platform*")
    st.divider()

    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose input",
        ["Use sample database (Northwind)", "Upload your own file"],
        label_visibility="collapsed",
    )

    db_path = None
    source_path = None

    if data_source == "Use sample database (Northwind)":
        # Resolve path relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        db_path     = str(project_root / "data/samples/northwind.db")
        source_path = db_path
        st.success("Northwind database loaded")
        st.caption("29 tables · e-commerce orders dataset")

    else:
        uploaded = st.file_uploader(
            "Upload CSV, Excel, or SQLite",
            type=["csv", "xlsx", "xls", "db", "sqlite"],
        )
        if uploaded:
            save_dir = Path(__file__).parent.parent.parent / "data/uploads"
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / uploaded.name
            save_path.write_bytes(uploaded.read())
            source_path = str(save_path)
            # For non-SQLite files, we use the same path for both
            db_path = source_path if uploaded.name.endswith((".db", ".sqlite")) else None
            st.success(f"Uploaded: {uploaded.name}")

    st.divider()

    # ── Sample questions ──────────────────────────────────────
    st.markdown("### Sample Questions")
    sample_questions = [
        "What are the top 5 products by total revenue?",
        "Show total sales by employee and their country",
        "Which customers have placed the most orders?",
        "What is the average order value by shipping country?",
        "Show monthly revenue trend",
    ]
    selected_sample = st.selectbox(
        "Pick one or write your own below",
        [""] + sample_questions,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Cost meter ────────────────────────────────────────────
    st.markdown("### Session Cost")
    if "total_cost" in st.session_state:
        st.metric("Total", f"${st.session_state.total_cost:.6f}")
        st.metric("API Calls", st.session_state.get("call_count", 0))
    else:
        st.caption("Run a query to see cost")

    st.divider()
    st.caption("Built with LangGraph · Groq · SQLite")
    st.caption("github.com/Abhinav-bb-09/project-aura")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown("## Ask your data anything")
st.markdown(
    "Natural language → SQL → Statistics → Strategy → Critic review",
)

# ── Question input ────────────────────────────────────────────
question = st.text_input(
    "Your question",
    value=selected_sample if selected_sample else "",
    placeholder="e.g. What are the top 5 products by total revenue?",
    label_visibility="collapsed",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("Analyze →", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────

if run_btn and question and source_path:
    if not db_path:
        st.error("SQLite database required for SQL queries. Upload a .db or .sqlite file.")
        st.stop()

    reset_session_cost()

    # Progress display
    progress_placeholder = st.empty()
    status_placeholder   = st.empty()

    stages = [
        ("⬡ Schema Discovery",    0.15),
        ("⚙ SQL Generation",       0.35),
        ("📊 Statistical Analysis", 0.55),
        ("💡 Strategy Generation",  0.75),
        ("🔍 Critic Review",        0.90),
        ("✓ Assembling output",    1.00),
    ]

    with st.spinner(""):
        for stage_name, progress in stages:
            progress_placeholder.progress(progress, text=stage_name)
            status_placeholder.caption(f"Running: {stage_name}")

        try:
            result = run_aura(
                question=question,
                source_path=source_path,
                db_path=db_path,
            )
            st.session_state["result"]     = result
            st.session_state["total_cost"] = result.get("total_cost_usd", 0.0)
            progress_placeholder.empty()
            status_placeholder.empty()

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"Pipeline error: {e}")
            st.stop()

elif run_btn and not source_path:
    st.warning("Please select or upload a data source first.")


# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()

    # ── Top metrics row ───────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    conf_score = result.get("confidence_score", 0)
    conf_label = result.get("confidence_label", "red")
    badge_class = f"badge-{conf_label}"

    with m1:
        st.markdown('<p class="section-label">Confidence</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="{badge_class}">{conf_score}/100</span>',
            unsafe_allow_html=True,
        )

    with m2:
        critic_verdict = result.get("critic_verdict", "—")
        verdict_class  = "verdict-approve" if critic_verdict == "APPROVE" else "verdict-revise"
        st.markdown('<p class="section-label">Critic Verdict</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="{verdict_class}">{critic_verdict}</span>',
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown('<p class="section-label">Revisions</p>', unsafe_allow_html=True)
        st.metric("", result.get("revision_count", 0), label_visibility="collapsed")

    with m4:
        st.markdown('<p class="section-label">Cost</p>', unsafe_allow_html=True)
        st.metric("", f"${result.get('total_cost_usd', 0):.6f}", label_visibility="collapsed")

    st.divider()

    # ── Main content tabs ─────────────────────────────────────
    tab_answer, tab_sql, tab_stats, tab_strategy, tab_critic = st.tabs([
        "Answer", "SQL", "Statistics", "Strategy", "Critic Review"
    ])

    # ── Tab 1: Answer ─────────────────────────────────────────
    with tab_answer:
        interp = result.get("interpretation", {})

        st.markdown("#### Summary")
        st.write(interp.get("summary", "No summary available."))

        key_findings = interp.get("key_findings", [])
        if key_findings:
            st.markdown("#### Key Findings")
            for kf in key_findings:
                conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                    kf.get("confidence", "low"), "⚪"
                )
                with st.expander(f"{conf_color} {kf.get('finding', '')}"):
                    st.write(f"**Business impact:** {kf.get('business_impact', '')}")
                    st.caption(f"Confidence: {kf.get('confidence', '').upper()}")

        caveats = interp.get("caveats", [])
        if caveats:
            st.markdown("#### Caveats")
            for c in caveats:
                st.warning(c)

    # ── Tab 2: SQL ────────────────────────────────────────────
    with tab_sql:
        st.markdown("#### Generated SQL")
        sql = result.get("sql", "")
        st.markdown(
            f'<div class="sql-block">{sql}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Query executed successfully.")

    # ── Tab 3: Statistics ─────────────────────────────────────
    with tab_stats:
        st.markdown("#### Confidence Score Breakdown")
        conf_factors = result.get("statistical_analysis", {}).get("confidence_factors", [])

        # Build gauge chart for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 40],  "color": "#2d0a0a"},
                    {"range": [40, 75], "color": "#2d2200"},
                    {"range": [75, 100],"color": "#0d3320"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": conf_score,
                },
            },
            title={"text": "Confidence Score"},
        ))
        fig_gauge.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if conf_factors:
            st.markdown("**Score factors:**")
            for f in conf_factors:
                st.caption(f)

        # Statistical findings
        findings = result.get("statistical_analysis", {}).get("findings", [])
        if findings:
            st.markdown("#### Detected Patterns")
            for f in findings:
                severity = f.get("severity", "low")
                css_class = f"finding-{severity}"
                st.markdown(
                    f'<div class="{css_class}">{f["description"]}</div>',
                    unsafe_allow_html=True,
                )

        # Numeric stats table
        num_stats = result.get("statistical_analysis", {}).get("numeric_stats", {})
        if num_stats:
            st.markdown("#### Numeric Column Statistics")
            stats_df = pd.DataFrame(num_stats).T
            display_cols = [c for c in ["mean", "median", "std", "min", "max", "skewness"] if c in stats_df.columns]
            st.dataframe(stats_df[display_cols].round(2), use_container_width=True)

    # ── Tab 4: Strategy ───────────────────────────────────────
    with tab_strategy:
        strategy = result.get("strategy", {})

        if strategy.get("executive_summary"):
            st.markdown("#### Executive Summary")
            st.info(strategy["executive_summary"])

        recs = strategy.get("recommendations", [])
        if recs:
            st.markdown("#### Prioritized Recommendations")
            for rec in recs:
                timeframe_color = {
                    "immediate": "🔴",
                    "short-term": "🟡",
                    "long-term": "🟢",
                }.get(rec.get("timeframe", ""), "⚪")

                with st.expander(
                    f"{timeframe_color} #{rec.get('priority', '?')} — {rec.get('action', '')}"
                ):
                    st.write(f"**Rationale:** {rec.get('rationale', '')}")
                    st.write(f"**Expected impact:** {rec.get('expected_impact', '')}")
                    st.caption(f"Timeframe: {rec.get('timeframe', '').upper()}")

        kpis = strategy.get("kpis_to_track", [])
        if kpis:
            st.markdown("#### KPIs to Track")
            kpi_df = pd.DataFrame(kpis)
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        if strategy.get("confidence_caveat"):
            st.markdown("#### Data Caveat")
            st.warning(strategy["confidence_caveat"])

    # ── Tab 5: Critic Review ──────────────────────────────────
    with tab_critic:
        critic_review = result.get("critic_review", {})

        verdict = result.get("critic_verdict", "—")
        if verdict == "APPROVE":
            st.success(f"✓ APPROVED — Pipeline output passed quality review")
        else:
            st.error(f"✗ REVISE — Issues found in pipeline output")

        issues = critic_review.get("issues", [])
        if issues:
            st.markdown("#### Issues Found")
            for issue in issues:
                sev = issue.get("severity", "minor")
                if sev == "critical":
                    st.error(f"**[{sev.upper()}]** {issue.get('description', '')}")
                elif sev == "major":
                    st.warning(f"**[{sev.upper()}]** {issue.get('description', '')}")
                else:
                    st.info(f"**[{sev.upper()}]** {issue.get('description', '')}")

        approved = critic_review.get("approved_sections", [])
        if approved:
            st.markdown("#### Approved Sections")
            for s in approved:
                st.success(f"✓ {s}")

        if critic_review.get("praise"):
            st.markdown("#### What Worked Well")
            st.write(critic_review["praise"])

        if critic_review.get("revision_guidance"):
            st.markdown("#### Revision Guidance")
            st.write(critic_review["revision_guidance"])

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**⬡ Auto Schema Discovery**")
        st.caption("Upload any CSV, Excel, or SQLite file. Aura profiles every column automatically.")
    with c2:
        st.markdown("**⚙ Self-Correcting SQL**")
        st.caption("If the first query fails, the Engineer agent reads the error and rewrites it.")
    with c3:
        st.markdown("**🔍 Critic Review**")
        st.caption("Every output is reviewed by a Critic agent before you see it.")