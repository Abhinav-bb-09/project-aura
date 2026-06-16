# Project Aura

A production-grade multi-agent analytics platform that turns natural language
questions into statistical insights and business recommendations — automatically.

## Demo

Ask: *"What are the top 5 products by total revenue?"*

Aura runs a 5-agent pipeline in seconds:
1. **Schema Agent** profiles your database automatically — no hardcoding
2. **Engineer Agent** writes SQL, executes it, and self-corrects on failure
3. **Scientist Agent** runs real statistical tests (Shapiro-Wilk, IQR, Pearson)
4. **Strategist Agent** translates findings to executive recommendations
5. **Critic Agent** reviews everything before you see it — APPROVE or REVISE

## What Makes This Different

| Feature | Generic Demo | Project Aura |
|---------|-------------|--------------|
| Schema handling | Hardcoded | Auto-discovered from any CSV/Excel/SQLite |
| SQL errors | Crashes | Self-corrects up to 3 times with reasoning logged |
| Confidence | Made up | Real p-values, sample size, outlier rates |
| Output review | None | Critic agent gates every response |
| Cost tracking | None | Per-query token usage and dollar cost |

## Stack

- **LangGraph** — agent orchestration and state management
- **Groq / Gemini** — LLM inference (switchable via .env)
- **SQLite + Pandas** — data layer
- **Streamlit + Plotly** — frontend
- **Scipy** — statistical testing (Shapiro-Wilk, Pearson, IQR)

## Setup

    git clone https://github.com/Abhinav-bb-09/project-aura.git
    cd project-aura
    conda create -n aura python=3.11 -y
    conda activate aura
    pip install -r requirements.txt
    pip install -e .
    cp .env.example .env
    # Add your GROQ_API_KEY to .env
    streamlit run src/aura/ui/streamlit_app.py

## Project Structure

    src/aura/
    ├── agents/        # LLM-powered reasoning units
    ├── tools/         # deterministic pandas/scipy logic
    ├── core/          # LLM gateway, state, caching
    └── graph/         # LangGraph orchestrator

## Author

Abhinav Sharma — MSBA, University of Illinois Urbana-Champaign (May 2026)

[LinkedIn](https://www.linkedin.com/in/abhinavsharmabb/) · [GitHub](https://github.com/Abhinav-bb-09)