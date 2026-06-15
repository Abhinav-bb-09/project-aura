"""
src/aura/config.py

Central configuration for the entire Aura system.
ACTIVE_LLM controls which provider is used — switch in .env, not here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Active LLM provider ───────────────────────────────────────
# Set ACTIVE_LLM=groq or ACTIVE_LLM=gemini in .env
ACTIVE_LLM = os.getenv("ACTIVE_LLM", "groq")

# ── Groq ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq free tier pricing (effectively $0 for free tier)
GROQ_COST_PER_M_INPUT_TOKENS  = 0.0
GROQ_COST_PER_M_OUTPUT_TOKENS = 0.0

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

GEMINI_COST_PER_M_INPUT_TOKENS  = 0.075
GEMINI_COST_PER_M_OUTPUT_TOKENS = 0.30

# ── Active model cost (resolves based on ACTIVE_LLM) ─────────
if ACTIVE_LLM == "groq":
    COST_PER_M_INPUT_TOKENS  = GROQ_COST_PER_M_INPUT_TOKENS
    COST_PER_M_OUTPUT_TOKENS = GROQ_COST_PER_M_OUTPUT_TOKENS
else:
    COST_PER_M_INPUT_TOKENS  = GEMINI_COST_PER_M_INPUT_TOKENS
    COST_PER_M_OUTPUT_TOKENS = GEMINI_COST_PER_M_OUTPUT_TOKENS

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ── Agent limits ─────────────────────────────────────────────
MAX_SQL_RETRIES      = 3
MAX_TOKENS_PER_CALL  = 8192
TEMPERATURE          = 0.2

# ── Confidence scoring ───────────────────────────────────────
MIN_ROWS_FOR_HIGH_CONFIDENCE = 100
HIGH_CONFIDENCE_THRESHOLD    = 75
LOW_CONFIDENCE_THRESHOLD     = 40