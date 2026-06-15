"""
src/aura/core/llm.py

Unified LLM gateway supporting Groq and Gemini.
Switch providers by changing ACTIVE_LLM in .env — no agent code changes needed.
"""

from groq import Groq
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from aura.config import (
    ACTIVE_LLM,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    COST_PER_M_INPUT_TOKENS,
    COST_PER_M_OUTPUT_TOKENS,
    MAX_TOKENS_PER_CALL,
    TEMPERATURE,
)

# ── Session cost tracker ──────────────────────────────────────
_session_cost = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_usd": 0.0,
    "call_count": 0,
}


def get_session_cost() -> dict:
    return _session_cost.copy()


def reset_session_cost() -> None:
    for key in _session_cost:
        _session_cost[key] = 0 if key != "total_usd" else 0.0


# ── Clients (initialized once at import time) ─────────────────
_groq_client   = Groq(api_key=GROQ_API_KEY)
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Public entry point ────────────────────────────────────────

def call_llm(
    prompt: str,
    system_instruction: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Make a single LLM call using whichever provider is active.
    Agents call this — they never touch Groq or Gemini SDKs directly.
    """
    if ACTIVE_LLM == "groq":
        return _call_groq(prompt, system_instruction, model, temperature, max_tokens)
    else:
        return _call_gemini(prompt, system_instruction, model, temperature, max_tokens)


# ── Groq implementation ───────────────────────────────────────

def _call_groq(
    prompt: str,
    system_instruction: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    """Call Groq API. Uses OpenAI-compatible chat format."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    # gemma2-9b-it has separate quota — use as fallback
    active_model = model or GROQ_MODEL
    response = _groq_client.chat.completions.create(
        model=active_model,
        messages=messages,
        temperature=temperature or TEMPERATURE,
        max_tokens=max_tokens or MAX_TOKENS_PER_CALL,
    )

    # Track usage
    usage = response.usage
    if usage:
        input_t  = usage.prompt_tokens or 0
        output_t = usage.completion_tokens or 0
        cost = (input_t / 1_000_000 * COST_PER_M_INPUT_TOKENS +
                output_t / 1_000_000 * COST_PER_M_OUTPUT_TOKENS)
        _session_cost["input_tokens"]  += input_t
        _session_cost["output_tokens"] += output_t
        _session_cost["total_usd"]     += cost
        _session_cost["call_count"]    += 1

    return response.choices[0].message.content


# ── Gemini implementation ─────────────────────────────────────

@retry(
    retry=retry_if_exception_type(ServerError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_gemini(
    prompt: str,
    system_instruction: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    """Call Gemini API. Only retries on ServerError (503), not quota errors."""
    config_params = types.GenerateContentConfig(
        temperature=temperature or TEMPERATURE,
        max_output_tokens=max_tokens or MAX_TOKENS_PER_CALL,
        system_instruction=system_instruction,
    )

    response = _gemini_client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=prompt,
        config=config_params,
    )

    usage = response.usage_metadata
    if usage:
        input_t  = usage.prompt_token_count or 0
        output_t = usage.candidates_token_count or 0
        cost = (input_t / 1_000_000 * COST_PER_M_INPUT_TOKENS +
                output_t / 1_000_000 * COST_PER_M_OUTPUT_TOKENS)
        _session_cost["input_tokens"]  += input_t
        _session_cost["output_tokens"] += output_t
        _session_cost["total_usd"]     += cost
        _session_cost["call_count"]    += 1

    return response.text