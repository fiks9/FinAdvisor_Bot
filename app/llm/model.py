"""
app/llm/model.py — Groq LLM initialisation
===========================================
Returns a cached ChatGroq instance with the configured model.

Why Groq?
  Groq runs LLaMA 3 on custom LPU hardware — inference is 5–10x faster
  than standard GPU cloud APIs. Free tier gives ~14,400 requests/day
  with llama-3.3-70b-versatile, which is enough for a portfolio project.

Why temperature 0.2?
  Financial advice must be factual and precise, not creative.
  Low temperature (0.0–0.3) keeps the model grounded in the context
  we provide and reduces hallucination risk.
"""

import logging
from functools import lru_cache

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm(
    model:       str   = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    api_key:     str   = "",
) -> ChatGroq:
    """
    Return a cached ChatGroq LLM instance.

    @lru_cache ensures the client is created once — not on every message.
    Note: lru_cache requires hashable args, so api_key is passed as str.

    Args:
        model:       Groq model identifier. Must match what's available in
                     your Groq account at console.groq.com/docs/models
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = wild).
        api_key:     Groq API key. Pass cfg.GROQ_API_KEY from chain.py.

    Returns:
        LangChain-compatible ChatGroq ready for LCEL chaining.
    """
    logger.info("Initialising ChatGroq | model=%s | temperature=%s", model, temperature)

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        # max_tokens: None = use model default (~8192 for llama-3.3-70b)
        # streaming: False for now — can be enabled later for typing effect
        streaming=False,
    )

    logger.info("ChatGroq ready")
    return llm
