"""
app/llm/chain.py — LCEL RAG chain assembly
===========================================
Ties together: retriever → history → prompt → LLM → output parser.

Public API:
  build_chain(cfg) → None       (call once in post_init)
  ask(user_id, question) → str  (call from Telegram handler)

LCEL data flow:
  user question
    ↓
  retrieve(question)          → context string  (RAG)
  get_history(user_id)        → list[Message]   (SQLite memory)
    ↓
  format into prompt          → ChatPromptValue
    ↓
  ChatGroq.invoke(prompt)     → AIMessage
    ↓
  StrOutputParser             → plain str
    ↓
  returned to handler

Why not use LangChain ConversationChain?
  ConversationChain stores history IN MEMORY — it resets on bot restart.
  We manage history ourselves in SQLite so it persists across restarts
  and multiple processes (e.g. scaling on Railway).
"""

import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.config import get_settings
from app.llm.model import get_llm
from app.llm.prompts import get_rag_prompt
from app.rag.retriever import retrieve, format_context
from app.db.crud import get_history

logger = logging.getLogger(__name__)

# Module-level chain, set by build_chain()
_chain = None


def build_chain() -> None:
    """
    Initialise the LLM and compile the LCEL chain.
    Called once inside post_init() in main.py.

    Why not build at module import?
      Building at import would load the LLM client before the config
      and logger are set up, making startup errors hard to debug.
    """
    global _chain

    cfg = get_settings()

    llm    = get_llm(
        model=cfg.LLM_MODEL,
        temperature=cfg.LLM_TEMPERATURE,
        api_key=cfg.GROQ_API_KEY,
    )
    prompt = get_rag_prompt()
    parser = StrOutputParser()

    # ── Chain definition (LCEL pipe syntax) ────────────────────────────
    # Input dict: {"context": str, "history": list[BaseMessage], "question": str}
    # Each step transforms or passes through the dict.
    _chain = prompt | llm | parser

    logger.info("LCEL RAG chain built successfully")


async def ask(user_id: int, question: str) -> str:
    """
    Main entry point called by the Telegram message handler.

    Steps:
      1. Retrieve relevant document chunks from ChromaDB
      2. Load last N messages from SQLite as LangChain Message objects
      3. Invoke the LCEL chain with the assembled input dict
      4. Return the LLM's response as a plain string

    Args:
        user_id:  Telegram user ID (used to fetch their specific history)
        question: The user's current message text

    Returns:
        LLM response string, ready to send back to Telegram.
    """
    if _chain is None:
        raise RuntimeError("Chain not built. Call build_chain() in post_init().")

    cfg = get_settings()

    # ── 1. RAG: retrieve relevant document chunks ──────────────────────
    chunks  = retrieve(question, k=cfg.RETRIEVER_TOP_K)
    context = format_context(chunks)

    # ── 2. Memory: load conversation history from SQLite ──────────────
    raw_history = await get_history(user_id, limit=cfg.HISTORY_LIMIT)

    # Convert our Message models → LangChain message objects
    # LangChain prompt expects HumanMessage / AIMessage types
    history_messages = []
    for msg in raw_history:
        if msg.role == "user":
            history_messages.append(HumanMessage(content=msg.content))
        else:
            history_messages.append(AIMessage(content=msg.content))

    logger.info(
        "Invoking chain | user=%s | history=%d msgs | context_chunks=%d",
        user_id,
        len(history_messages),
        len(chunks),
    )

    # ── 3. Invoke chain ────────────────────────────────────────────────
    # _chain is synchronous (Groq client uses httpx under the hood).
    # We call it directly — PTB's handler runs in a thread pool executor,
    # so this won't block the event loop for other users.
    response: str = _chain.invoke({
        "context":  context,
        "history":  history_messages,
        "question": question,
    })

    logger.info("Chain response ready | user=%s | length=%d chars", user_id, len(response))
    return response
