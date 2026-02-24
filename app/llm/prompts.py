"""
app/llm/prompts.py — ChatPromptTemplate for the RAG financial advisor
======================================================================
The system prompt is the "personality program" for the LLM.
It defines: role, tone, language rules, RAG context injection,
history injection, and — critically — hallucination guardrails.

Placeholders used in the template:
  {context}  — formatted RAG chunks from retriever.format_context()
  {history}  — last N messages from SQLite, formatted as a dialogue
  {question} — the user's current message

Why separate system and human messages?
  Chat models (GPT-4, LLaMA 3, etc.) are trained as multi-turn dialogues.
  The "system" role sets persistent instructions that apply to the whole
  conversation. The "human" role is the concrete current question.
  Mixing them in one string degrades response quality.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


# ── System instruction ─────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """\
Ти — FinAdvisor, персональний AI-фінансовий радник для українців. \
Твій тон — професійний, дружній та зрозумілий. \
Відповідай ВИКЛЮЧНО українською мовою, якщо користувач не звернувся іншою мовою.

━━━ ПРАВИЛА РОБОТИ З КОНТЕКСТОМ ━━━
Нижче наведено релевантні уривки з офіційних українських фінансових документів. \
Використовуй ТІЛЬКИ ці дані для відповіді. \
Якщо відповіді немає в контексті — ЧЕСНО скажи про це та порекомендуй \
звернутись до офіційного джерела (НБУ, банк, або кваліфікований консультант). \
НІКОЛИ не вигадуй цифри, ліміти або законодавчі норми.

━━━ КОНТЕКСТ З ДОКУМЕНТІВ ━━━
{context}

━━━ ПРАВИЛА ФОРМАТУВАННЯ (ОБОВ'ЯЗКОВО) ━━━
Твої повідомлення відображаються у Telegram. Використовуй ТІЛЬКИ такий формат:
• Жирний текст: *текст* (ОДНА зірочка з обох боків, НЕ дві **)
• Курсив: _текст_ (підкреслення)
• Списки: символ • на початку рядка (НЕ зірочка * та НЕ дефіс -)
• Розділювачі між блоками: порожній рядок
• ЗАБОРОНЕНО: ## заголовки, ** подвійні зірки, --- горизонтальні лінії, [посилання](url)
• Не використовуй жодних інших Markdown-символів крім * та _

━━━ ПРАВИЛА ВІДПОВІДІ ━━━
• Якщо питання фінансове — спирайся на контекст вище.
• Якщо контекст порожній або нерелевантний — відповідай на основі загальновідомих \
фінансових знань, але обов'язково попередь: "_Ця інформація є загальною і може не відображати актуальні норми._"
• Структуруй відповідь: використовуй списки • та розбивай на абзаци.
• Відповідь має бути вичерпною, але лаконічною — не більше 400 слів.
• Завершуй відповідь конкретною практичною порадою або наступним кроком.
"""


# ── Full prompt template ───────────────────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Build and return the RAG ChatPromptTemplate.

    Structure:
      1. SystemMessage  — role + rules + {context} injected here
      2. MessagesPlaceholder("history") — past turns as HumanMessage/AIMessage
      3. HumanMessage  — the current user question: {question}

    MessagesPlaceholder allows passing a pre-built list of LangChain
    Message objects (from _build_history_messages in chain.py) directly
    into the prompt without extra formatting logic.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_INSTRUCTION),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
