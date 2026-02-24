# FinAdvisor Bot 🇺🇦💸

FinAdvisor Bot — це персональний фінансовий AI-радник у Telegram, побудований за допомогою технології RAG (Retrieval-Augmented Generation). Бот надає точні та актуальні відповіді на фінансові питання, спираючись на контекст із офіційних документів (НБУ, банківських правил тощо).

## 🚀 Основні можливості

- **Контекстна база знань (RAG)**: Читає документи (`.txt`, `.md` тощо) і розділяє їх на фрагменти, з яких шукає найрелевантнішу інформацію для відповіді.
- **Довготривала пам'ять розмови**: Використовує вбудовану БД `SQLite` для історії діалогу з контекстом попередніх повідомлень.
- **Локальні векторні вбудовування**: Використовує модель `all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`) для перетворення тексту у вектори абсолютно безкоштовно та швидше.
- **Векторна база даних**: Зберігає всі вбудовування локально в `ChromaDB`.
- **Розумна LLM**: Інтеграція з Groq API (`llama-3.3-70b-versatile` або іншими) за допомогою `LangChain`. Швидкість генерації сягає понад 400+ токенів/с.
- **Асинхронний дизайн**: Всі завантаження, запити до баз даних і виклики API обробляються асинхронно через `aiosqlite`, `asyncio` та пули потоків, не блокуючи роботу інших користувачів у Telegram.
- **Захист від галюцинацій та Roleplay**: Збалансований промпт, який унеможливлює "втечу з рамки" бота та відсікає спроби користувача отримати інсайти щодо інвестування чи ринкових курсів.
- **Дотримання формату**: Всі відповіді коректно відформатовані під `Markdown v1` Telegram (жирний текст, списки тощо).

## 🧰 Технологічний стек

- **Мова**: Python 3.11+
- **Бібліотеки та платформи**: 
  - `python-telegram-bot` v20+ 
  - `langchain`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`
  - `chromadb`
  - `aiosqlite`, `sqlite3`
  - `pydantic`, `pydantic-settings`

## ⚙️ Перед початком

1. Зробіть клон репозиторію:
   ```bash
   git clone https://github.com/YourUsername/FinAdvisor_Bot.git
   cd FinAdvisor_Bot
   ```

2. Створіть та активуйте віртуальне середовище:
   ```bash
   python -m venv venv
   # На Windows:
   venv\Scripts\activate
   # На macOS/Linux:
   source venv/bin/activate
   ```

3. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Конфігурація (.env)

Перейменуйте файл `.env.example` (якщо є) або просто створіть новий файл `.env` у кореневій папці:

```env
TELEGRAM_BOT_TOKEN="ВАШ_ТОКЕН_З_BOTFATHER"
GROQ_API_KEY="ВАШ_КЛЮЧ_ВІД_GROQ"

# Зберігання
DB_PATH="data/memory.db"
CHROMA_PATH="data/chroma"

# Моделі
EMBEDDING_MODEL="all-MiniLM-L6-v2"
LLM_MODEL="llama-3.3-70b-versatile"
LLM_TEMPERATURE=0.2

# Параметри RAG і пам'яті
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_TOP_K=4
HISTORY_LIMIT=10
```

## 📚 Завантаження Бази Знань

Додайте свої документи (текстові інструкції НБУ, правила тощо) у папку `data/docs`. Далі, запустіть скрипт для парсингу та векторизації (потрібно зробити **1 раз** при оновленні бази знань):

```bash
python scripts/ingest_data.py
```

Ви побачите лог про те, що документи були успішно зачитані, розбиті на фрагменти і збережені у `ChromaDB`.

## 🎮 Запуск Бота (Локально)

Після успішної генерації бази знань просто запустіть `main.py`:

```bash
python main.py
```

## ☁️ Деплой на Railway (продакшен)

Бот підготовлений до безперебійного розміщення у хмарному сервісі Railway із прив'язаним GitHub-репозиторієм.

1. Увійдіть у [Railway.app](https://railway.app/).
2. Створіть новий проект (`New Project`) → `Deploy from GitHub repo` та виберіть цей репозиторій.
3. Railway автоматично розпізнає `Procfile` (`worker: python main.py`) та почне збирати середовище. Дайте йому декілька хвилин.
4. **Важливо**: В розділі **Variables** на Railway додайте такі змінні:
   - `TELEGRAM_BOT_TOKEN`
   - `GROQ_API_KEY`
5. Тепер збережіть налаштування. Railway автоматично перезавантажить застосунок. Бот працює!
  
> **Примітка щодо `ChromaDB` та SQLite**: Railway за замовчуванням запускає ефемерні контейнери. Будь-які зміни (база даних SQLite та зміни в `ChromaDB`) будуть скинуті при наступному деплої. Для зберігання додайте Railway **Volume** і підключіть до шляху `data/`.

## 📝 Структура проекту

```
📦FinAdvisor Bot
 ┣ 📂app
 ┃ ┣ 📂bot
 ┃ ┃ ┗ 📜handlers.py      # Команди (/start, /clear) та обробка повідомлень
 ┃ ┣ 📂db
 ┃ ┃ ┣ 📜crud.py          # Операції запису та зчитування історії і користувачів 
 ┃ ┃ ┣ 📜database.py      # Ініціалізація SQLite та WAL-режиму
 ┃ ┃ ┗ 📜models.py        # Pydantic моделі для БД
 ┃ ┣ 📂llm
 ┃ ┃ ┣ 📜chain.py         # Ланцюжок LangChain (RAG, Prompt, Відповідь)
 ┃ ┃ ┣ 📜model.py         # Ініціалізація ChatGroq
 ┃ ┃ ┗ 📜prompts.py       # Інструкції та промпти AI
 ┃ ┣ 📂rag
 ┃ ┃ ┣ 📜embeddings.py    # Модель HuggingFace
 ┃ ┃ ┣ 📜parser.py        # Завантаження та розбиття док. на фрагменти
 ┃ ┃ ┣ 📜retriever.py     # Пошук контексту по базі ChromaDB
 ┃ ┃ ┗ 📜vectorstore.py   # Конфігурація ChromaDB Client
 ┃ ┣ 📜config.py          # Перевірка та завантаження .env (Pydantic Settings)
 ┃ ┗ 📜logger.py          # Логування та його форматування
 ┣ 📂data
 ┃ ┣ 📂chroma             # Векторна БД (генерується)
 ┃ ┣ 📂docs               # Файли документації для RAG
 ┃ ┗ 📜memory.db          # SQLite база збережених повідомлень користувачів
 ┣ 📂scripts
 ┃ ┗ 📜ingest_data.py     # Парсер бази знань у ChromaDB
 ┣ 📜Procfile             # Команда запуску для Railway
 ┣ 📜runtime.txt          # Версія Python
 ┣ 📜main.py              # Точка входу в застосунок
 ┗ 📜requirements.txt     # Залежності Python
```
