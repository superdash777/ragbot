"""
Карьерный ассистент — Agentic RAG Telegram-бот
Деплой-версия: запуск через python career_bot.py
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional
from collections import defaultdict, Counter

import numpy as np
import telebot
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

# ─── Секреты ───
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# ─── Логирование ───
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("career_bot")

# ─── Параметры ───
COLLECTION_NAME = "career_assistant_e5"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
VECTOR_SIZE = 768
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
TOP_K = 5
MAX_AGENT_ITERATIONS = 8
MAX_CLARIFICATIONS = 2
MAX_HISTORY_PAIRS = 8
DATA_FILE = "text2_expanded.txt"

# ─── Клиенты ───
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

logger.info("Загрузка эмбеддера %s ...", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
logger.info("Эмбеддер загружен.")


# ═══════════════════════════════════════
# ИНДЕКСАЦИЯ
# ═══════════════════════════════════════

def index_documents(filepath: str) -> int:
    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    full_text = documents[0].text
    logger.info("Документов: %d", len(documents))

    roles = []
    for match in re.finditer(r"^(Навыки .+)$", full_text, re.MULTILINE):
        roles.append({"name": match.group(1).strip(), "start": match.start()})
    logger.info("Ролей: %d", len(roles))

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    texts = [n.text for n in nodes]
    logger.info("Чанков: %d", len(texts))

    chunk_roles = []
    for node in nodes:
        chunk_start = full_text.find(node.text[:50])
        if chunk_start == -1:
            chunk_start = 0
        current_role = "Общее"
        for role in roles:
            if role["start"] <= chunk_start:
                current_role = role["name"]
            else:
                break
        chunk_roles.append(current_role)

    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

    passages = [f"passage: {role}. {text}" for text, role in zip(texts, chunk_roles)]
    vectors = embedder.encode(passages, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    points = [
        models.PointStruct(
            id=idx,
            vector=vectors[idx].tolist(),
            payload={"text": texts[idx], "chunk_id": idx, "source": filepath, "role": chunk_roles[idx]},
        )
        for idx in range(len(texts))
    ]
    for i in range(0, len(points), 100):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])

    logger.info("Загружено в Qdrant: %d точек", len(points))
    return len(points)


# ═══════════════════════════════════════
# ПОИСК С QUERY EXPANSION
# ═══════════════════════════════════════

def expand_query(query: str) -> List[str]:
    prompt = f"""Ты помогаешь улучшить поиск по базе знаний о карьерных ролях и навыках.

Пользователь ищет: "{query}"

Сгенерируй 2-3 альтернативные формулировки этого запроса,
которые могут использовать другие названия ролей или синонимы.

Верни JSON-список строк. Только JSON, без пояснений."""

    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).rstrip("`").strip()
    try:
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [str(v) for v in variants]
    except json.JSONDecodeError:
        pass
    return []


def search_knowledge_base(query: str, top_k: int = TOP_K) -> List[Dict]:
    queries = [query] + expand_query(query)
    seen_ids = set()
    all_hits = []

    for q in queries:
        qvec = embedder.encode([f"query: {q}"], normalize_embeddings=True)[0].tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec,
            limit=top_k,
            with_payload=True,
        ).points

        for r in results:
            cid = r.payload.get("chunk_id", -1)
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_hits.append({
                    "text": r.payload["text"],
                    "chunk_id": cid,
                    "score": r.score,
                    "role": r.payload.get("role", "unknown"),
                })

    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:top_k]


# ═══════════════════════════════════════
# АГЕНТ
# ═══════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Поиск по внутренней базе знаний компании о карьерных ролях и навыках. "
                "Используй для поиска информации о навыках, ролях, обязанностях и компетенциях. "
                "Можешь вызывать несколько раз с разными запросами. "
                "Формулируй запросы конкретно."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос к базе знаний."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_clarification",
            "description": (
                "Задать уточняющий вопрос, если запрос слишком общий. Максимум 2 раза. "
                "Не задавай, если можешь найти ответ через search_knowledge_base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Уточняющий вопрос пользователю."}
                },
                "required": ["question"],
            },
        },
    },
]

SYSTEM_PROMPT = """Ты — карьерный ассистент компании. Твоя задача — помогать сотрудникам \
разобраться в навыках, ролях и компетенциях для профессионального развития.

## Главное правило
Ты отвечаешь ТОЛЬКО на основе данных, полученных через инструмент search_knowledge_base. \
Если инструмент не вернул релевантной информации по вопросу — честно скажи, \
что в базе знаний нет данных по этой теме. НЕ достраивай ответ из общих знаний, \
НЕ придумывай навыки и рекомендации, которых нет в результатах поиска.

## Критически важно
НИКОГДА не говори "у меня нет информации" или "в базе нет данных", \
если ты еще не вызвал search_knowledge_base. Сначала ВСЕГДА ищи, \
потом делай выводы. Единственное исключение — явный офтопик \
(вопросы не про карьеру и навыки).

Если пользователь спрашивает про конкретную роль (например, blockchain-разработчик, \
геймдизайнер), а search_knowledge_base вернул данные по ДРУГИМ ролям — \
это НЕ ответ на вопрос. В таком случае честно скажи, что данных по запрошенной \
роли в базе нет, и предложи посмотреть смежные роли, которые есть. \
НЕ переупаковывай навыки одной роли под названием другой.

## Проверка релевантности результатов
После каждого вызова search_knowledge_base задай себе вопрос: \
содержат ли результаты ПРЯМОЕ упоминание той роли, про которую спросил пользователь? \
Если пользователь спросил про "геймдизайнера", а в результатах фигурируют \
"продуктовый дизайнер", "фронтенд-разработчик" — это НЕ релевантные данные. \
Не пытайся адаптировать, экстраполировать или обобщать данные с других ролей. \
Скажи прямо: "В базе знаний нет данных по роли [название]. \
Могу рассказать про смежные роли: [список найденных]."

## Как работать с инструментами
1. Получив вопрос о роли или навыках — СРАЗУ вызывай search_knowledge_base. \
Не пытайся ответить без поиска.
2. Если в вопросе упоминаются ДВЕ или более ролей (сравнение, общие навыки, \
переход между ролями) — ОБЯЗАТЕЛЬНО сделай отдельный поиск по КАЖДОЙ роли.
3. Если вопрос слишком общий — задай уточняющий вопрос через ask_user_clarification.
4. Для поиска формулируй конкретные запросы.
5. Не задавай уточняющие вопросы, если можешь найти ответ через поиск.

## Формат ответа
- Адаптируй формат под вопрос.
- Будь конкретным.
- Если информация найдена частично — укажи это.

## Тон
Профессиональный, но дружелюбный. Конкретный и полезный.

## Ограничения
- Ты НЕ отвечаешь на вопросы вне карьерного развития.
- Не давай советов по зарплатам, конкретным вакансиям или персональным оценкам.
"""

sessions: Dict[int, List[Dict]] = defaultdict(list)
clarification_counts: Dict[int, int] = defaultdict(int)


def trim_history(chat_id: int):
    history = sessions[chat_id]
    pairs, cut = 0, 0
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "user":
            pairs += 1
        if pairs > MAX_HISTORY_PAIRS:
            cut = i + 1
            break
    sessions[chat_id] = history[cut:]


def execute_tool(tool_name: str, arguments: dict):
    if tool_name == "search_knowledge_base":
        hits = search_knowledge_base(arguments.get("query", ""))
        if not hits:
            return "Поиск не дал результатов."
        return "\n\n".join(f"[Результат {i+1}, score={h['score']:.3f}]\n{h['text']}" for i, h in enumerate(hits))
    elif tool_name == "ask_user_clarification":
        return None
    return f"Неизвестный инструмент: {tool_name}"


def run_agent(chat_id: int, user_message: str) -> str:
    sessions[chat_id].append({"role": "user", "content": user_message})
    trim_history(chat_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + sessions[chat_id]

    for iteration in range(MAX_AGENT_ITERATIONS):
        logger.info("Agent iteration %d/%d for chat %d", iteration + 1, MAX_AGENT_ITERATIONS, chat_id)
        response = openai_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                logger.info("Tool: %s(%s)", fn_name, json.dumps(fn_args, ensure_ascii=False)[:200])

                if fn_name == "ask_user_clarification":
                    if clarification_counts[chat_id] >= MAX_CLARIFICATIONS:
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                            "content": "Лимит уточнений исчерпан. Ответь на основе имеющейся информации."})
                        continue
                    question = fn_args.get("question", "")
                    clarification_counts[chat_id] += 1
                    sessions[chat_id].append({"role": "assistant", "content": question})
                    return question

                result = execute_tool(fn_name, fn_args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result or ""})
        else:
            answer = msg.content or ""
            sessions[chat_id].append({"role": "assistant", "content": answer})
            clarification_counts[chat_id] = 0
            return answer

    fallback = "Не смог сформировать ответ. Попробуй переформулировать вопрос."
    sessions[chat_id].append({"role": "assistant", "content": fallback})
    clarification_counts[chat_id] = 0
    return fallback


# ═══════════════════════════════════════
# TELEGRAM-БОТ
# ═══════════════════════════════════════

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

WELCOME = """Привет! Я карьерный ассистент.

Помогу разобраться в навыках и компетенциях для различных IT-ролей. Могу рассказать про конкретную роль, сравнить направления или помочь составить план развития.

Примеры вопросов:
- Какие навыки нужны бэкенд-разработчику?
- Чем отличается системный аналитик от бизнес-аналитика?
- Я фронтенд-разработчик, хочу стать техлидом — что развивать?
- Какие навыки общие у ML-инженера и дата-инженера?

Просто напиши свой вопрос!
"""


def safe_reply(message, text):
    parts = [text[i:i+4000] for i in range(0, len(text), 4000)] if text else [""]
    for part in parts:
        try:
            bot.reply_to(message, part, parse_mode="Markdown")
        except Exception:
            bot.reply_to(message, part)


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    sessions[message.chat.id] = []
    clarification_counts[message.chat.id] = 0
    safe_reply(message, WELCOME)


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    sessions[message.chat.id] = []
    clarification_counts[message.chat.id] = 0
    safe_reply(message, "История сброшена. Задай новый вопрос!")


@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_text(message):
    q = (message.text or "").strip()
    if not q:
        return
    try:
        bot.send_chat_action(message.chat.id, "typing")
        response = run_agent(message.chat.id, q)
        safe_reply(message, response)
    except Exception as e:
        logger.exception("Ошибка обработки сообщения от chat_id=%d", message.chat.id)
        safe_reply(message, "Произошла ошибка. Попробуй ещё раз.")


# ═══════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════

def main():
    logger.info("Начало индексации...")
    if not os.path.exists(DATA_FILE):
        logger.error("Файл %s не найден!", DATA_FILE)
        return

    num = index_documents(DATA_FILE)
    logger.info("Индексация завершена: %d чанков", num)

    logger.info("Запуск Telegram-бота...")
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    main()
