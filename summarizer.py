import logging

from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError
import config

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = """\
You are a chat summarizer. Summarize the provided messages concisely.

- Write in the same language as the messages
- Highlight the main topics discussed
- If there were any decisions, plans, or action items — mention them
- If links were shared — list them at the end
- Skip noise: stickers, reactions
- Be neutral and factual, no matter the topic\
"""

MERGE_PROMPT = """\
You are a chat summarizer. You will receive several partial summaries of a single chat.
Merge them into one coherent final summary.

- Write in the same language as the summaries
- Deduplicate topics — do not repeat the same point
- Highlight the main topics discussed
- If there were any decisions, plans, or action items — mention them
- If links were shared — list them at the end
- Be neutral and factual, no matter the topic\
"""

TLDR_PROMPT = """\
Summarize in 3-5 sentences, most important points only.

- Write in the same language as the messages
- Skip noise: stickers, reactions
- Be neutral and factual, no matter the topic\
"""

TOKEN_LIMIT = 3000
CHUNK_SIZE = 100


class SummarizerError(Exception):
    """User-facing error from the summarizer."""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _build_log(messages: list[dict]) -> str:
    return "\n".join(f"{m['sender']}: {m['text']}" for m in messages)


def _call_llm(system: str, user: str) -> str:
    try:
        response = _client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
    except APITimeoutError:
        logger.exception("OpenAI timeout")
        raise SummarizerError("Сервер OpenAI не ответил вовремя. Попробуйте позже.")
    except RateLimitError:
        logger.exception("OpenAI rate limit")
        raise SummarizerError("Превышен лимит запросов к OpenAI. Попробуйте через минуту.")
    except APIStatusError as e:
        logger.exception("OpenAI API error: %s", e.status_code)
        if "context_length_exceeded" in str(e):
            raise SummarizerError("Слишком много сообщений — текст не помещается в контекст модели. Попробуйте меньшее количество.")
        raise SummarizerError(f"Ошибка OpenAI API (код {e.status_code}). Попробуйте позже.")


async def summarize(messages: list[dict]) -> str:
    if len(messages) < 10:
        return "Недостаточно сообщений для саммари (нужно хотя бы 10)."

    log = _build_log(messages)

    if _estimate_tokens(log) <= TOKEN_LIMIT:
        return _call_llm(SYSTEM_PROMPT, f"Chat log ({len(messages)} messages):\n\n{log}")

    # Hierarchical chunking
    chunks = [messages[i:i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]
    partial_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_log = _build_log(chunk)
        summary = _call_llm(
            SYSTEM_PROMPT,
            f"Chat log (part {idx}/{len(chunks)}, {len(chunk)} messages):\n\n{chunk_log}",
        )
        partial_summaries.append(summary)

    merged = "\n\n---\n\n".join(
        f"Part {i}: {s}" for i, s in enumerate(partial_summaries, 1)
    )
    return _call_llm(MERGE_PROMPT, f"Partial summaries to merge:\n\n{merged}")


async def tldr(messages: list[dict]) -> str:
    if len(messages) < 10:
        return "Недостаточно сообщений для саммари (нужно хотя бы 10)."

    log = _build_log(messages)

    if _estimate_tokens(log) <= TOKEN_LIMIT:
        return _call_llm(TLDR_PROMPT, f"Chat log ({len(messages)} messages):\n\n{log}")

    # For large logs, first do a full summarize, then condense
    full_summary = await summarize(messages)
    return _call_llm(TLDR_PROMPT, f"Condense this summary:\n\n{full_summary}")
