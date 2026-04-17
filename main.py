import asyncio
import os
import json
import base64
import time
import re
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, APITimeoutError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LIVECHAT_PAT = os.getenv("LIVECHAT_PAT")
LIVECHAT_WEBHOOK_SECRET = os.getenv("LIVECHAT_WEBHOOK_SECRET")
LIVECHAT_ACCOUNT_ID = os.getenv("LIVECHAT_ACCOUNT_ID")

LIVECHAT_API_BASE = "https://api.livechatinc.com/v3.5"

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "15"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
LIVECHAT_HTTP_TIMEOUT_SECONDS = float(os.getenv("LIVECHAT_HTTP_TIMEOUT_SECONDS", "10"))

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set")

if not LIVECHAT_PAT:
    print("WARNING: LIVECHAT_PAT not set")

if not LIVECHAT_WEBHOOK_SECRET:
    print("WARNING: LIVECHAT_WEBHOOK_SECRET not set")

if not LIVECHAT_ACCOUNT_ID:
    print("WARNING: LIVECHAT_ACCOUNT_ID not set")

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
)

# simple in-memory state per chat
CHAT_STATE: dict[str, dict] = {}

http_client: Optional[httpx.AsyncClient] = None

LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

MAX_RECENT_VISITOR_MESSAGES = 4
MAX_RECENT_AGENT_MESSAGES = 4
MAX_RECENT_VISITOR_DETECTIONS = 6

GENERIC_ENGLISH_SHORT_MESSAGES = {
    "ok",
    "okay",
    "yes",
    "no",
    "done",
    "sent",
    "already sent",
    "please check",
    "please check fast",
    "check fast",
    "check now",
    "why",
    "why delay",
    "same issue",
    "first one",
    "second one",
    "third one",
    "screenshot sent",
    "ss sent",
    "please help",
    "fast",
    "wait",
    "not yet",
    "now",
    "same",
    "first",
    "second",
    "third",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=LIVECHAT_HTTP_TIMEOUT_SECONDS)
    print("App startup complete")
    try:
        yield
    finally:
        if http_client is not None:
            await http_client.aclose()
        print("App shutdown complete")


app = FastAPI(lifespan=lifespan)


def get_livechat_auth_header() -> Optional[str]:
    """
    Build Basic auth header using:
        base64("ACCOUNT_ID:PAT")
    """
    if not LIVECHAT_ACCOUNT_ID or not LIVECHAT_PAT:
        print("LiveChat credentials missing")
        return None

    raw = f"{LIVECHAT_ACCOUNT_ID}:{LIVECHAT_PAT}"
    token_b64 = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return f"Basic {token_b64}"


@app.get("/")
async def root():
    return {"status": "ok", "message": "LiveChat translator backend running"}


@app.get("/livechat/oauth/callback")
async def livechat_oauth_callback(code: str = "", state: str = ""):
    print("Received OAuth code:", code)
    return JSONResponse({"ok": True, "received_code": bool(code)})


@app.get("/test-openai")
async def test_openai():
    start = time.perf_counter()
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
        )
        return JSONResponse(
            {
                "ok": True,
                "seconds": round(time.perf_counter() - start, 2),
                "reply": (resp.choices[0].message.content or "").strip(),
            }
        )
    except Exception as e:
        return JSONResponse(
            {
                "ok": False,
                "seconds": round(time.perf_counter() - start, 2),
                "error": repr(e),
            }
        )


def classify_author(author_id: str, event: dict | None = None) -> str:
    if not author_id:
        return "unknown"

    event = event or {}

    possible_type = (
        event.get("author_type")
        or event.get("author", {}).get("type")
        or event.get("source", {}).get("type")
    )

    if possible_type == "agent":
        return "agent"
    if possible_type in {"customer", "visitor"}:
        return "visitor"

    if "@" in author_id:
        return "agent"

    return "visitor"


def safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    return None


def detect_script(text: str) -> str:
    latin = len(LATIN_RE.findall(text))
    arabic = len(ARABIC_RE.findall(text))
    devanagari = len(DEVANAGARI_RE.findall(text))

    counts = {
        "latin": latin,
        "arabic": arabic,
        "devanagari": devanagari,
    }

    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "other"
    return best


def matches_required_script(text: str, required_script: str) -> bool:
    has_latin = bool(LATIN_RE.search(text))
    has_arabic = bool(ARABIC_RE.search(text))
    has_devanagari = bool(DEVANAGARI_RE.search(text))

    if required_script == "latin":
        return has_latin and not has_arabic and not has_devanagari

    if required_script == "arabic":
        return has_arabic and not has_devanagari

    if required_script == "devanagari":
        return has_devanagari and not has_arabic

    return True


def normalize_script_label(script: str) -> str:
    s = (script or "").strip().lower()
    if s in {"latin", "roman", "romanized", "roman urdu", "roman hindi", "hinglish"}:
        return "latin"
    if s in {"arabic", "urdu", "urdu script"}:
        return "arabic"
    if s in {"devanagari", "hindi", "hindi script"}:
        return "devanagari"
    return "other"


def normalize_roman_support_text(text: str) -> str:
    if not text:
        return text

    normalized = text

    replacements = {
        r"\bkaryea\b": "kariye",
        r"\bkarayea\b": "kariye",
        r"\bkriye\b": "kariye",
        r"\bkarain\b": "karein",
        r"\bkrain\b": "karein",
        r"\bmatlb\b": "matlab",
        r"\bplz\b": "please",
        r"\bpls\b": "please",
        r"\bthx\b": "thanks",
        r"\brha\b": "raha",
        r"\brhi\b": "rahi",
        r"\bnhi\b": "nahi",
        r"\bnai\b": "nahi",
    }

    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_or_create_chat_state(chat_id: str) -> dict:
    state = CHAT_STATE.get(chat_id)
    if state is None:
        state = {
            "lang": "unknown",  # effective dominant lang
            "script": "other",  # effective dominant script
            "sample": "",
            "recent_visitor_messages": [],
            "recent_agent_messages": [],
            "recent_visitor_detections": [],
        }
        CHAT_STATE[chat_id] = state
    return state


def append_recent_message(chat_id: str, author_type: str, text: str) -> None:
    state = get_or_create_chat_state(chat_id)

    if author_type == "visitor":
        items = state.setdefault("recent_visitor_messages", [])
        items.append(text)
        if len(items) > MAX_RECENT_VISITOR_MESSAGES:
            del items[:-MAX_RECENT_VISITOR_MESSAGES]

    elif author_type == "agent":
        items = state.setdefault("recent_agent_messages", [])
        items.append(text)
        if len(items) > MAX_RECENT_AGENT_MESSAGES:
            del items[:-MAX_RECENT_AGENT_MESSAGES]


def build_context_block(chat_id: str) -> str:
    state = get_or_create_chat_state(chat_id)

    visitor_msgs = state.get("recent_visitor_messages", [])
    agent_msgs = state.get("recent_agent_messages", [])

    lines = ["RECENT CHAT CONTEXT:"]

    if not agent_msgs and not visitor_msgs:
        lines.append("- none")
        return "\n".join(lines)

    if agent_msgs:
        lines.append("Recent agent messages:")
        for i, msg in enumerate(agent_msgs[-MAX_RECENT_AGENT_MESSAGES:], start=1):
            lines.append(f"{i}. {msg}")

    if visitor_msgs:
        lines.append("Recent visitor messages:")
        for i, msg in enumerate(visitor_msgs[-MAX_RECENT_VISITOR_MESSAGES:], start=1):
            lines.append(f"{i}. {msg}")

    return "\n".join(lines)


def is_short_generic_english_message(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False

    raw = re.sub(r"\s+", " ", raw)

    if raw in GENERIC_ENGLISH_SHORT_MESSAGES:
        return True

    if not raw.isascii():
        return False

    words = raw.split()
    if len(words) <= 3:
        return True

    if len(words) <= 5 and all(ch.isalnum() or ch.isspace() or ch in "?.!," for ch in raw):
        return raw in GENERIC_ENGLISH_SHORT_MESSAGES or any(
            phrase in raw
            for phrase in [
                "please check",
                "already sent",
                "check fast",
                "first one",
                "second one",
                "same issue",
                "please help",
                "why delay",
            ]
        )

    return False


def append_visitor_detection(
    chat_id: str,
    detected_lang: str,
    detected_script: str,
    text: str,
) -> None:
    state = get_or_create_chat_state(chat_id)
    items = state.setdefault("recent_visitor_detections", [])
    items.append(
        {
            "lang": (detected_lang or "unknown").lower().strip(),
            "script": normalize_script_label(detected_script),
            "text": text,
            "is_short_generic_english": is_short_generic_english_message(text),
        }
    )
    if len(items) > MAX_RECENT_VISITOR_DETECTIONS:
        del items[:-MAX_RECENT_VISITOR_DETECTIONS]


def find_last_non_english_detection(detections: list[dict]) -> Optional[dict]:
    for item in reversed(detections):
        lang = (item.get("lang") or "").lower().strip()
        if lang not in {"", "unknown", "en"}:
            return item
    return None


def compute_effective_language_and_script(chat_id: str) -> tuple[str, str, str]:
    """
    Dominant language logic:
    - do not let one short English message flip the whole chat to English
    - preserve prior non-English preference unless recent visitor behavior clearly changes
    Returns: (effective_lang, effective_script, effective_sample)
    """
    state = get_or_create_chat_state(chat_id)
    detections = state.get("recent_visitor_detections", [])

    if not detections:
        return "unknown", "other", ""

    latest = detections[-1]
    latest_lang = (latest.get("lang") or "unknown").lower().strip()
    latest_script = normalize_script_label(latest.get("script") or "other")
    latest_text = latest.get("text") or ""

    # If latest message is clearly non-English, trust it immediately.
    if latest_lang not in {"unknown", "en"}:
        return latest_lang, latest_script, latest_text

    last_non_en = find_last_non_english_detection(detections)

    # If latest message is English but short/generic, preserve previous non-English preference.
    if latest_lang == "en" and latest.get("is_short_generic_english") and last_non_en:
        return (
            last_non_en.get("lang") or "unknown",
            normalize_script_label(last_non_en.get("script") or "other"),
            last_non_en.get("text") or latest_text,
        )

    # If there are two most recent meaningful English messages in a row, allow switch to English.
    if len(detections) >= 2:
        last_two = detections[-2:]
        if all(
            (item.get("lang") or "").lower().strip() == "en"
            and not item.get("is_short_generic_english")
            for item in last_two
        ):
            return "en", "latin", latest_text

    # Weighted voting across recent visitor detections.
    lang_scores: dict[str, float] = {}
    best_non_en_candidate: Optional[dict] = None

    total = len(detections)
    for idx, item in enumerate(detections):
        lang = (item.get("lang") or "unknown").lower().strip()
        text = item.get("text") or ""
        is_short_en = bool(item.get("is_short_generic_english"))

        weight = float(idx + 1) / float(total)

        if lang == "unknown":
            weight *= 0.25
        elif lang == "en" and is_short_en:
            weight *= 0.2

        lang_scores[lang] = lang_scores.get(lang, 0.0) + weight

        if lang not in {"unknown", "en"}:
            best_non_en_candidate = item

    best_lang = max(lang_scores, key=lang_scores.get)
    best_lang_score = lang_scores.get(best_lang, 0.0)
    non_en_score = sum(
        score for lang, score in lang_scores.items() if lang not in {"unknown", "en"}
    )
    en_score = lang_scores.get("en", 0.0)

    # If English barely wins but there is a recent non-English preference, keep non-English.
    if best_lang == "en" and best_non_en_candidate:
        if non_en_score > 0 and en_score < non_en_score * 1.5:
            return (
                best_non_en_candidate.get("lang") or "unknown",
                normalize_script_label(best_non_en_candidate.get("script") or "other"),
                best_non_en_candidate.get("text") or latest_text,
            )

    if best_lang in {"unknown", ""} and last_non_en:
        return (
            last_non_en.get("lang") or "unknown",
            normalize_script_label(last_non_en.get("script") or "other"),
            last_non_en.get("text") or latest_text,
        )

    if best_lang == "en":
        return "en", "latin", latest_text

    # for non-English dominant language, use most recent detection of that language
    for item in reversed(detections):
        if (item.get("lang") or "").lower().strip() == best_lang:
            return (
                best_lang,
                normalize_script_label(item.get("script") or "other"),
                item.get("text") or latest_text,
            )

    return latest_lang, latest_script, latest_text


async def detect_language_and_translate_to_english(
    text: str,
    chat_id: str,
) -> tuple[str, str, str]:
    """
    Returns (detected_language_code, english_translation, detected_script)
    """
    if not text.strip():
        return "unknown", text, "other"

    start = time.perf_counter()
    script = detect_script(text)
    normalized_text = normalize_roman_support_text(text)
    context_block = build_context_block(chat_id)

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly careful customer-support translation engine.\n"
                        "Your job is to translate noisy support-chat messages into English.\n\n"
                        "The writer may be a customer or an agent.\n"
                        "The customer may write in English, Hindi, Urdu, Hinglish, Roman Hindi, Roman Urdu, "
                        "Indonesian, or mixed English + roman text.\n\n"
                        "Important rules:\n"
                        "- Return ONLY valid JSON.\n"
                        "- Use recent chat context to resolve short or ambiguous replies.\n"
                        "- Do NOT invent game names, product names, or proper nouns unless clearly supported by context.\n"
                        "- If a romanized token is ambiguous, prefer the grammatical meaning over treating it as a named entity.\n"
                        "- Translate conservatively and naturally.\n"
                        "- If already English, keep the meaning clear.\n\n"
                        "Return JSON in this exact shape:\n"
                        "{"
                        "\"language\": \"<ISO 639-1 code like en, hi, ur, id>\", "
                        "\"translation_en\": \"<best English translation>\", "
                        "\"confidence\": <number from 0 to 1>, "
                        "\"ambiguous_tokens\": [\"...\", \"...\"]"
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"RAW_CUSTOMER_TEXT: {text}\n"
                        f"NORMALIZED_HINT_TEXT: {normalized_text}\n"
                        f"DETECTED_SCRIPT_BY_SYSTEM: {script}"
                    ),
                },
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").strip()
        print("OpenAI raw detect response:", content)

        data = safe_json_loads(content)
        if not data:
            print("Failed to parse JSON from OpenAI")
            return "unknown", text, script

        lang = (data.get("language") or "unknown").lower().strip()
        translation = (data.get("translation_en") or "").strip()
        confidence = data.get("confidence")
        ambiguous_tokens = data.get("ambiguous_tokens") or []

        if not translation:
            translation = text

        print("Visitor translation confidence:", confidence)
        print("Visitor translation ambiguous_tokens:", ambiguous_tokens)

        return lang, translation, script

    except APITimeoutError as e:
        print("OpenAI timeout in detect_language_and_translate_to_english:", repr(e))
        return "unknown", text, script

    except Exception as e:
        print("OpenAI error in detect_language_and_translate_to_english:", repr(e))
        return "unknown", text, script

    finally:
        print(
            f"detect_language_and_translate_to_english took "
            f"{time.perf_counter() - start:.2f}s"
        )


async def detect_agent_language_and_prepare_translations(
    text: str,
    chat_id: str,
) -> tuple[str, str, str]:
    """
    Returns (source_language, translation_en, translation_id)
    source_language is expected to be en, id, or unknown.
    """
    if not text.strip():
        return "unknown", text, text

    start = time.perf_counter()
    context_block = build_context_block(chat_id)

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze support-agent replies.\n"
                        "The agent normally writes in either English or Indonesian.\n"
                        "Determine which language the reply is primarily written in, "
                        "then provide both an English and an Indonesian version.\n\n"
                        "Important rules:\n"
                        "- Return ONLY valid JSON.\n"
                        "- source_language must be exactly one of: en, id, unknown.\n"
                        "- Use recent chat context to resolve short or ambiguous replies.\n"
                        "- Keep the support meaning intact.\n"
                        "- Do NOT invent details, product names, or proper nouns.\n"
                        "- If the text is already English, translation_en should stay natural English.\n"
                        "- If the text is already Indonesian, translation_id should stay natural Indonesian.\n\n"
                        "Return JSON in this exact shape:\n"
                        "{"
                        "\"source_language\": \"en|id|unknown\", "
                        "\"translation_en\": \"<best English rendering>\", "
                        "\"translation_id\": \"<best Indonesian rendering>\""
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"AGENT_REPLY: {text}"
                    ),
                },
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").strip()
        print("OpenAI raw agent analyze response:", content)

        data = safe_json_loads(content)
        if not data:
            print("Failed to parse JSON from agent analyze response")
            raise ValueError("invalid agent analyze json")

        source_language = (data.get("source_language") or "unknown").lower().strip()
        if source_language not in {"en", "id", "unknown"}:
            source_language = "unknown"

        translation_en = (data.get("translation_en") or "").strip() or text
        translation_id = (data.get("translation_id") or "").strip() or text

        return source_language, translation_en, translation_id

    except APITimeoutError as e:
        print("OpenAI timeout in detect_agent_language_and_prepare_translations:", repr(e))

    except Exception as e:
        print("OpenAI error in detect_agent_language_and_prepare_translations:", repr(e))

    finally:
        print(
            f"detect_agent_language_and_prepare_translations took "
            f"{time.perf_counter() - start:.2f}s"
        )

    (fallback_lang, fallback_en, _), fallback_id = await asyncio.gather(
        detect_language_and_translate_to_english(text, chat_id),
        translate_to_indonesian(text, chat_id),
    )
    if fallback_lang not in {"en", "id"}:
        fallback_lang = "unknown"
    return fallback_lang, fallback_en, fallback_id


async def translate_to_indonesian(text: str, chat_id: str) -> str:
    if not text.strip():
        return text

    start = time.perf_counter()
    normalized_text = normalize_roman_support_text(text)
    context_block = build_context_block(chat_id)

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful translation engine for customer-support chats.\n"
                        "Translate the message into natural Indonesian.\n\n"
                        "Rules:\n"
                        "- Use recent chat context to interpret short replies and ambiguous words.\n"
                        "- The source may be English, Hindi, Urdu, Hinglish, Roman Urdu, or mixed text.\n"
                        "- Do NOT invent proper nouns.\n"
                        "- Preserve the support meaning clearly and naturally.\n"
                        "- Return ONLY the Indonesian translation text.\n"
                        "- No comments. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"RAW_TEXT: {text}\n"
                        f"NORMALIZED_HINT_TEXT: {normalized_text}"
                    ),
                },
            ],
            temperature=0,
        )

        translated = (resp.choices[0].message.content or "").strip()
        return translated if translated else text

    except APITimeoutError as e:
        print("OpenAI timeout in translate_to_indonesian:", repr(e))
        return text

    except Exception as e:
        print("OpenAI error in translate_to_indonesian:", repr(e))
        return text

    finally:
        print(f"translate_to_indonesian took {time.perf_counter() - start:.2f}s")


async def repair_to_required_script(
    text: str,
    target_script: str,
    sample_customer_text: str,
    chat_id: str,
) -> str:
    if not text.strip():
        return text

    context_block = build_context_block(chat_id)

    script_instruction = {
        "latin": (
            "Rewrite the text using only Latin letters. "
            "Do NOT use Arabic script. Do NOT use Devanagari. "
            "Keep meaning the same. "
            "Use natural romanized wording similar to CUSTOMER_TEXT."
        ),
        "arabic": (
            "Rewrite the text using Urdu/Arabic script. "
            "Do NOT use Devanagari. "
            "Avoid Latin transliteration except unavoidable brand names."
        ),
        "devanagari": (
            "Rewrite the text using Devanagari/Hindi script. "
            "Do NOT use Arabic script. "
            "Avoid Latin transliteration except unavoidable brand names."
        ),
    }.get(target_script, "Rewrite the text preserving meaning.")

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{script_instruction}\n"
                        "Use recent chat context only to preserve intended meaning.\n"
                        "Return ONLY the rewritten text. No explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"CUSTOMER_TEXT: {sample_customer_text}\n"
                        f"TEXT_TO_REWRITE: {text}"
                    ),
                },
            ],
            temperature=0,
        )
        repaired = (resp.choices[0].message.content or "").strip()
        return repaired if repaired else text

    except Exception as e:
        print("repair_to_required_script error:", repr(e))
        return text


async def translate_agent_reply_to_customer(
    agent_text: str,
    source_lang_hint: str,
    sample_customer_text: str,
    target_lang: str,
    target_script: str,
    chat_id: str,
) -> str:
    if not agent_text.strip():
        return agent_text

    start = time.perf_counter()

    target_script = normalize_script_label(target_script)
    context_block = build_context_block(chat_id)

    script_rule = {
        "latin": (
            "TARGET_SCRIPT is latin.\n"
            "- Every word must be written using Latin letters only.\n"
            "- Never output Arabic script.\n"
            "- Never output Devanagari.\n"
            "- Use natural Roman Urdu / Roman Hindi / Hinglish style that matches CUSTOMER_TEXT.\n"
            "- Keep the wording readable and normal for chat.\n"
        ),
        "arabic": (
            "TARGET_SCRIPT is arabic.\n"
            "- Output Urdu/Arabic script.\n"
            "- Never output Devanagari.\n"
            "- Avoid Latin transliteration except unavoidable brand names.\n"
        ),
        "devanagari": (
            "TARGET_SCRIPT is devanagari.\n"
            "- Output Hindi in Devanagari script.\n"
            "- Never output Arabic script.\n"
            "- Avoid Latin transliteration except unavoidable brand names.\n"
        ),
    }.get(target_script, "Match the script of CUSTOMER_TEXT exactly.\n")

    try:
        for attempt in range(3):
            try:
                resp = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a highly careful customer-support translation engine.\n"
                                "Translate AGENT_REPLY into the same language as CUSTOMER_TEXT.\n"
                                "You MUST obey the required script exactly.\n"
                                "Use recent chat context to keep the reply aligned with the ongoing conversation.\n\n"
                                f"SOURCE_LANGUAGE_HINT: {source_lang_hint or 'unknown'}\n"
                                f"TARGET_LANGUAGE: {target_lang}\n"
                                f"TARGET_SCRIPT: {target_script}\n\n"
                                f"{script_rule}\n"
                                "Important rules:\n"
                                "- Keep the original customer-support meaning intact.\n"
                                "- If the customer's style is romanized, keep the reply romanized.\n"
                                "- For latin target script, do not suddenly switch into Hindi or Urdu script.\n"
                                "- Preserve clarity for deposit, payment, verification, support, and UPI-related messages.\n\n"
                                "Return ONLY valid JSON in this exact format:\n"
                                "{"
                                "\"translated_text\": \"...\", "
                                "\"language\": \"...\", "
                                "\"script\": \"latin|arabic|devanagari|other\""
                                "}\n"
                                "No markdown. No comments. No extra text."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"{context_block}\n\n"
                                f"CUSTOMER_TEXT: {sample_customer_text}\n"
                                f"AGENT_REPLY: {agent_text}"
                            ),
                        },
                    ],
                    temperature=0,
                )

                content = (resp.choices[0].message.content or "").strip()
                print(f"OpenAI raw translate response attempt {attempt + 1}:", content)

                data = safe_json_loads(content)
                if not data:
                    print("Failed to parse translate JSON; retrying")
                    continue

                translated = (data.get("translated_text") or "").strip()
                returned_script = normalize_script_label(data.get("script") or "")

                if not translated:
                    print("Empty translated_text; retrying")
                    continue

                if matches_required_script(translated, target_script):
                    return translated

                print(
                    "Script mismatch detected. "
                    f"Required={target_script}, returned={returned_script}, text={translated}"
                )

                repaired = await repair_to_required_script(
                    translated,
                    target_script,
                    sample_customer_text,
                    chat_id,
                )

                if repaired and matches_required_script(repaired, target_script):
                    return repaired

                print("Repair attempt still failed script validation; retrying")

            except APITimeoutError as e:
                print("OpenAI timeout in translate_agent_reply_to_customer:", repr(e))

            except Exception as e:
                print("OpenAI error in translate_agent_reply_to_customer attempt:", repr(e))

        print("All translation attempts failed script validation; falling back to original agent text")
        return agent_text

    finally:
        print(
            f"translate_agent_reply_to_customer took "
            f"{time.perf_counter() - start:.2f}s"
        )


async def post_livechat_event(payload: dict, label: str) -> bool:
    if not LIVECHAT_PAT:
        print(f"{label}: LIVECHAT_PAT not set")
        return False

    auth_header = get_livechat_auth_header()
    if not auth_header:
        print(f"{label}: auth header missing")
        return False

    if http_client is None:
        print(f"{label}: http_client not initialized")
        return False

    url = f"{LIVECHAT_API_BASE}/agent/action/send_event"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        print(f"{label} payload:", json.dumps(payload, ensure_ascii=False))
        r = await http_client.post(url, headers=headers, json=payload)
        print(f"{label} status:", r.status_code)
        print(f"{label} response:", r.text)
        return r.status_code < 400

    except httpx.TimeoutException as e:
        print(f"{label} timeout:", repr(e))
        return False

    except Exception as e:
        print(f"{label} error:", repr(e))
        return False

    finally:
        print(f"{label} took {time.perf_counter() - start:.2f}s")


async def send_agent_only_message(chat_id: str, text: str) -> bool:
    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": f"[EN] {text}",
            "visibility": "agents",
            "custom_id": "translator-bot",
        },
    }
    return await post_livechat_event(payload, "send_agent_only_message")


async def send_agent_only_message_id(chat_id: str, text: str) -> bool:
    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": f"[ID] {text}",
            "visibility": "agents",
            "custom_id": "translator-bot",
        },
    }
    return await post_livechat_event(payload, "send_agent_only_message_id")


async def send_visitor_message(chat_id: str, text: str, lang: str | None = None) -> bool:
    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": text,
            "visibility": "all",
            "custom_id": "translator-bot",
        },
    }
    return await post_livechat_event(payload, "send_visitor_message")


async def process_livechat_event(body: dict):
    start_total = time.perf_counter()
    try:
        payload = body.get("payload", {})
        event = payload.get("event", {})

        if event.get("type") != "message":
            return

        if event.get("custom_id") == "translator-bot":
            print("Ignoring translator-bot message to avoid loop")
            return

        text = event.get("text") or ""
        author_id = event.get("author_id") or ""
        chat_id = payload.get("chat_id")

        if not chat_id or not text.strip():
            return

        author_type = classify_author(author_id, event)
        print(f"Author {author_id} classified as: {author_type}")

        append_recent_message(chat_id, author_type, text)

        # VISITOR FLOW
        if author_type == "visitor":
            print("Visitor message:", text)

            (
                (detected_lang, translated_en, detected_script),
                translated_id,
            ) = await asyncio.gather(
                detect_language_and_translate_to_english(
                    text,
                    chat_id,
                ),
                translate_to_indonesian(text, chat_id),
            )
            print(f"Detected language for chat {chat_id}: {detected_lang}")
            print(f"Detected script for chat {chat_id}: {detected_script}")
            print("English translation:", translated_en)
            print("Indonesian translation:", translated_id)

            append_visitor_detection(chat_id, detected_lang, detected_script, text)

            effective_lang, effective_script, effective_sample = compute_effective_language_and_script(chat_id)
            print(f"Effective dominant language for chat {chat_id}: {effective_lang}")
            print(f"Effective dominant script for chat {chat_id}: {effective_script}")
            print(f"Effective sample for chat {chat_id}: {effective_sample}")

            state = get_or_create_chat_state(chat_id)
            state["lang"] = effective_lang
            state["script"] = effective_script
            state["sample"] = effective_sample or text

            ok_en = await send_agent_only_message(chat_id, translated_en)
            print("Agent-only EN note sent:", ok_en)

            ok_id = await send_agent_only_message_id(chat_id, translated_id)
            print("Agent-only ID note sent:", ok_id)

            return

        # AGENT FLOW
        if author_type == "agent":
            print("Agent message:", text)

            chat_state = CHAT_STATE.get(chat_id)
            if not chat_state:
                print("No stored language for this chat, skipping agent translation.")
                return

            lang = chat_state.get("lang", "unknown")
            script = chat_state.get("script", "other")
            sample = chat_state.get("sample", "")

            agent_source_lang, translated_en, translated_id = (
                await detect_agent_language_and_prepare_translations(text, chat_id)
            )
            print(f"Agent source language for chat {chat_id}: {agent_source_lang}")
            print("Agent message English translation:", translated_en)
            print("Agent message Indonesian translation:", translated_id)

            if agent_source_lang == "id":
                ok_en = await send_agent_only_message(chat_id, translated_en)
                print("Agent-only EN note sent for agent message:", ok_en)
            elif agent_source_lang == "en":
                ok_id = await send_agent_only_message_id(chat_id, translated_id)
                print("Agent-only ID note sent for agent message:", ok_id)
            else:
                ok_en = await send_agent_only_message(chat_id, translated_en)
                ok_id = await send_agent_only_message_id(chat_id, translated_id)
                print("Agent-only EN note sent for agent message:", ok_en)
                print("Agent-only ID note sent for agent message:", ok_id)

            if lang in {"", "unknown"}:
                print("Stored dominant client language is unknown; skipping visitor translation.")
                return

            translated_reply: Optional[str] = None

            if lang == "en":
                if agent_source_lang == "en":
                    print("Agent already replied in English for an English chat; skipping visitor translation.")
                    return
                translated_reply = translated_en
            elif lang == "id":
                if agent_source_lang == "id":
                    print("Agent already replied in Indonesian for an Indonesian chat; skipping visitor translation.")
                    return
                translated_reply = translated_id
            else:
                translated_reply = await translate_agent_reply_to_customer(
                    text,
                    agent_source_lang,
                    sample,
                    lang,
                    script,
                    chat_id,
                )
            print("Translated agent reply for visitor:", translated_reply)

            if not matches_required_script(translated_reply, normalize_script_label(script)):
                print("Blocked visitor message due to script mismatch after all attempts")
                return

            ok = await send_visitor_message(chat_id, translated_reply, lang)
            print("Visitor message sent:", ok)
            return

        print("Unknown author type, ignoring.")

    except Exception as e:
        print("Unhandled error in process_livechat_event:", repr(e))

    finally:
        print(f"process_livechat_event total took {time.perf_counter() - start_total:.2f}s")


@app.post("/livechat/webhook")
async def livechat_webhook(request: Request, background_tasks: BackgroundTasks):
    start_total = time.perf_counter()

    try:
        body = await request.json()
        print("Webhook payload:", body)

        incoming_secret = body.get("secret_key") or ""
        if LIVECHAT_WEBHOOK_SECRET and incoming_secret != LIVECHAT_WEBHOOK_SECRET:
            print("Invalid webhook secret, ignoring")
            return JSONResponse({"ok": False})

        action = body.get("action")
        if action != "incoming_event":
            return JSONResponse({"ok": True})

        payload = body.get("payload", {})
        event = payload.get("event", {})

        if event.get("type") != "message":
            return JSONResponse({"ok": True})

        if event.get("custom_id") == "translator-bot":
            print("Ignoring translator-bot message to avoid loop")
            return JSONResponse({"ok": True})

        background_tasks.add_task(process_livechat_event, body)
        return JSONResponse({"ok": True})

    except Exception as e:
        print("Unhandled error in livechat_webhook:", repr(e))
        return JSONResponse({"ok": True})

    finally:
        print(f"livechat_webhook ack path took {time.perf_counter() - start_total:.2f}s")
