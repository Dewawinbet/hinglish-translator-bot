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

# simple in-memory state: chat_id -> { lang, script, sample }
CHAT_STATE: dict[str, dict] = {}

http_client: Optional[httpx.AsyncClient] = None

LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


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
    """
    Keeps your original idea, with a couple of safe fallbacks if available.
    """
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

    # original rule
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
    if s in {"latin", "roman", "romanized", "roman urdu", "roman hindi"}:
        return "latin"
    if s in {"arabic", "urdu", "urdu script"}:
        return "arabic"
    if s in {"devanagari", "hindi", "hindi script"}:
        return "devanagari"
    return "other"


async def detect_language_and_translate_to_english(text: str) -> tuple[str, str, str]:
    """
    Detect language and translate to English.
    Returns (language_code, english_translation, script).
    Script is determined deterministically in Python.
    """
    if not text.strip():
        return "unknown", text, "other"

    start = time.perf_counter()
    script = detect_script(text)

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. "
                        "Given a customer message, you must respond ONLY with valid JSON in this exact format:\n"
                        "{"
                        "\"language\": \"<ISO 639-1 code like en, hi, ur, es>\", "
                        "\"translation_en\": \"<English translation of the message>\""
                        "}\n"
                        "No extra text, no comments, no markdown."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )

        content = (resp.choices[0].message.content or "").strip()
        print("OpenAI raw detect response:", content)

        data = safe_json_loads(content)
        if not data:
            print("Failed to parse JSON from OpenAI")
            return "unknown", text, script

        lang = (data.get("language") or "unknown").lower().strip()
        translation = (data.get("translation_en") or "").strip()

        if not translation:
            translation = text

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


async def repair_to_required_script(text: str, target_script: str, sample_customer_text: str) -> str:
    if not text.strip():
        return text

    script_instruction = {
        "latin": (
            "Rewrite the text using only Latin letters. "
            "Do NOT use Arabic script. Do NOT use Devanagari. "
            "Keep the meaning the same. "
            "Use a natural romanized style similar to CUSTOMER_TEXT."
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
                        "Return ONLY the rewritten text. No explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
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
    sample_customer_text: str,
    target_lang: str,
    target_script: str,
) -> str:
    """
    Translate agent reply into the same language/script/style as customer text.
    Wrong-script responses are rejected and retried.
    """
    if not agent_text.strip():
        return agent_text

    start = time.perf_counter()

    target_script = normalize_script_label(target_script)

    script_rule = {
        "latin": (
            "TARGET_SCRIPT is latin.\n"
            "- Every word must be written using Latin letters only.\n"
            "- Never output Arabic script.\n"
            "- Never output Devanagari.\n"
            "- Use natural romanized wording matching CUSTOMER_TEXT.\n"
            "- Good example: 'main yahan hoon'\n"
            "- Bad example: 'میں یہاں ہوں'\n"
            "- Bad example: 'मैं यहाँ हूँ'\n"
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
                                "You are a translation engine.\n"
                                "Translate AGENT_REPLY into the same language as CUSTOMER_TEXT.\n"
                                "You MUST obey the required script exactly.\n"
                                f"TARGET_LANGUAGE: {target_lang}\n"
                                f"TARGET_SCRIPT: {target_script}\n\n"
                                f"{script_rule}\n"
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
                                f"CUSTOMER_TEXT: {sample_customer_text}\n"
                                f"AGENT_REPLY (English): {agent_text}"
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

        # VISITOR FLOW
        if author_type == "visitor":
            print("Visitor message:", text)

            lang, translated_en, script = await detect_language_and_translate_to_english(text)
            print(f"Detected language for chat {chat_id}: {lang}")
            print(f"Detected script for chat {chat_id}: {script}")
            print("English translation:", translated_en)

            CHAT_STATE[chat_id] = {
                "lang": lang,
                "script": script,
                "sample": text,
            }

            if lang != "en":
                ok = await send_agent_only_message(chat_id, translated_en)
                print("Agent-only note sent:", ok)

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

            if lang == "en":
                print("Chat language is English; not translating agent reply.")
                return

            translated_reply = await translate_agent_reply_to_customer(
                text,
                sample,
                lang,
                script,
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

        # Return immediately; do the actual work in background
        background_tasks.add_task(process_livechat_event, body)
        return JSONResponse({"ok": True})

    except Exception as e:
        print("Unhandled error in livechat_webhook:", repr(e))
        return JSONResponse({"ok": True})

    finally:
        print(f"livechat_webhook ack path took {time.perf_counter() - start_total:.2f}s")
