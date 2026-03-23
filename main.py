import os
import json
import base64
import time
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

# simple in-memory state: chat_id -> { lang, sample }
CHAT_STATE: dict[str, dict] = {}

http_client: Optional[httpx.AsyncClient] = None


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


async def detect_language_and_translate_to_english(text: str) -> tuple[str, str]:
    """
    Detect language and translate to English.
    Returns (language_code, english_translation).
    """
    if not text.strip():
        return "unknown", text

    start = time.perf_counter()
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
            return "unknown", text

        lang = (data.get("language") or "unknown").lower().strip()
        translation = (data.get("translation_en") or "").strip()

        if not translation:
            translation = text

        return lang, translation

    except APITimeoutError as e:
        print("OpenAI timeout in detect_language_and_translate_to_english:", repr(e))
        return "unknown", text

    except Exception as e:
        print("OpenAI error in detect_language_and_translate_to_english:", repr(e))
        return "unknown", text

    finally:
        print(
            f"detect_language_and_translate_to_english took "
            f"{time.perf_counter() - start:.2f}s"
        )


async def translate_agent_reply_to_customer(agent_text: str, sample_customer_text: str) -> str:
    """
    Translate agent reply into the same language/script/style as customer text.
    """
    if not agent_text.strip():
        return agent_text

    start = time.perf_counter()
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate English customer support replies so that they match BOTH the language "
                        "and the writing style of CUSTOMER_TEXT.\n"
                        "- First, detect the language of CUSTOMER_TEXT.\n"
                        "- If AGENT_REPLY is already in the same language/script as CUSTOMER_TEXT, "
                        "return AGENT_REPLY exactly as-is (no paraphrasing or edits).\n"
                        "- Then translate AGENT_REPLY into that language.\n"
                        "- VERY IMPORTANT: match the same script and format as CUSTOMER_TEXT.\n"
                        "  * If CUSTOMER_TEXT uses Latin letters (roman Urdu/Hindi like 'yar kahan ho'), "
                        "    your reply MUST also use Latin letters, not Arabic or Devanagari script.\n"
                        "  * If CUSTOMER_TEXT uses a native script (Arabic, Devanagari, etc.), "
                        "    reply in that script.\n"
                        "- Keep a similar level of formality and style.\n"
                        "Return ONLY the translated reply text. No comments, no explanations."
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
            temperature=0.2,
        )

        translated = (resp.choices[0].message.content or "").strip()
        return translated if translated else agent_text

    except APITimeoutError as e:
        print("OpenAI timeout in translate_agent_reply_to_customer:", repr(e))
        return agent_text

    except Exception as e:
        print("OpenAI error in translate_agent_reply_to_customer:", repr(e))
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

            lang, translated_en = await detect_language_and_translate_to_english(text)
            print(f"Detected language for chat {chat_id}: {lang}")
            print("English translation:", translated_en)

            CHAT_STATE[chat_id] = {
                "lang": lang,
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
            sample = chat_state.get("sample", "")

            if lang == "en":
                print("Chat language is English; not translating agent reply.")
                return

            translated_reply = await translate_agent_reply_to_customer(text, sample)
            print("Translated agent reply for visitor:", translated_reply)

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
