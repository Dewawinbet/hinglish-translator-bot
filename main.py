from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import os
import json
import base64

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LIVECHAT_PAT = os.getenv("LIVECHAT_PAT")
LIVECHAT_WEBHOOK_SECRET = os.getenv("LIVECHAT_WEBHOOK_SECRET")
LIVECHAT_ACCOUNT_ID = os.getenv("LIVECHAT_ACCOUNT_ID")

LIVECHAT_API_BASE = "https://api.livechatinc.com/v3.5"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set in .env")

if not LIVECHAT_PAT:
    print("WARNING: LIVECHAT_PAT not set in .env")

if not LIVECHAT_WEBHOOK_SECRET:
    print("WARNING: LIVECHAT_WEBHOOK_SECRET not set in .env")

if not LIVECHAT_ACCOUNT_ID:
    print("WARNING: LIVECHAT_ACCOUNT_ID not set in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# simple in-memory state: chat_id -> { lang, sample }
CHAT_STATE: dict[str, dict] = {}


def get_livechat_auth_header():
    """
    Build Basic auth header for LiveChat Agent Chat API using:
        base64("ACCOUNT_ID:PAT")
    """
    if not LIVECHAT_ACCOUNT_ID or not LIVECHAT_PAT:
        print("LiveChat credentials missing (LIVECHAT_ACCOUNT_ID or LIVECHAT_PAT)")
        return None

    raw = f"{LIVECHAT_ACCOUNT_ID}:{LIVECHAT_PAT}"
    token_b64 = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return f"Basic {token_b64}"


@app.get("/")
async def root():
    return {"status": "ok", "message": "LiveChat translator backend running"}


@app.get("/livechat/oauth/callback")
async def livechat_oauth_callback(code: str = "", state: str = ""):
    # not used yet – we rely on Personal Access Token (PAT) for now
    print("Received OAuth code:", code)
    return JSONResponse({"ok": True, "received_code": bool(code)})


def classify_author(author_id: str) -> str:
    """
    Very simple classifier:
    - if author_id contains '@' → agent
    - else → visitor
    """
    if not author_id:
        return "unknown"
    if "@" in author_id:
        return "agent"
    return "visitor"


async def detect_language_and_translate_to_english(text: str) -> tuple[str, str]:
    """
    Use OpenAI to:
      - detect language
      - translate to clear English
    Returns (language_code, english_translation).
    """
    if not text.strip():
        return "unknown", text

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translation engine. "
                    "Given a customer message, you must respond ONLY with JSON in this exact format:\n"
                    "{"
                    "\"language\": \"<ISO 639-1 code like en, hi, ur, es>\", "
                    "\"translation_en\": \"<English translation of the message>\""
                    "}\n"
                    "No extra text, no comments."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()
    print("OpenAI raw JSON response:", content)

    try:
        data = json.loads(content)
        lang = (data.get("language") or "unknown").lower()
        translation = (data.get("translation_en") or "").strip()
        if not translation:
            translation = text
        return lang, translation
    except Exception as e:
        print("Failed to parse JSON from OpenAI:", e)
        return "unknown", text


async def translate_agent_reply_to_customer(agent_text: str, sample_customer_text: str) -> str:
    """
    Translate agent's English reply into the same language and style as the sample customer text.
    If the customer wrote in romanized form (e.g. Urdu/Hindi using Latin letters),
    keep the reply in the same romanized script and style.
    """
    if not agent_text.strip():
        return agent_text

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate English customer support replies so that they match BOTH the language "
                    "and the writing style of CUSTOMER_TEXT.\n"
                    "- First, detect the language of CUSTOMER_TEXT.\n"
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

    return resp.choices[0].message.content.strip()


async def send_agent_only_message(chat_id: str, text: str):
    """
    Send a message visible only to agents in a given chat (English translation of visitor).
    """
    if not LIVECHAT_PAT:
        print("LIVECHAT_PAT not set, cannot send agent-only message")
        return

    auth_header = get_livechat_auth_header()
    if not auth_header:
        return

    url = f"{LIVECHAT_API_BASE}/agent/action/send_event"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": f"[EN] {text}",
            "visibility": "agents",  # agents-only translation
            "custom_id": "translator-bot",  # so we can ignore our own messages
        },
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("send_agent_only_message status:", r.status_code, r.text)


async def send_visitor_message(chat_id: str, text: str, lang: str | None = None):
    """
    Send a translated message to the visitor (visible to all).
    Visitor will see this as the localized reply.
    Agent will also see it in the conversation, which is okay.
    The message is clearly marked as auto-translated.
    """
    if not LIVECHAT_PAT:
        print("LIVECHAT_PAT not set, cannot send visitor message")
        return

    auth_header = get_livechat_auth_header()
    if not auth_header:
        return

    lang_label = (lang or "").upper() if lang else ""
    prefix = f"[Auto-translated {lang_label}] " if lang_label else "[Auto-translated] "
    display_text = prefix + text

    url = f"{LIVECHAT_API_BASE}/agent/action/send_event"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": display_text,
            "visibility": "all",
            "custom_id": "translator-bot",
        },
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("send_visitor_message status:", r.status_code, r.text)


@app.post("/livechat/webhook")
async def livechat_webhook(request: Request):
    body = await request.json()
    print("Webhook payload:", body)

    # 1) Verify secret_key from LiveChat
    incoming_secret = body.get("secret_key") or ""
    if LIVECHAT_WEBHOOK_SECRET and incoming_secret != LIVECHAT_WEBHOOK_SECRET:
        print("Invalid webhook secret, ignoring")
        return JSONResponse({"ok": False})

    action = body.get("action")
    if action != "incoming_event":
        # ignore other actions for now
        return JSONResponse({"ok": True})

    payload = body.get("payload", {})
    event = payload.get("event", {})

    if event.get("type") != "message":
        return JSONResponse({"ok": True})

    # Ignore our own messages (we mark them with custom_id = translator-bot)
    if event.get("custom_id") == "translator-bot":
        print("Ignoring translator-bot message to avoid loop")
        return JSONResponse({"ok": True})

    text = event.get("text") or ""
    author_id = event.get("author_id") or ""
    chat_id = payload.get("chat_id")
    if not chat_id or not text.strip():
        return JSONResponse({"ok": True})

    author_type = classify_author(author_id)
    print(f"Author {author_id} classified as: {author_type}")

    # === VISITOR MESSAGE FLOW ===
    if author_type == "visitor":
        print("Visitor message:", text)

        # detect language + translate to English
        lang, translated_en = await detect_language_and_translate_to_english(text)
        print(f"Detected language for chat {chat_id}: {lang}")
        print("English translation:", translated_en)

        # store state for this chat
        CHAT_STATE[chat_id] = {
            "lang": lang,
            "sample": text,  # we use this to guide agent reply translation (script + style)
        }

        # If visitor language is not English, show English version to agent
        if lang != "en":
            await send_agent_only_message(chat_id, translated_en)

        # If it's already English, we do nothing extra for now
        return JSONResponse({"ok": True})

    # === AGENT MESSAGE FLOW ===
    if author_type == "agent":
        print("Agent message:", text)

        # get chat language state
        chat_state = CHAT_STATE.get(chat_id)
        if not chat_state:
            print("No stored language for this chat, skipping agent translation.")
            return JSONResponse({"ok": True})

        lang = chat_state.get("lang", "unknown")
        sample = chat_state.get("sample", "")

        # If chat language is English, no need to translate back
        if lang == "en":
            print("Chat language is English; not translating agent reply.")
            return JSONResponse({"ok": True})

        # Translate agent's English reply into customer's language & script style
        translated_reply = await translate_agent_reply_to_customer(text, sample)
        print("Translated agent reply for visitor:", translated_reply)

        await send_visitor_message(chat_id, translated_reply, lang)
        return JSONResponse({"ok": True})

    # Unknown author type, ignore for now
    print("Unknown author type, ignoring.")
    return JSONResponse({"ok": True})
