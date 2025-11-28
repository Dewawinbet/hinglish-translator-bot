Think of 4 main players:

LiveChat (widget + agent panel)

Your FastAPI app (translator brain)

OpenAI (does the actual translations)

ngrok (the tunnel that lets LiveChat reach your local FastAPI)

And 2 keys to prove identity:

Webhook secret → LiveChat → FastAPI

PAT + Account ID → FastAPI → LiveChat

1. What each thing actually is
LiveChat App (in Dev Console)

This is a configuration object inside LiveChat.

You told it:

“When any chat incoming_event happens, send it to this URL: /livechat/webhook”

“Here is a secret string. Include it in the payload so my backend can verify it.”

“This app is allowed to use the Agent Chat API with scopes chats--all:rw, chats--access:rw”

So the App is just:

“Send events here, and allow this app to talk back to chats.”

PAT + Account ID

PAT = Personal Access Token for API calls.

Account ID = which LiveChat account this token belongs to.

LiveChat expects Basic auth with:

base64("ACCOUNT_ID:PAT")


In main.py you do:

raw = f"{LIVECHAT_ACCOUNT_ID}:{LIVECHAT_PAT}"
token_b64 = base64.b64encode(raw.encode()).decode()
Authorization: Basic {token_b64}


So:

PAT + Account ID = “I am this app, in this account. Please let me send messages into chats.”

Webhook

A webhook is LiveChat saying:

“When something happens in chats, I will POST JSON to your URL.”

You gave LiveChat:

URL: https://<your-ngrok>/livechat/webhook

Secret: 123adler3dwebfighting

So every time:

Visitor sends a message

Agent sends a message

LiveChat sends JSON to your FastAPI @app.post("/livechat/webhook").

Your code checks:

if incoming_secret != LIVECHAT_WEBHOOK_SECRET:
    ignore


This stops random people on the internet from faking LiveChat events.

ngrok

Your FastAPI runs on localhost:8000 (only visible to your machine).

LiveChat is on the internet, so it can’t reach localhost directly.

ngrok ≈ “public tunnel → your laptop”.

It gives you:

https://demisable-machineless-haven.ngrok-free.dev
   ↓ forwards to
http://127.0.0.1:8000


So LiveChat posts to https://ngrok/livechat/webhook, ngrok forwards that to your FastAPI.

OpenAI client

In main.py:

openai_client = OpenAI(api_key=OPENAI_API_KEY)


This is used in 2 places:

Visitor side → detect language + translate to English

Agent side → translate English reply back into visitor’s language/style

OpenAI is never talking to LiveChat directly; it only talks to your FastAPI.

2. Full flow: Visitor message

Visitor types Hinglish in the widget.

Step-by-step

Visitor → LiveChat

Widget sends the message to LiveChat servers.

LiveChat stores/shows it in the agent console and visitor UI.

LiveChat → FastAPI via webhook

LiveChat posts JSON to /livechat/webhook:

action: "incoming_event"

payload.event.type: "message"

payload.event.text: "bhai kitni dair aur? ..."

secret_key: "123adler3dwebfighting"

FastAPI verifies + classifies author

incoming_secret = body["secret_key"]
assert incoming_secret == LIVECHAT_WEBHOOK_SECRET

author_type = classify_author(author_id)
# no '@' → "visitor"


FastAPI → OpenAI (detect + translate to English)

lang, translated_en = detect_language_and_translate_to_english(text)
# e.g. lang="hi"


It also saves:

CHAT_STATE[chat_id] = {
    "lang": "hi",
    "sample": "bhai kitni dair aur? ..."
}


FastAPI → LiveChat (agent-only English note)

send_agent_only_message(chat_id, translated_en)


Internally:

Authorization: Basic base64("ACCOUNT_ID:PAT")
POST /agent/action/send_event
{
  "chat_id": "...",
  "event": {
    "type": "message",
    "text": "[EN] Brother, how much longer? ...",
    "visibility": "agents",
    "custom_id": "translator-bot"
  }
}


LiveChat shows that note only to agents
Visitor UI doesn’t see it (because visibility: "agents").

3. Full flow: Agent reply

Agent answers in English in the normal message composer (public message).

Step-by-step

Agent → LiveChat

Agent sends: "im sorry if you feel this way...".

LiveChat shows this English message to both visitor + agent.

LiveChat → FastAPI via webhook

Same webhook, but now:

author_id contains @ → classified as "agent".

visibility: "all".

FastAPI checks chat state

chat_state = CHAT_STATE[chat_id]  # {'lang': 'hi', 'sample': 'bhai kitni dair aur?...'}
lang = "hi"
sample = "bhai kitni dair aur?..."


FastAPI → OpenAI (translate agent reply back)

translated_reply = translate_agent_reply_to_customer(agent_text, sample)
# uses a system prompt that:
# - matches language
# - matches script (roman vs native)


FastAPI → LiveChat (send translated message to everyone)

send_visitor_message(chat_id, translated_reply, lang="hi")


Internally:

POST /agent/action/send_event
{
  "chat_id": "...",
  "event": {
    "type": "message",
    "text": "[Auto-translated HI] bhai agar aap aise mehsos...",
    "visibility": "all",
    "custom_id": "translator-bot"
  }
}


LiveChat shows that translated bubble

Visitor: sees English + [Auto-translated HI] ...

Agent: also sees both.

Next messages reuse state

For this chat_id, language + style are remembered in CHAT_STATE.

Every new visitor line resets sample, and every new agent line reuses that.

Loop prevention

When LiveChat sends back the translator-bot messages as webhook events, you skip them:

if event.get("custom_id") == "translator-bot":
    return


So your bot never translates its own messages.

4. All the pieces in one line

LiveChat App + Webhook config → “send every chat message as JSON to my FastAPI, signed with a secret”.

ngrok → “make my local FastAPI reachable at a public HTTPS URL for LiveChat”.

PAT + Account ID → “let my FastAPI act as an agent and send new messages into chats via Agent Chat API”.

FastAPI + OpenAI → “take those webhook events, translate them, and use the Agent Chat API to post helpful translations back”.