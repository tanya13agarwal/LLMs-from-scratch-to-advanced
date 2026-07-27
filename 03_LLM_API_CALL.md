# 📘 Module 3 — Free LLM APIs (Groq, Gemini, Ollama) · Complete Notes

> **Difficulty:** 🟢 Easy · **Time:** 5–6 hours · **Prerequisites:** Module 1 (the stack) + Module 2 (how LLMs work)
> **What you'll build:** two working chatbots with memory — one on the **cloud** (Groq, free + fast) and the same one running **locally on your laptop** (Ollama, free + private + offline).

**How to read these notes:** top to bottom, running each file as it appears. Every code file is explained line by line in plain English. Analogies used throughout: the **wizard in a tower** (the AI), the **window** (the API), and the **notebook** (memory).

---

# PART 1 — What *is* an LLM API?

## 🧒 The wizard in the tower

In Module 2 you learned the LLM "brain" is a giant pile of math living on a powerful computer far away. You can't fit it in your pocket — so how do you talk to it? **You pass it notes.**

🧙 Picture a super-smart wizard locked at the top of a tower. You can't go in, but there's a tiny **window**. You write your question on a slip, slide it through, and a moment later a slip slides back with the answer.

- **Window** = the **API** (Application Programming Interface)
- **Slip you slide in** = the **request** (your question + settings)
- **Slip that comes back** = the **response** (the answer)
- **Wizard** = the **LLM** (Llama, Gemini, etc.)

When you build a chatbot, your Python program is the thing sliding notes through the window — over and over, fast. And several wizards let you do this **for free** — those are the ones we use.

## 🌍 Adult analogy — the restaurant kitchen window

An API is like a **kitchen window**: you hand over a written **order** (request), the kitchen cooks, and hands back a **plate** (response). You don't need to know *how* they cooked — only how to **write the order** and **read what comes back**.

🎯 **The punchline that makes this whole module easy:** almost every major provider's window speaks the *same* order format (nicknamed "OpenAI-compatible"). **Learn the format once → order from many kitchens** (Groq, Ollama, OpenAI…) with barely any code change. That's why later we build the same chatbot on Groq *and* locally with ~2 lines different.

## 📖 The most important concept: a request is a **list of messages**

The "order" is just data — a Python **list**, where each item is a small **dictionary** with a `role` and `content`. There are **three roles**:

| Role | Who's "speaking" | What it's for |
|---|---|---|
| `system` | **You**, secretly setting rules | Personality + instructions ("You are a polite tutor."). Sent once, at the top. |
| `user` | **The human** | The actual question/message. |
| `assistant` | **The AI** | The AI's replies. ⭐ **You save these** — that's how the bot gets memory. |

A real order, read like a chat transcript:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What's the capital of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
    {"role": "user",      "content": "How many people live there?"}   # ← newest question
]
```

- **Line 1 (system):** you quietly set the rules first.
- **Lines 2–3 (user → assistant):** a round that already happened.
- **Line 4 (user):** the new question being asked *right now*.

The wizard reads this **entire list**, top to bottom, and writes the next `assistant` message.

> 🔗 **Module 2 callback:** "reads the entire list" = the **context window** (the model's desk). Every word counts as **tokens**. Too long → oldest notes fall off the desk → the bot "forgets." You already understand why.

## 🎛️ The settings — you already know the main one!

Along with the messages, the order carries a few settings:
- `model` — which wizard (e.g. `llama-3.3-70b-versatile`)
- `temperature` — 🔗 **the exact dial from Module 2, Lab 3.** 0 = precise/repetitive, ~0.7 = balanced, >1 = creative/wild.
- `max_tokens` — a cap on answer length.
- `stream` — send the answer word-by-word (feels fast, like ChatGPT typing).

## 💸 Why we start FREE

- **Groq** 🏎️ — a *cloud* window with free access to strong open models (Llama, etc.), and it's **very fast**. Our main wizard.
- **Ollama** 🏠 — runs a model **on your own laptop**: zero cost, fully private, no internet needed.

We only reach for paid wizards (later modules) when a free one genuinely can't do the job.

## 🔑 Get your free Groq key (2 min)
1. Go to **https://console.groq.com** → sign in (Google is fastest, no card).
2. Sidebar → **API Keys** → **Create API Key** → **copy it** (shown once!).

> 🛡️ **If a model ID ever stops working** (providers retire them occasionally): the live list is at **https://console.groq.com/docs/models**. Swap the `model=` string for a current one. A good fast/cheap alternative to `llama-3.3-70b-versatile` is `llama-3.1-8b-instant`.

---

# PART 2 — Your First API Call (the brute-force way)

We make the first call using only `requests` (plain HTTP, no AI library) so you *see* exactly what travels through the window. Understanding this first is what makes the shortcut (Part 3) safe to use.

## 🧰 Setup
```bat
cd C:\genai-bootcamp\module-03-free-llm-apis
mkdir project-01-groq-chatbot
cd project-01-groq-chatbot
python -m venv venv
venv\Scripts\activate
pip install requests python-dotenv
```
✅ Your prompt should now show `(venv)` — you're in the project's private package box.

**Create `.env`** (key never goes in code):
```
GROQ_API_KEY=paste_your_groq_key_here
```
**Create `.gitignore`** (so the key never leaks to GitHub):
```
.env
venv/
__pycache__/
```
> 🔒 This one habit matters most: leaked keys get scraped off GitHub and abused within minutes.

## 💻 `01_raw_call.py`
```python
"""
Module 3 - Step 2: Call the LLM the RAW way - no SDK, just HTTP (BRUTE FORCE).
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {                                        # the ENVELOPE
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {                                        # the LETTER
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "In one sentence, what is an API?"},
    ],
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=payload)   # slide the note

print("=== FULL RAW RESPONSE ===")
print(json.dumps(response.json(), indent=2))

data = response.json()
print("\n=== JUST THE ANSWER ===")
print(data["choices"][0]["message"]["content"])

print("\n=== TOKENS USED ===")
print(data["usage"])
```

## 🔍 Line by line
- `import os, json, requests` + `from dotenv import load_dotenv` → four toolkits: read env vars, pretty-print, make HTTP calls, read `.env`.
- `load_dotenv()` → opens `.env` and loads it; now `GROQ_API_KEY` is visible.
- `api_key = os.environ["GROQ_API_KEY"]` → grabs the key out of the loaded environment. (Key lives in a file, never typed in code.)
- `url = ".../openai/v1/chat/completions"` → the **window's exact address** on Groq's servers. (`/openai/v1/` is the "speaks OpenAI format" part — why Ollama later barely changes the code.)
- `headers = {...}` — the **ENVELOPE** 📨 (info *about* the request):
  - `"Authorization": f"Bearer {api_key}"` → proves it's you. `Bearer` = "the bearer of this token is allowed in."
  - `"Content-Type": "application/json"` → "the letter inside is JSON."
- `payload = {...}` — the **LETTER** ✉️ (the actual order): `model`, the `messages` list (system + user roles), and `temperature` (🔗 the Lab 3 dial).
- `requests.post(url, headers=headers, json=payload)` → the moment the note slides through. `json=payload` converts your dict to JSON automatically. Reply lands in `response`.
- `json.dumps(response.json(), indent=2)` → turn the reply into a dict and print it neatly so you can read the whole raw plate.
- `data["choices"][0]["message"]["content"]` → **peeling the onion** 🧅: `["choices"]` is a *list* of answers (the API can return several); `[0]` = first answer; `["message"]` = its message object; `["content"]` = the actual text.
- `data["usage"]` → token counts. 🔗 From Module 2 — what you'd be billed on, what fills the context window.

## ▶️ Run & observe
```bat
python 01_raw_call.py
```
Notice: the big raw JSON blob (`choices`, `usage`, `model`…), the clean answer dug out of the nesting, and the token count. See how much packaging surrounds one little answer — that's the busywork the SDK removes next.

---

# PART 3 — The Same Call, the Easy Way (the SDK shortcut)

## 🤔 Why a shortcut?
Writing the URL, headers, post, and onion-peeling *every time* is a lot of plumbing. A **SDK** (Software Development Kit) is a helper library that does that exact plumbing for you. Groq's is called `groq`.

🧒 In Part 2 you posted a letter yourself (envelope, address, stamp, walk to the postbox). The SDK is an **assistant** who does all that when you just say "send this." **Same post office — not magic.** It's only fine *because* you already know what's underneath.

## 🧰 Setup (same folder, `(venv)` showing)
```bat
pip install groq
```

## 💻 `02_sdk_call.py`
```python
"""
Module 3 - Step 3: The SAME call, using the groq SDK (THE SHORTCUT).
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])     # hire the postal assistant

response = client.chat.completions.create(            # one clean command
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "In one sentence, what is an API?"},
    ],
    temperature=0.7,
)

print("=== ANSWER ===")
print(response.choices[0].message.content)            # dot-access, no JSON peeling

print("\n=== TOKENS USED ===")
print(response.usage.total_tokens)
```

## 🔍 Line by line — notice what *disappeared*
- `from groq import Groq` → import the helper (`Groq`, capital G).
- `client = Groq(api_key=...)` → create your **client** = hiring the assistant and handing them your key once. ➡️ **This replaced Part 2's `url` AND `headers`.** No endpoint typing, no `Bearer`, no `Content-Type`.
- `client.chat.completions.create(...)` → read it as plain English: *"client, on chat completions, create a reply."* Same three inputs as before (`model`, `messages`, `temperature`). ➡️ **This replaced Part 2's `payload` AND `requests.post(...)`.**
- `response.choices[0].message.content` → the **same onion** as Part 2, but dots instead of brackets/quotes:
  ```
  Raw:  data["choices"][0]["message"]["content"]
  SDK:  response.choices[0].message.content
  ```
- `response.usage.total_tokens` → same token info, clean dot-access, hands you the total directly.

## ▶️ Run & compare
```bat
python 02_sdk_call.py
```
| | Raw HTTP | SDK |
|---|---|---|
| Wrote the URL? | ✅ by hand | ❌ client does it |
| Built the envelope? | ✅ by hand | ❌ client does it |
| Converted + sent? | ✅ `requests.post` | ❌ one `.create()` |
| Dug out the answer | `["choices"][0]...` | `.choices[0]...` |

The SDK isn't mysterious — it's doing your Part 2 code for you. From here on we use it.

---

# PART 4 — Giving the Bot a Memory (the most important idea)

## 🧠 The surprise: the LLM has NO memory of its own
Almost everyone gets this wrong. **Every API call is a blank slate.** The wizard reads your note, answers, and *instantly forgets it ever happened.*

🧒 The tower wizard has the worst memory in the world — the moment your note leaves their hands, *poof*, they forget you. The next note? To them, you're a stranger.

So how do chatbots seem to remember? Let's prove the forgetting first.

## 🔨 `03_no_memory.py` — watch it forget
```python
"""
Module 3 - Step 4a: Prove the LLM has NO built-in memory.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

r1 = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "My name is Tanvi. Please remember it."}],
)
print("Bot:", r1.choices[0].message.content)

r2 = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What's my name?"}],
)
print("Bot:", r2.choices[0].message.content)   # It won't know!
```
**Line by line:** Call 1 sends *"My name is Tanvi."* Call 2 is a **separate** `.create()` whose `messages` contains *only* "What's my name?" — we sent **nothing** about Call 1. So the wizard has no idea, and replies something like "I don't have access to your name."

**Run:** `python 03_no_memory.py` → Call 2 fails to recall your name. Not because the AI is dumb — because **we never told it.**

## ✅ The trick: keep the transcript and resend it every time
> **The memory isn't in the AI. It's in YOUR code.** Keep the whole conversation in a Python **list** and resend the *entire list* each turn, so the wizard re-reads the full story before answering.

🧒 Since the wizard forgets instantly, *you* become their **notebook**. Each note you slide in re-writes the *whole conversation so far*. The wizard reads it fresh, answers, forgets — but you kept the notebook.

Two things to write in the notebook every turn: **(1)** what the user said (`role: user`), and **(2)** what the assistant replied (`role: assistant`) ← people forget this one, then the bot can't recall its *own* answers.

## 💻 `04_with_memory.py`
```python
"""
Module 3 - Step 4b: REAL memory - keep appending to the messages list.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

messages = [{"role": "system", "content": "You are a helpful assistant."}]   # the notebook

# Turn 1
messages.append({"role": "user", "content": "My name is Tanvi. Please remember it."})
r1 = client.chat.completions.create(model=MODEL, messages=messages)
answer1 = r1.choices[0].message.content
print("Bot:", answer1)
messages.append({"role": "assistant", "content": answer1})   # write the reply in too!

# Turn 2 (resend the WHOLE notebook)
messages.append({"role": "user", "content": "What's my name?"})
r2 = client.chat.completions.create(model=MODEL, messages=messages)
answer2 = r2.choices[0].message.content
print("Bot:", answer2)   # "Your name is Tanvi." ✅
messages.append({"role": "assistant", "content": answer2})
```

## 🔍 Line by line
- `messages = [{"role": "system", ...}]` → start the notebook with the rules. This list **is** the memory.
- **Turn 1:** append the user's line → send the whole notebook → ⭐ append the bot's reply (*the line everyone forgets*).
- **Turn 2:** append the new question (notebook now: system → "My name is Tanvi" → reply → "What's my name?") → send the whole list again → since the name is still in there, the wizard answers **"Your name is Tanvi."**

**Run:** `python 04_with_memory.py` → it remembers, purely because we kept and resent the notebook.

## 🔗 Two Module 2 callbacks (why that module mattered)
1. **The notebook grows forever** → eventually exceeds the **context window (the desk)** → oldest notes fall off → bot forgets the start of long chats. (Module 10 teaches trimming/summarizing — just managing this same list.)
2. **Every entry is tokens** → the whole notebook counts toward usage *each call*. Longer chat = more tokens per turn = the real cost of memory.

---

# PART 5 — Project 1: Your Interactive Chatbot 🤖

## 🤔 What's new vs Part 4?
In Part 4 we hardcoded the messages. A real chatbot must **keep asking you for input forever** until you quit. We wrap the notebook logic in a loop.

🧒 A tireless **receptionist**: ask → wait for your answer → reply → ask again → … until you say "done." Each round they jot both lines into the notebook. The loop that does "repeat until told to stop" is `while True`.

## 💻 `chatbot.py`
```python
"""
Project 1: Groq CLI Chatbot with Memory.
Run:  python chatbot.py    (type 'quit' to leave)
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

messages = [
    {
        "role": "system",
        "content": "You are a friendly, concise assistant. "
                   "Answer clearly and avoid rambling.",
    }
]

print("🤖 Chatbot ready! Type 'quit', 'exit', or 'bye' to leave.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit", "bye"):
        print("Bot: Goodbye! 👋")
        break

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    reply = response.choices[0].message.content

    print(f"Bot: {reply}\n")
    messages.append({"role": "assistant", "content": reply})
```

## 🔍 Line by line — only the new parts
(Setup, notebook, and the API call you already know.)
- `while True:` → 🧒 "repeat everything indented below me, **forever**" (the receptionist going round and round). Stops only via `break`.
- `user_input = input("You: ").strip()` → `input("You: ")` prints `You: ` and **pauses** for you to type + Enter; `.strip()` trims accidental spaces.
- `if user_input.lower() in ("quit","exit","bye"):` → `.lower()` makes "QUIT"/"Quit"/"quit" all match; checks for an escape word.
- `break` → 🧒 the magic word that **jumps out of the loop** — the only way to stop it.
- `if not user_input: continue` → if you pressed Enter with nothing typed, `continue` skips this round and asks again (no point sending an empty note).
- The three notebook lines (append user → `create` → append assistant) → your **exact Part 4 logic**, now running *inside the loop*, fresh each round. That's what keeps memory across a real conversation.
- `print(f"Bot: {reply}\n")` → f-string drops `{reply}` into the text; `\n` adds a blank line for readability.

## ▶️ Run & test memory
```bat
python chatbot.py
```
```
You: Hi! My name is Tanvi and I'm learning GenAI.
You: Suggest one fun beginner project for me.
You: Remind me — what's my name and what am I learning?
Bot: You're Tanvi, and you're learning GenAI!   ← memory works across the chat! ✅
You: quit
```
If it recalls your name, **you've built a working chatbot with memory.** 🎉

## 🛠️ Break it yourself
1. System message → `"You are a sarcastic pirate."` → personality flips (the power of the `system` role; Module 6 goes deep).
2. `temperature=0` → identical answers (🔗 greedy, Lab 3). `temperature=1.3` → wildly different.
3. Delete the assistant `append` line → the bot forgets its own replies (the amnesia bug, on purpose).

---

# PART 6 — Project 2: The Same Bot, Running Locally 🦙

## 🤔 Why build it twice?
So far the wizard lived in a faraway tower (Groq's data center) — needs internet, needs a key, your messages leave your machine. What if the wizard lived **inside your own house**? 🏠 Smaller and less brilliant, but free forever, fully private, works wifi-off. That's **Ollama**.

| | Cloud (Groq) | Local (Ollama) |
|---|---|---|
| Speed | ⚡ very fast | depends on your laptop |
| Brainpower | large, smart | smaller, simpler |
| Privacy | data leaves machine | 🔒 nothing leaves |
| Internet | required | not needed |
| Cost | free tier, then paid | 100% free forever |

## 🧰 Install Ollama (one-time)
1. **https://ollama.com/download** → run the Windows installer (installs a small background helper = your local "wizard's house").
2. New Command Prompt → check: `ollama --version`

## 🧰 Pull a small model (one-time)
🧒 "Pulling" = downloading the wizard's brain once. `llama3.2` is small (~2 GB), runs on normal laptops.
```bat
ollama pull llama3.2
```
Test it bare (no Python): `ollama run llama3.2` → ask something → `/bye` to exit. 🤯 That ran entirely on your machine.
> Try later: `ollama pull phi3`, `ollama pull mistral`, `ollama pull qwen2.5`.

## ✨ The magic: it speaks the SAME format
Ollama opens a local window at `http://localhost:11434/v1`. 🧒 `localhost` = "this computer, right here" — sliding the note to the wizard down the hall instead of mailing it across the internet. Same format → code is almost identical to Project 1.

```bat
cd C:\genai-bootcamp\module-03-free-llm-apis
mkdir project-02-ollama-chatbot
cd project-02-ollama-chatbot
python -m venv venv
venv\Scripts\activate
pip install openai
```

## 💻 `local_chatbot.py`
```python
"""
Project 2: The SAME chatbot, but the model runs LOCALLY via Ollama.
No API key. No internet needed. 100% private and free.
Run:  python local_chatbot.py    (type 'quit' to leave)
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",   # the local 'house', not the cloud
    api_key="ollama",                        # required by the library, ignored by Ollama
)

MODEL = "llama3.2"   # must match a model you pulled

messages = [
    {"role": "system", "content": "You are a friendly, concise assistant."}
]

print("🦙 Local Ollama Chatbot ready (running on YOUR machine)! Type 'quit' to leave.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "bye"):
        print("Bot: Goodbye! 👋")
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    reply = response.choices[0].message.content

    print(f"Bot: {reply}\n")
    messages.append({"role": "assistant", "content": reply})
```

## 🔍 Line by line — only what changed from Project 1
Almost everything is **identical** to `chatbot.py`. The only differences:
- `from openai import OpenAI` (instead of `from groq import Groq`) → we use the `openai` library. **Not** OpenAI's paid service — it just *speaks the OpenAI-compatible format* that Ollama also speaks. A translator that knows the right note format.
- `client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` → the whole trick:
  - `base_url=...` → "send notes to the wizard in *this* computer." `localhost:11434` is where Ollama listens.
  - `api_key="ollama"` → the library insists on a key, but Ollama doesn't check it, so any dummy text works. (No real key, no signup — it's your own machine.)
- `MODEL = "llama3.2"` → the local model you pulled (Ollama uses short names, no `-versatile`).
- **Everything else** — notebook, `while True`, `break`, `continue`, append user/assistant — is **exactly** Project 1. 🔗 You swapped the entire backend by changing **2 lines**.

## ▶️ Run (then pull the wifi! 😄)
```bat
python local_chatbot.py
```
Chat normally, then **turn off your wifi** and keep chatting — it still works, because the wizard lives in your house now. 🏠
**Observe the trade-offs, now real:** likely **slower** (laptop ≠ data center) and **less sharp** (`llama3.2` is small), but **100% private, free, offline.**

---

# 🪢 The big comparison (lock this in)

| | **Project 1 — Groq (cloud)** | **Project 2 — Ollama (local)** |
|---|---|---|
| Where the AI lives | Faraway data center | Your own laptop |
| Speed | ⚡ very fast | depends on laptop |
| Brainpower | large, smart (Llama 3.3 70B) | smaller (Llama 3.2) |
| Privacy | data leaves machine | 🔒 nothing leaves |
| Internet | required | not needed |
| Cost | free tier, then paid | 100% free forever |
| **Code difference** | `Groq()` client | `OpenAI(base_url=localhost)` — **~2 lines!** |

# 🧠 The three things from Module 3 that matter most
1. **An API is a window** — your code slides a note (request) through and gets one back (response). You did it raw, then let the SDK tidy it.
2. **The AI has no memory.** *Your code* remembers, by keeping a notebook (the `messages` list) and resending the whole thing each turn. This explains every chatbot ever.
3. **Cloud vs local is a real choice** — speed/brainpower vs privacy/cost/offline. You built both.

---

## 🔧 Setup recap (everything Module 3 needs)
```bash
# Cloud project
pip install groq python-dotenv requests
# Local project
pip install openai
# Plus: install Ollama from ollama.com, then  ollama pull llama3.2
```

## 🛠️ Commit to GitHub
```bash
cd C:\genai-bootcamp
git add .
git commit -m "Module 3: Groq cloud chatbot + Ollama local chatbot (with memory)"
git push
```
> 🔒 After pushing, confirm `.env` is **not** in the file list on github.com.

## ⚠️ Common Mistakes
1. **Hardcoding the API key** (or pushing `.env`). Always use `.env` + `.gitignore`.
2. **Forgetting to append the assistant's reply** → the bot forgets its own answers.
3. **Forgetting to activate the venv** (`venv\Scripts\activate`) → `ModuleNotFoundError`. Look for `(venv)`.
4. **Using a deprecated model ID** → "model not found." Check the live model list and swap it.
5. **Expecting the local model to match Groq's quality** → it won't; it's far smaller. That's expected.
6. **Letting `messages` grow forever** → eventually blows past the context window (fine for learning; fixed in Module 10).

## 💼 Real Production Example
Speed-sensitive products (live voice assistants, real-time chat) route to fast cloud APIs like Groq, where low latency is the difference between "feels alive" and "feels sluggish." Privacy-sensitive setups (healthcare, internal tools, offline) run open models locally via Ollama so no data leaves the building. Many real apps do both. You built one of each.

## 📋 Decision Table — which provider when?
| Situation | Pick |
|---|---|
| Default chat / fast prototyping | **Groq** |
| Private data / offline / no rate limits | **Ollama** |
| Image understanding, very long documents | **Gemini** (Module 25) |
| A/B testing many models with one key | **OpenRouter** |
| You truly need top-tier proprietary capability | OpenAI / Claude (paid) |

## 💬 Plain English Summary
An **API** is a window your code uses to pass a note (request) to an AI and get an answer (response). The note is a **list of messages** with `system`/`user`/`assistant` roles, plus settings like `temperature`. Under the hood it's just an HTTP POST with JSON (you did it raw), and the SDK just tidies that up. **The LLM has no memory** — *you* create memory by keeping the messages list (a notebook) and resending it each turn (which costs tokens and fills the context window). You built the same chatbot on **Groq** (cloud, fast, free-tier) and **Ollama** (local, private, free, offline), and the code barely changed because both speak the OpenAI-compatible format.

## 👨‍🏫 Interview Tips
- **The three roles?** `system` sets behavior (once, top); `user` is the human's input; `assistant` is the model's output, which you store to give memory.
- **How does a chatbot remember if the LLM is stateless?** The app keeps the transcript in a list and resends it every turn; memory lives in the app, not the model. History must eventually be trimmed/summarized for the context window.
- **Why run a model locally?** Privacy/compliance, no per-token cost, no rate limits, offline. Trade-off: limited to models your hardware can run (usually smaller).
- **What is "OpenAI-compatible"?** Many providers expose the same request/response format, so you can switch providers by changing the base URL + model name without rewriting code.
- **What is `temperature`?** Controls randomness in token selection (low = precise/repeatable, high = creative). Ties to the softmax sampling from Module 2.

## 🔥 Cross-Questions (test yourself)
1. Your bot answered once correctly, then got confused on a follow-up. Name two different bugs that could cause this.
2. A friend says "just hardcode the key, it's faster." One-sentence reason that's bad.
3. Project 1 and 2 had nearly identical code — *why* was the swap so easy?
4. Building for a hospital that must never send patient data outside — which project is the right starting architecture, and what do you give up?

## ✅ Quick Check
1. What are the three message roles and what is each for?
2. Where does a chatbot's "memory" actually live — the model or your code?
3. What two files keep your API key from leaking to GitHub?
4. Name one thing Groq gives you that Ollama doesn't, and vice versa.
5. What does `temperature=0` do to the answers?

## 📚 Best Free Resources
- **Groq Docs — Models & Quickstart** (`console.groq.com/docs`): source of truth for model IDs + the endpoint.
- **Ollama README + ollama.com/library**: install + every local model you can pull.
- **Google AI Studio docs** (`ai.google.dev`): Gemini free-tier quickstart (used in Module 25).
- **OpenAI Python SDK README**: since Groq/Ollama are OpenAI-compatible, it doubles as their reference.
- **YouTube:** search "Groq API Python tutorial" and "Ollama Python tutorial" for short, current walkthroughs.

---

*End of Module 3. Next up (when you're ready): Module 6 — Prompt Engineering is the recommended high-payoff next step (it supercharges the chatbots you just built); Module 4 (OpenAI, paid) and Module 5 (Claude) are optional.*
