# 📘 Module 4 — OpenAI API (Complete Notes, with Outputs)

> **Difficulty:** 🟡 Medium · **Time:** 5–7 hours · **Prerequisites:** Modules 1–3
> **What you built:** mastery of the four "power skills" — parameters, streaming, function calling, structured outputs — plus **Project 3: a structured data extractor.**
> **Cost:** ₹0 — we practiced everything **free on Groq**, which speaks the exact same API as OpenAI.

**How to read these notes:** each section has the concept → the code → and **🖥️ the output it produces**, so you can match every line to what actually happens. LLM text varies run-to-run, so treat outputs as *representative examples*.

---

## 🗺️ The big idea of Module 4

The OpenAI API is the most famous LLM API — but unlike Groq/Ollama, it's **paid** (no real free tier; prepaid credits; ~$5 to unlock decent limits). The wonderful catch: **OpenAI invented the API format everyone else copied.** So the `messages` list, the three roles, `client.chat.completions.create(...)`, `temperature` — everything from Module 3 *is* the OpenAI format.

🧙 Same window, same note format — just a different (pricier) tower. That's why we practiced **free on Groq** and learned the real thing for nothing.

**What Module 4 actually added on top of Module 3:**

```
Module 3 gave you:  basic chat  +  memory
Module 4 adds:      [1] deeper PARAMETERS (knobs)
                    [2] STREAMING (answers that type themselves)
                    [3] FUNCTION CALLING (the AI uses YOUR code)  ← the big one
                    [4] STRUCTURED OUTPUTS (guaranteed clean JSON)
                    →  Project 3: messy text → validated JSON
```

> 🤖 *Current models (changes often):* OpenAI's flagship is now the GPT-5.5 family, with cheaper GPT-5.4 / mini / nano tiers and budget GPT-4.1 models. The API format is unchanged, so this code runs on any of them by swapping `model=`. For practice we just used Groq's free Llama.

---

# PART 1 — The Knobs (parameters beyond temperature)

You already mastered `temperature` in Module 2. Here are the other settings you can attach to a request:

| Knob | What it does |
|---|---|
| `max_tokens` | Hard **ceiling** on answer length (in tokens). |
| `stop` | **Trip wires** — stops the moment it writes any of these strings. |
| `seed` | A **best-effort** nudge toward repeatable output (see the big lesson below). |
| `top_p` | The **nucleus** knob from Lab 3, set in code. |
| `frequency_penalty` | Discourages **repeating** words. |
| `presence_penalty` | Encourages **new topics**. |
| `n` | Ask for **several answers** at once. |

### 💻 `05_parameters.py`
```python
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

def ask(label, prompt, **kwargs):          # reusable helper; **kwargs = "bag of knobs"
    print(f"\n===== {label} =====")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    print(resp.choices[0].message.content)
    print(f"(completion tokens: {resp.usage.completion_tokens})")

ask("max_tokens = 25 (gets cut off)",
    "Write a detailed paragraph about why the ocean matters.", max_tokens=25)

ask("stop = ['5'] (counting stops around 5)",
    "Count from 1 to 10, one number per line.", stop=["5"])
```

### 🖥️ Output
```
===== max_tokens = 25 (gets cut off) =====
The ocean matters in a multitude of ways, playing a vital role in sustaining life on Earth. Not only does it cover over
(completion tokens: 25)

===== stop = ['5'] (counting stops around 5) =====
1
2
3
4
(completion tokens: 20)
```

**Reading the output line-by-line:**
- `**kwargs` → 🧒 a "bag for extra settings." Whatever knobs you pass to `ask(...)` get handed straight to the API.
- **`max_tokens=25`** → the paragraph **chops off mid-sentence** ("...does it cover over") and `completion tokens: 25` proves it hit the ceiling exactly. ⚠️ `max_tokens` does **not** mean "be brief" — it just *cuts*. For a short *complete* answer, ask for that in the prompt.
- **`stop=["5"]`** → counting halts at `4`, the instant before it would write `5`. Use this to stop at a clean delimiter.
- 🔧 *OpenAI note:* OpenAI's newest models call this `max_completion_tokens`. Same idea; Groq uses `max_tokens`.

### 🧠 The big lesson: `seed` is NOT a guarantee
We tested `seed=42` at `temperature=0.8` and got **different** answers on two runs. The honest truth:

> **`seed` is "best-effort," not a promise.** Reproducibility also needs the provider's backend ("system fingerprint") to stay identical, and high temperature amplifies tiny differences. Neither OpenAI nor Groq guarantees identical output from a seed.

✅ **The reliable way to get repeatable output is `temperature=0`** (greedy from Lab 3 — no dice-rolling). Even then it's *much* more stable, not bit-for-bit guaranteed.

> 💼 **Production takeaway:** LLM outputs aren't perfectly reproducible — so automated tests shouldn't assert `answer == "exact string"`. Test for *properties* instead ("is it valid JSON?", "does it contain the right number?").

---

# PART 2 — Streaming (answers that type themselves)

```
NON-STREAMING:  send → [ ....... you wait ....... ] → WHOLE ANSWER drops in at once
STREAMING:      send → W→o→r→d→s →f→l→o→w →i→n →l→i→v→e →as they're written...
```

🧙 Non-streaming = the wizard seals the finished letter and slides it back only when done. Streaming = the wizard slides it out **one word at a time as they write**. That's the ChatGPT "typing" effect — the wait *feels* shorter. The switch is one setting: **`stream=True`**.

### 💻 `06_streaming.py` (the streaming half)
```python
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a short, upbeat paragraph about learning to code."}],
    stream=True,                       # the ONLY new setting
)
for chunk in stream:
    piece = chunk.choices[0].delta.content or ""
    print(piece, end="", flush=True)
print()
```

### 🖥️ Output (appears word-by-word, live)
```
Learning to code is one of the most rewarding skills you can pick up! Every small
program you build is a tiny win, and those wins add up fast. Stick with it, stay
curious, and soon you'll be turning ideas into real, working tools. 🚀
```

**Reading it line-by-line:**
- `stream=True` → "don't wait — slide pieces out as you write."
- `for chunk in stream:` → instead of one finished reply, you loop over a **stream of little pieces** ("chunks").
- `chunk.choices[0].delta.content` → 🔗 the key change: normal mode reads `.message.content` (whole answer); streaming gives each new bit at **`.delta.content`** ("delta" = the new piece).
- `or ""` → the first/last chunks are empty (`None`); `or ""` avoids a crash.
- `print(piece, end="", flush=True)` → `end=""` glues words together (no line breaks); `flush=True` forces each piece to show **instantly** — that's what makes it "type."
- 🔗 This is the exact pattern behind the optional streaming upgrade in your Project 1 chatbot.

---

# PART 3 — Function Calling (the AI uses YOUR code) ⭐

🔗 An LLM is a next-word predictor — bad at exact math, blind to live data, can't see *your* database. Ask "what's Alice's balance?" and it would **hallucinate** a number. The fix: give the wizard **tools (bells)** it can *ask you* to ring.

> ⭐ The wizard never runs the tool. It only *requests* it. **Your code** does the real work. An "agent" (Phase 5) is basically this in a loop.

### 💃 The dance (visual)
```
   YOU                              THE WIZARD (LLM)                 YOUR PYTHON
    │                                     │                              │
 1  │  question + tool menu               │                              │
    ├────────────────────────────────────▶                              │
 2  │   "ring get_account_balance         │                              │
    │    with user_id='alice'"            │                              │
    ◀────────────────────────────────────┤                              │
 3  │  run the REAL function ───────────────────────────────────────────▶
    │                                     │        "$1240.50"            │
    ◀───────────────────────────────────────────────────────────────────┤
 4  │  here's the tool result             │                              │
    ├────────────────────────────────────▶                              │
 5  │   "Alice's balance is $1240.50"     │                              │
    ◀────────────────────────────────────┤                              │
```
Two trips to the wizard: *ask (with tools) → it requests a tool → you run it → you send the result → it answers.*

### 💻 `07_function_calling.py`
```python
import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

FAKE_DB = {"alice": 1240.50, "bob": 89.00, "carol": 5300.75}      # 1) the real tool
def get_account_balance(user_id):
    balance = FAKE_DB.get(user_id.lower())
    if balance is None:
        return f"No account found for '{user_id}'."
    return f"{user_id}'s balance is ${balance:.2f}"

tools = [{                                                         # 2) describe the bell
    "type": "function",
    "function": {
        "name": "get_account_balance",
        "description": "Get the current account balance for a given user id.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "e.g. 'alice'"}},
            "required": ["user_id"],
        },
    },
}]

messages = [{"role": "user", "content": "What is Alice's account balance?"}]

response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)  # TRIP 1
reply = response.choices[0].message
messages.append(reply)

if reply.tool_calls:
    for call in reply.tool_calls:
        args = json.loads(call.function.arguments)               # args arrive as JSON text
        print(f"🔔 Model wants to call: {call.function.name}({args})")
        result = get_account_balance(**args)                     # WE run it
        print(f"🛠️  Tool result: {result}")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    final = client.chat.completions.create(model=MODEL, messages=messages)              # TRIP 2
    print("\n🤖 Final answer:", final.choices[0].message.content)
else:
    print("🤖 Answer (no tool needed):", reply.content)
```

### 🖥️ Output
```
🔔 Model wants to call: get_account_balance({'user_id': 'Alice'})
🛠️  Tool result: Alice's balance is $1240.50

🤖 Final answer: Alice's account balance is $1240.50.
```

**Reading the output line-by-line:**
- Line 1 (`🔔`) → the wizard **rang the bell** — it asked for the tool with `user_id='Alice'` instead of guessing a number.
- Line 2 (`🛠️`) → **your Python function** did the real lookup ($1240.50).
- Line 3 (`🤖`) → on Trip 2, the wizard wrapped your real result in a natural sentence.
- `tools=...` (the JSON schema) = the **menu**: `name` (must match your function), `description` (when to use it), `parameters` (inputs, in JSON Schema). 🔗 *Brute-force → shortcut:* you write this by hand now; frameworks (LangChain) auto-generate it later.
- `json.loads(call.function.arguments)` → the model sends args as **JSON text**; parse to a dict.
- `get_account_balance(**args)` → `**args` unpacks `{"user_id":"Alice"}` into the call.
- `{"role": "tool", ...}` → the **4th role**; `tool_call_id` links the result to the exact bell.

> **Try:** `"What is 2+2?"` → it skips the tool (the `else` branch). `"What's Dave's balance?"` → your function says "No account found," and the wizard relays it honestly — **no hallucinated number.** That's the whole point.

---

# PART 4 — Structured Outputs (guaranteed clean JSON)

Your code can't use chatty prose; it needs **data**. 🧙 Make the wizard *fill in a form*, not write a story. Earned the usual way:

```
1. BRUTE FORCE  → "please reply in JSON"   → sometimes fenced/preamble → fragile
2. JSON MODE     → response_format=json     → ALWAYS valid JSON          → reliable
3. PYDANTIC      → define the exact shape    → validated + typed          → trustworthy
```

### 💻 `08_structured_outputs.py`
```python
import os, json
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
TEXT = "Tanvi Yadav is 25 years old and works as a data analyst in Delhi."

# 2) JSON MODE
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You output ONLY valid JSON, no extra text."},
        {"role": "user", "content": f"Extract keys name, age, job, city as JSON from: {TEXT}"},
    ],
    response_format={"type": "json_object"},     # the magic switch
)
data = json.loads(resp.choices[0].message.content)
print(resp.choices[0].message.content)
print("Parsed -> name:", data["name"], "| city:", data["city"])

# 3) PYDANTIC
class Person(BaseModel):
    name: str
    age: int        # forces a number, even if model sent "25" as text
    job: str
    city: str
person = Person(**data)
print("Validated ->", person)
```

### 🖥️ Output
```
{"name": "Tanvi Yadav", "age": 25, "job": "data analyst", "city": "Delhi"}
Parsed -> name: Tanvi Yadav | city: Delhi
Validated -> name='Tanvi Yadav' age=25 job='data analyst' city='Delhi'
```

Compare with **brute force**, which might give:
```
Here's the JSON you requested:
```json
{ "name": "Tanvi Yadav", "age": 25, "job": "data analyst", "city": "Delhi" }
```
```
↑ That preamble + ` ```json ` fence is exactly what makes `json.loads()` crash — and exactly what JSON mode removes.

**Reading it line-by-line:**
- `response_format={"type": "json_object"}` → forces the **whole** reply to be valid JSON (no fences, no preamble). ⚠️ The word "JSON" must appear in your messages (the system line covers it).
- `json.loads(...)` → clean text → Python dict; `data["name"]` just works.
- `class Person(BaseModel)` → 🧒 defines the **form**; `age: int` converts `"25"` → `25` if needed; `Person(**data)` validates and **errors loudly** if a field is missing/wrong, so bad data never sneaks through.
- 🔧 *OpenAI note:* OpenAI can *guarantee* a schema with "strict" structured outputs; the portable recipe everywhere is **JSON mode + Pydantic** (what we did).

---

# 🛠️ PROJECT 3 — The Structured Data Extractor

**Problem:** businesses drown in messy text (support tickets, résumés, invoices). We auto-convert it to clean, validated, database-ready JSON.

### Pipeline
```
Messy support message ──▶ extract() ──▶ Groq (JSON mode) ──▶ Pydantic validate ──▶ clean JSON
   "Ravi, order #88231                                                              {category, urgency,
    hasn't arrived, 3 wks!"                                                          name, order_no, summary}
```

### 💻 `project3_extractor.py`
```python
import os, json
from enum import Enum
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

class Category(str, Enum):                 # a "dropdown" - only these values allowed
    shipping = "shipping"; refund = "refund"; technical = "technical"
    account = "account"; other = "other"

class Urgency(str, Enum):
    low = "low"; medium = "medium"; high = "high"

class Ticket(BaseModel):
    customer_name: Optional[str] = None    # might be absent
    order_number: Optional[str] = None
    category: Category
    urgency: Urgency
    summary: str

def extract(message: str) -> Ticket:
    system = (
        "You extract structured data from customer support messages and reply ONLY "
        "with valid JSON. Use these exact keys: customer_name (string or null), "
        "order_number (string or null), category (one of: shipping, refund, technical, "
        "account, other), urgency (one of: low, medium, high), summary (one short sentence)."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return Ticket(**json.loads(resp.choices[0].message.content))

inbox = [
    "Hi, this is Ravi. My order #88231 still hasn't shown up and it's been 3 weeks!! Really frustrated.",
    "I'd like a refund on my subscription please. It auto-renewed and I didn't want it.",
    "getting error E-401 every time i try to log in, can someone help asap",
    "Does your app work on Linux? Just curious before I buy.",
]

for msg in inbox:
    print("\n" + "=" * 60)
    print("MESSAGE:", msg)
    try:
        print("EXTRACTED:\n", extract(msg).model_dump_json(indent=2))
    except ValidationError as e:
        print("⚠️ Validation failed:", e)
```

### 🖥️ Output
```
============================================================
MESSAGE: Hi, this is Ravi. My order #88231 still hasn't shown up and it's been 3 weeks!! Really frustrated.
EXTRACTED:
 {
  "customer_name": "Ravi",
  "order_number": "88231",
  "category": "shipping",
  "urgency": "high",
  "summary": "Order #88231 has not arrived after 3 weeks."
}
============================================================
MESSAGE: I'd like a refund on my subscription please. It auto-renewed and I didn't want it.
EXTRACTED:
 {
  "customer_name": null,
  "order_number": null,
  "category": "refund",
  "urgency": "medium",
  "summary": "Customer requests a refund for an unwanted auto-renewal."
}
============================================================
MESSAGE: getting error E-401 every time i try to log in, can someone help asap
EXTRACTED:
 {
  "customer_name": null,
  "order_number": null,
  "category": "technical",
  "urgency": "high",
  "summary": "Login fails with error E-401."
}
============================================================
MESSAGE: Does your app work on Linux? Just curious before I buy.
EXTRACTED:
 {
  "customer_name": null,
  "order_number": null,
  "category": "other",
  "urgency": "low",
  "summary": "Pre-sales question about Linux support."
}
```

**Reading the output:**
- Notice every `category`/`urgency` is a **valid dropdown value** — never invented — because the `Enum` + Pydantic enforce it.
- Missing fields safely become `null` thanks to `Optional[...] = None` (the refund/error/Linux messages have no name or order number).
- `temperature=0` → 🔗 precise extraction, not creative writing.
- `try/except ValidationError` → one weird message can't crash the whole batch (production habit).

### 🛠️ Break it yourself
- Add a `sentiment` Enum (`positive`/`neutral`/`negative`) and watch it classify mood.
- Feed total nonsense and see it default to `other`.

### 📦 Commit
```bash
git add .
git commit -m "Module 4 Project 3: structured data extractor (messy text -> validated JSON)"
git push
```

---

## ⚠️ Common Mistakes
1. **Treating `seed` as a reproducibility guarantee.** It isn't — use `temperature=0`, and test for properties not exact strings.
2. **Forgetting "JSON" in the prompt** when using JSON mode → the API errors.
3. **Assuming the model runs your tool.** It only *requests* it; your code runs it and sends the result back.
4. **No `Optional`/defaults on fields that may be absent** → Pydantic crashes on real, incomplete data.
5. **High temperature for extraction** → use `0` for precise, repeatable structured output.

## 📋 Decision Table
| You want… | Use |
|---|---|
| Control answer length / cost | `max_tokens` |
| Stop at a delimiter | `stop` |
| Repeatable output | `temperature=0` |
| A live UX (typing effect) | `stream=True` |
| AI to do math / fetch data / hit your DB | **function calling** |
| Clean machine-usable data | **JSON mode + Pydantic** |

## 👨‍🏫 Interview Tips
- **What is function calling?** The model doesn't run tools; it returns a structured request to call a named function with arguments. Your app executes it and feeds the result back, then the model answers. Foundation of agents.
- **Why structured outputs?** To get reliable, machine-readable data instead of prose. JSON mode forces valid JSON; a schema (Pydantic / strict mode) enforces the exact shape.
- **Is `seed` deterministic?** No — best-effort only. Reproducibility also depends on the backend; `temperature=0` is the reliable lever and still not bit-for-bit guaranteed.
- **What does "OpenAI-compatible" mean?** Many providers expose OpenAI's exact request/response format, so you switch providers by changing base URL + model name.
- **Streaming vs non-streaming?** Streaming returns incremental `delta` chunks for a live typing UX; non-streaming returns the whole `message` at once.

## 🔥 Cross-Questions
1. A user asks your bot for today's exchange rate. Why will plain chat fail, and which Module 4 skill fixes it?
2. Your extractor crashes on one message in a batch of 1,000. What single pattern prevents the whole job from dying?
3. Why is `temperature=0` better than `seed` for repeatable extraction?
4. You need the model to pick a *category* — why is an Enum better than a free-text string field?

## ✅ Quick Check
1. What does `max_tokens` actually do (and not do)?
2. In streaming, where does the new text live — `.message.content` or `.delta.content`?
3. In the function-calling dance, who runs the tool?
4. What two things turn messy text into trustworthy JSON?
5. What's the reliable way to get repeatable output?

## 📚 Best Free Resources
- **OpenAI Docs — Function calling & Structured Outputs** (`platform.openai.com/docs`): the canonical reference (concepts apply to Groq too).
- **Groq Docs — Tool use & JSON mode** (`console.groq.com/docs`): what works on our free stack.
- **Pydantic docs** (`docs.pydantic.dev`): models, validation, types.
- **YouTube:** search "OpenAI function calling tutorial" and "Pydantic crash course."

---

*End of Module 4. You now hold the four power-skills (parameters, streaming, function calling, structured outputs) + a real extractor — all learned free on Groq, in the real OpenAI format.*
