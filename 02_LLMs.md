# 📘 Module 2 — How LLMs Work (Complete Notes)

> **Difficulty:** 🟡 Medium · **Time:** 6–8 hours · **Prerequisites:** Module 1 (the 4-layer GenAI stack)
> **What you'll learn:** what's actually happening inside an LLM — how it reads text (tokens), how it "thinks" (attention), how it writes (the generation loop + temperature/top-k/top-p), how it's built (training stages), what limits it (context window, model size), and why it lies sometimes (hallucination). Four runnable labs included, each explained line by line in plain English.

**How to read these notes:** go top to bottom, and when a lab appears, *run it* before moving on. The labs are where the ideas click. Nothing here needs heavy math.

---

## 🧒 The one-sentence secret of LLMs

You know how your phone suggests the next word while you type? *"I'm going to the…"* → *store, park, movies.*

**An LLM is that suggestion box — trained on basically the whole internet, and so good it can keep suggesting word after word until it has written a whole story, email, or computer program.**

People think ChatGPT is "thinking." Really it plays one game, billions of times, very fast:

> **"Given all the words so far, what's the most sensible next word?"**

Pick a word → add it → ask again with the longer sentence → pick the next → repeat. That loop, running on a giant brain, *looks* like thinking.

**Why this simple game produces intelligence:** finish these in your head — *"Twinkle twinkle little ___"*, *"The capital of France is ___"*, *"2 plus 2 equals ___"*. You answered *star, Paris, four*. To guess the next word well, you were forced to "know" rhymes, geography, and math. So a great next-word predictor secretly has to learn facts, grammar, and reasoning. That's the whole magic.

**Adult version:** imagine someone who read everything ever written but remembers no single source — they absorbed the *patterns*. They fluently continue any text in the most likely way. Superpower: fluent, knows a bit of everything. Flaw: it predicts what *sounds* right, not what *is* right (see Hallucination at the end).

---

# PART A — TOKENIZATION (turning text into numbers)

## Why tokens exist

A neural network is a giant pile of math, and math needs **numbers** — but we feed it **text**. So step one of everything is converting text into numbers. Those numbered chunks are called **tokens**. The real question is *how* to chop text into pieces. Let's try the obvious ideas first and watch them fail (this is the "brute force before the shortcut" habit — understand *why* the real tool exists).

### 🔨 Brute-force idea #1 — split by spaces (one word = one token)
`"I love GenAI"` → `["I", "love", "GenAI"]`, number each. Simple!
**Why it breaks:** punctuation glues on (`"amazing!!"` becomes one weird token); every rare/new word or typo (`"GenAI"`, `"amazzzing"`) would need its own number, so the dictionary would have to be *infinite*; and any word never seen before becomes `[UNKNOWN]` and loses all meaning.

### 🔨 Brute-force idea #2 — split by single characters
`"love"` → `["l","o","v","e"]`. Tiny dictionary, no unknown words ever!
**Why it breaks:** a single letter means almost nothing, so the model works much harder; and sentences become *enormously* long (≈ one token per character), which is slow and expensive (cost scales with token count).

### ✅ The real solution — subword tokenization (BPE)
Don't choose between whole-words and single-letters — **let the data decide.** Scan huge amounts of text, find the **most common chunks**: frequent whole words (`the`, `love`) become single tokens; rare/new words break into familiar *pieces*.

```
"tokenization" → ["token", "ization"]
"GenAI"        → ["Gen", "AI"]
"amazzzing"    → ["amaz", "zz", "ing"]
```

The algorithm is **BPE (Byte-Pair Encoding)**. In plain English:
1. Start with single characters.
2. Find the pair that appears together most often (e.g. `t`+`h` → `th`).
3. Merge it into one new symbol; add to the vocabulary.
4. Repeat thousands of times — common pairs keep merging into bigger chunks.

Result = **best of both worlds**: common stuff is compact, and nothing is ever truly "unknown" (worst case, it falls back to characters). **That's why every major LLM uses subword tokenization.**

### 🧠 Token facts that save you pain later
- **Tokens ≈ ¾ of a word in English.** 100 tokens ≈ 75 words. 1,000 tokens ≈ ~1.5 pages.
- A space is usually part of the token (`" the"` with a leading space = one token).
- Numbers, code, and non-English text cost *more* tokens per character.
- **You pay per token, and context limits are measured in tokens** — this single fact drives a lot of GenAI engineering.

---

## 💻 LAB 1 — Tokenize the naive (brute-force) way

No installation. Pure Python. Save as `01_naive_tokenizer.py`:

```python
"""
Module 2 - Lab 1: The NAIVE ways to turn text into tokens (BRUTE FORCE).
"""
text = "I'm learning GenAI, and it's amazing!! 🚀"

# ATTEMPT 1: split by spaces
print("=== Attempt 1: Split by spaces ===")
tokens_by_space = text.split(" ")
print(tokens_by_space)
print("Number of tokens:", len(tokens_by_space))

# ATTEMPT 2: split by every character
print("\n=== Attempt 2: Split by characters ===")
tokens_by_char = list(text)
print(tokens_by_char)
print("Number of tokens:", len(tokens_by_char))

# ATTEMPT 3: words AND punctuation kept separate
print("\n=== Attempt 3: Crude word + punctuation split (regex) ===")
import re
tokens_regex = re.findall(r"\w+|[^\w\s]", text)
print(tokens_regex)
print("Number of tokens:", len(tokens_regex))
```

### Line by line, in plain English
- `text = "..."` → the sentence we're going to chop up.
- `text.split(" ")` → `.split(" ")` cuts the string wherever it finds a space, giving a **list of words**. Result: `"amazing!!"` stays as one messy blob (punctuation attached), and `"I'm"`/`"it's"` are single blobs.
- `len(...)` → just counts how many pieces we got.
- `list(text)` → turns the string into a list of **single characters**. Now `"learning"` is 8 separate items — way too many, each almost meaningless.
- `re.findall(r"\w+|[^\w\s]", text)` → a regular expression (pattern matcher). `\w+` means "grab a run of letters/digits", `[^\w\s]` means "grab a single punctuation mark", and `|` means "either one". So it neatly separates words from punctuation. Better! But it still can't handle a brand-new word like `"GenAI"` gracefully — a pure word list would call it unknown.

**Run it:** `python 01_naive_tokenizer.py`
**Observe:** none of the three feels right. That discomfort is *exactly* the problem BPE was invented to fix.

---

## 💻 LAB 2 — Tokenize the real way (`tiktoken`)

Install once: `pip install tiktoken`. Save as `02_tiktoken_demo.py`:

```python
"""
Module 2 - Lab 2: The REAL way - subword tokenization with tiktoken.
tiktoken is the EXACT tokenizer OpenAI's GPT models use.
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # tokenizer for GPT-3.5 & GPT-4
text = "I'm learning GenAI, and it's amazing!! 🚀"

token_ids = enc.encode(text)                 # text -> numbers
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))

print("\nWhat each token really is:")
for tid in token_ids:
    print(f"  {tid:>7}  ->  {enc.decode([tid])!r}")

print("\nDecoded back to text:", enc.decode(token_ids))   # reversible

print("\nCharacters:", len(text), "| Tokens:", len(token_ids))

enc_4o = tiktoken.get_encoding("o200k_base")  # tokenizer for GPT-4o
print("GPT-4 tokens:", len(enc.encode(text)), "| GPT-4o tokens:", len(enc_4o.encode(text)))
```

### Line by line, in plain English
- `import tiktoken` → load OpenAI's tokenizer library.
- `enc = tiktoken.get_encoding("cl100k_base")` → pick a specific tokenizer. `cl100k_base` is the exact one GPT-3.5 and GPT-4 use. (`o200k_base` is GPT-4o's.) Different models can chop text differently.
- `token_ids = enc.encode(text)` → `.encode()` turns the sentence into the **list of ID numbers** the model actually sees.
- the `for` loop with `enc.decode([tid])` → `.decode()` turns numbers back into text. Passing one ID at a time shows you the **exact piece of text** behind each token. (Notice common words are 1 token and the leading space is part of it, like `' learning'`.)
- `enc.decode(token_ids)` → decode the whole list at once. You get the original sentence back perfectly — tokenization is **reversible**.
- the last two lines → compare token counts across two tokenizers; same sentence, slightly different counts.

**Run it:** `python 02_tiktoken_demo.py`
**Observe:** common words = 1 token; the emoji 🚀 splits into several tokens (emojis are expensive!); token count sits *between* the character count and the word count — the sweet spot we argued for.

**🔧 Break it yourself:** paste a Hindi paragraph and compare tokens-per-character vs English (non-English costs more); tokenize Python code; tokenize `"strawberry"` and look at the pieces — *that's* why LLMs historically miscount its letters (they see chunks, not letters).

---

# PART B — INSIDE THE BOX (Transformers & Attention)

## From tokens to "meaning," then attention

After tokenizing, each token becomes a long list of numbers called an **embedding** — its "meaning coordinates." Similar meanings get similar numbers. (Module 7 is entirely about embeddings; for now just hold: *token → numbers that capture meaning*.)

Then comes the breakthrough that made modern AI possible: **attention.**

### 🧒 Attention, for a 10-year-old
Read: **"The animal didn't cross the road because *it* was too tired."** Does *"it"* mean the animal or the road? The **animal** (roads don't get tired). Your brain *paid attention* to the right earlier word.

Now: **"…because *it* was too wide."** Now *"it"* = the **road**. Same word, different meaning — you figured it out by looking back at the relevant word.

**Attention is the LLM doing exactly that.** For every word, it asks *"which earlier words should I focus on to understand this one?"*, scores each one's importance (a "weight"), and blends them. That's how it tracks context, pronouns, and long-range meaning.

### Why "Attention Is All You Need" changed everything
That was the title of the 2017 Google paper that introduced the **Transformer**. Before it, models read text one word at a time, in order — slow, and they forgot the start of long sentences. The new idea: let every word look at every other word **at once, in parallel**. Two huge results:
1. **Better understanding** of long-range links ("it" → "animal", even far apart).
2. **Parallel training** across thousands of GPUs at once — the only reason giant models became possible. This engine is the "Transformer," the **T** in **GPT** (Generative Pre-trained **Transformer**).

### Three flavors of Transformer (so the names never confuse you)
| Type | Famous example | Good at | Mental model |
|---|---|---|---|
| **Decoder-only** | GPT, Llama, Mistral, Claude, Gemini | **Generating** text left→right | A writer choosing the next word |
| **Encoder-only** | BERT | **Understanding/classifying** (no generating) | A reader compressing text into meaning |
| **Encoder-Decoder** | T5 | **Transforming** input→output (e.g. translate) | A translator: read all of A, produce B |

**99% of the chatbots and agents you'll build use decoder-only models.** Encoder-only (BERT) returns in Module 7 for embeddings. You don't need the internal math — the intuition above is enough.

---

# PART C — HOW TEXT IS GENERATED (the loop + the knobs)

## The autoregressive loop (a scary word for a simple loop)
```
1. Take all the tokens so far (the prompt).
2. The model outputs a SCORE for every possible next token.
3. Choose ONE token from those scores.   <-- temperature / top-k / top-p live here
4. Stick it on the end.
5. Go back to step 1 with the longer text.
6. Stop at a stop-signal or max length.
```
Step 3 — *how you choose* — is everything. Lab 3 below shows it with real numbers.

---

## 💻 LAB 3 — See the knobs with your own eyes (the most important lab)

A *fake* tiny vocabulary so you can literally watch each knob reshape the choice. Install once: `pip install numpy`. Save as `03_sampling_from_scratch.py`:

```python
"""
Module 2 - Lab 3: How an LLM PICKS the next word (BRUTE FORCE -> SHORTCUTS).
"""
import numpy as np

# Pretend the model just read: "The weather today is ___"
vocab  = ["sunny", "rainy", "cloudy", "hot", "cold", "banana", "purple", "running"]
logits = np.array([3.0,    2.5,     2.0,     1.5,   1.0,    -1.0,    -2.0,      -3.0])

def softmax(x, temperature=1.0):
    x = x / temperature
    x = x - np.max(x)            # numerical stability
    e = np.exp(x)
    return e / e.sum()

def show(probs):
    for word, p in sorted(zip(vocab, probs), key=lambda z: -z[1]):
        print(f"  {word:>9}: {p:6.1%} {'#' * int(p * 50)}")

# BRUTE FORCE: greedy
print("=== GREEDY: always pick the highest score ===")
print("  Picks:", vocab[int(np.argmax(logits))], "-> EVERY time. Repetitive.\n")

# SHORTCUT 1: TEMPERATURE
print("=== Temperature 0.5 ===");  show(softmax(logits, 0.5))
print("\n=== Temperature 1.0 ==="); show(softmax(logits, 1.0))
print("\n=== Temperature 2.0 ==="); show(softmax(logits, 2.0))

# SHORTCUT 2: TOP-K
def top_k(logits, k):
    idx = np.argsort(logits)[::-1][:k]
    masked = np.full_like(logits, -np.inf); masked[idx] = logits[idx]
    return softmax(masked)
print("\n=== Top-k (k=3) ==="); show(top_k(logits, 3))

# SHORTCUT 3: TOP-P / NUCLEUS
def top_p(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cutoff = np.searchsorted(np.cumsum(probs[order]), p) + 1
    keep = order[:cutoff]
    masked = np.full_like(logits, -np.inf); masked[keep] = logits[keep]
    return softmax(masked)
print("\n=== Top-p (p=0.9) ==="); show(top_p(logits, 0.9))
```

### The big picture first
When an LLM writes, it picks one word at a time by doing two jobs: **(1) give every candidate word a score**, then **(2) turn those scores into a final pick.** This lab lets you watch job #2. `temperature`, `top-k`, and `top-p` are just *different ways of doing job #2*.

### Step 1 — the candidates and their scores
```python
vocab  = ["sunny", "rainy", "cloudy", "hot", "cold", "banana", "purple", "running"]
logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, -1.0, -2.0, -3.0])
```
Pretend the model read **"The weather today is ___"** and is staring at 8 possible next words. `vocab` = those words. `logits` = the **raw scores** the model gave each — think of a **teacher handing out marks**:
```
sunny  3.0   ← liked most       banana  -1.0  ← disliked (negative!)
rainy  2.5                       purple  -2.0
cloudy 2.0                       running -3.0
hot    1.5
cold   1.0
```
Two things to burn in: **higher score = liked more**, and **logits are NOT probabilities yet** — they can be any number and don't add up to anything tidy.

### Step 2 — why we can't use logits directly
A probability must be **between 0 and 1**, and all of them must **add up to 1**. Our logits break both rules. So we need a machine that converts scores → probabilities. That machine is **softmax**.

### Step 3 — softmax, one line at a time
*(Ignore `temperature` for now — pretend it's 1.0, i.e. "divide by 1" = no change. It's the star, coming next.)*

**`x = x - np.max(x)`  — "shrink the numbers, keep the order."**
`np.max(x)` finds the biggest score (3.0). Subtract it from all:
```
3.0-3.0= 0.0    2.5-3.0=-0.5    2.0-3.0=-1.0   ...
```
Why? Pure safety — the next step (`e^x`) explodes on big numbers, so we keep everything ≤ 0. **Does it change who's winning?** No: `3>2.5>2` becomes `0>-0.5>-1` — same order, same gaps.

**`e = np.exp(x)`  — "stretch the gaps apart."**
`np.exp(x)` means **e^x** (e ≈ 2.718). Key fact: it turns any number positive and **stretches differences**. It grows fast: `e^0=1, e^1≈2.7, e^2≈7.4, e^3≈20`. Applied:
```
 0.0 → 1.000     -0.5 → 0.607     -1.0 → 0.368     -1.5 → 0.223
```
The favorite word now pulls clearly ahead, and every value is positive.

**`return e / e.sum()`  — "turn into percentages."**
`e.sum()` adds them all; dividing each by the total forces them to sum to 100%. For the full 8 words at **temperature 1.0**:
```
sunny    ≈ 42%   #####################
rainy    ≈ 26%   #############
cloudy   ≈ 16%   ########
hot      ≈ 9%    ####
cold     ≈ 6%    ###
banana   ≈ 0.8%
purple   ≈ 0.3%
running  ≈ 0.1%
```
Now they're real probabilities. **That's softmax: scores in → probabilities out.**

> 💡 Why "*soft*max"? A *hard* max gives the top word 100% and everyone else 0%. Softmax is the gentle version — top word gets the most, others still get a slice. (That "hard max" is basically **greedy**, below.)

### Step 4 — NOW temperature (the part everyone gets wrong)
```python
x = x / temperature
```
This runs **before** softmax, so temperature changes the **scores** — specifically the **size of the gaps** — before they become probabilities. Using simple scores `[3, 2, 1]`:

- **Temperature 1.0** → ÷1 → `[3,2,1]` → softmax ≈ **66% / 24% / 10%** (normal gap).
- **Temperature 0.5** → ÷0.5 (=×2) → `[6,4,2]` → gap **doubled** → softmax ≈ **87% / 12% / 2%**. Top word **dominates** → confident, safe, repetitive.
- **Temperature 2.0** → ÷2 → `[1.5,1,0.5]` → gap **shrank** → softmax ≈ **50% / 31% / 19%**. More equal → exploratory, creative, riskier.

On your lab's real 8 words, watch the gaps squeeze as temperature rises:

| Word | temp **0.5** | temp **1.0** | temp **2.0** |
|---|---|---|---|
| sunny | **64%** | 42% | **29%** |
| rainy | 23% | 26% | 22% |
| cloudy | 9% | 16% | 17% |
| hot | 3% | 9% | 14% |
| cold | 1% | 6% | 11% |
| banana | ~0% | ~1% | **4%** |
| purple | ~0% | ~0% | 2% |
| running | ~0% | ~0% | 1% |

As temperature **goes up**, "sunny" drops from **64% → 29%**, while silly "banana" climbs from ~0% → **4%**.

> 🚨 **Kill this myth:** temperature does **NOT** "add randomness" or "pick random words." It only **reshapes the probabilities**. The candidate words never change — only their *chances*.
>
> 🧒 **Teacher analogy:** original marks 98/95/90. Low temp = strict teacher *exaggerates* → 100/90/60 (top student almost always wins). High temp = lenient teacher *compresses* → 98/97/96 (anyone could win). Same students, different odds.

### Step 5 — Greedy (the brute-force baseline)
`np.argmax(logits)` returns the **position** of the highest score → always "sunny." No dice roll, identical output forever: *"The weather is sunny. The weather is sunny."* This is the "hard max"; the other knobs improve on it.

### Step 6 — Top-k (keep the K best, delete the rest)
```python
idx = np.argsort(logits)[::-1][:k]
masked = np.full_like(logits, -np.inf); masked[idx] = logits[idx]
return softmax(masked)
```
- `np.argsort(logits)` → positions that sort scores low→high; `[::-1]` reverses to high→low; `[:k]` keeps the top **k** (with `k=3`: sunny, rainy, cloudy).
- `np.full_like(logits, -np.inf)` → a fresh array filled with **−∞ (negative infinity)**.
- `masked[idx] = logits[idx]` → copy back real scores only for the top-k; everyone else stays −∞.
- `softmax(masked)` → since `e^(−∞) = 0`, the losers get **0% — impossible to pick.** Good: "banana"/"running" were nonsense, so banishing them is smart.

### Step 7 — Top-p / nucleus (the smart, adaptive one)
```python
probs = softmax(logits)
order = np.argsort(probs)[::-1]
cutoff = np.searchsorted(np.cumsum(probs[order]), p) + 1
keep = order[:cutoff]
```
- `probs = softmax(logits)` → get normal probabilities first.
- `order = ...[::-1]` → positions sorted high→low by probability.
- `np.cumsum(...)` → a **running total**: `sunny 42% → +rainy 68% → +cloudy 84% → +hot 93%` (just passed 90%).
- `np.searchsorted(running_total, p)` → finds **where p (0.9) fits** in that running total — between 84% and 93%, so include through "hot"; `+1` makes the count right.
- `keep = order[:cutoff]` → keep that group; mask the rest to −∞ and re-softmax.

🧒 **Why it's clever — it adapts:** if the model is **unsure** (spread-out probabilities), it takes *more* words to reach 90%, so top-p keeps more. If it's **very confident** (sunny 96%), 90% is reached by sunny alone, so top-p keeps just one. `k` is fixed; `p` adjusts itself. **That's why modern LLMs love top-p**, often paired with temperature (e.g. `temperature=0.7, top_p=0.95` = "coherent, naturally varied, ignore the truly unlikely junk").

### Step 8 — the `show` helper (just drawing)
`zip(vocab, probs)` pairs each word with its probability; `sorted(..., key=lambda z: -z[1])` sorts biggest-first; the f-string prints the word, the percentage, and a **bar of `#`** as long as `probability × 50` (42% → 21 hashes). It's how you *see* the distribution.

**Run it:** `python 03_sampling_from_scratch.py`
**🔧 Break it yourself:** temperature `0.01` (you've rebuilt greedy); temperature `100` (near-equal chaos); `top_p` `0.5` (how many words survive?); `logits = np.zeros(8)` (what does each method do when the model has no preference?).

### 🧠 The one mental model
> Picture **8 runners** in a race. **Logits** = points each earned. **Softmax** = converts points into each runner's *chance of winning*. **Temperature** = how far apart they look: low (0.2–0.5) stretches the gap (fastest almost always wins); normal (≈1) keeps it; high (1.5–2+) squishes it (slow runners get a real shot). **Top-k** = only the top K race. **Top-p** = only the fewest runners who together cover 90% race. The runners (words) never change — these knobs only change *who's likely to win*.

---

# PART D — THE REST OF THE MENTAL MODEL

## How a model is built: Pre-training → Fine-tuning → RLHF → DPO
1. **Pre-training — "read the whole library."** 🧒 A baby reads millions of books, learning language by the next-word game. Hugely expensive (months, thousands of GPUs). Result: knows language + facts, but **not helpful yet** — ask a question and it might just continue with *more* questions (that's what it saw in text).
2. **Fine-tuning / SFT — "a focused class."** 🧒 Show it thousands of clean *"instruction → good response"* examples. It learns the *behavior* of a helpful assistant. (Also how you'd specialize it for medical/legal text — Modules 19–20.)
3. **RLHF (Reinforcement Learning from Human Feedback) — "manners via feedback."** 🧒 Like training a puppy with treats: show two answers, a human picks the better, the model learns to prefer what humans like (helpful, polite, safe). *Brute-force detail:* classic RLHF trains a separate "reward model," then optimizes against it — powerful but fiddly.
4. **DPO (Direct Preference Optimization) — the shortcut to RLHF.** 🧒 Same goal, but skips the separate reward model and optimizes directly on the "this answer beats that one" pairs. Reaches similar quality with far less complexity → cleaner and more stable.

> **Punchline:** pre-training gives *knowledge*; fine-tuning + RLHF/DPO give *behavior*. A model with only step 1 is a library with no librarian.

## Context window — the model's "desk space"
🧒 You do homework on a small desk; only so many papers fit, and a new one pushes an old one off. The **context window** is that desk — the max number of **tokens** the model sees at once (your prompt + its answer + history, all counted).
- Measured in **tokens** (Part A — that's why tokens matter).
- Exceed it → the **oldest part is forgotten**. *This* is why long chats "lose track."
- Also why you can't paste a 500-page book and ask questions — it won't fit.

**The fix is RAG (Module 9):** keep a filing cabinet (database) and fetch only the few relevant pages right before answering. You now understand RAG's *purpose* before we build it. 🎯

## Model sizes — "7B / 70B / 400B"
🧒 A model is a giant mixing board with billions of knobs called **parameters**, set during training. More knobs = more capacity, but heavier/slower. "7B" = 7 **billion** parameters.

| Size | Roughly needs | Runs on | Vibe |
|---|---|---|---|
| **1–3B** (Phi, Gemma-2B) | ~2–4 GB RAM | Laptop, even phones | Fast, basic |
| **7–8B** (Llama-3-8B, Mistral-7B) | ~6–8 GB (quantized) | Good laptop / one GPU | The "runs locally" sweet spot |
| **70B** (Llama-3-70B) | ~40 GB+ | Serious GPU / cloud | Strong but heavy |
| **200B–1T+** (frontier) | Data center | Cloud only (API) | Smartest, priciest |

**Quantization** (Module 20) shrinks a model (store knobs with less precision) so a 7B fits on a laptop with little quality loss — *brute force = full precision on a huge GPU; shortcut = quantize and run on your laptop.*

## Open vs Closed models
| | **Closed / Proprietary** | **Open weights** |
|---|---|---|
| Examples (change often!) | GPT-family, Claude-family, Gemini-family | Llama, Mistral, Qwen, Phi, Gemma, DeepSeek |
| Access | Paid API only | Download & run yourself (often free) |
| Privacy | Data leaves your machine | Can run fully offline/private |
| Capability | Usually strongest | Catching up fast |
| Cost model | Pay per token | Pay for your own hardware |
| Customization | Limited | Full (fine-tune freely) |

> ⚠️ The specific **names and rankings change every few months** — treat categories as stable, names as a snapshot.

## Hallucination — why LLMs confidently make things up
🧒 A kid who *hates* saying "I don't know," so gives a confident, smooth, made-up answer.
**Why (the key insight of the module):** an LLM is a **next-word predictor**, not a fact database. Its job is *plausible, fluent* text. It has **no built-in truth-check** and (by default) **no live access to reality** — a fake citation and a real one look equally "likely" as text.
**Reduce it:** **RAG** (Module 9 — give it real sources; biggest lever), **lower temperature** for facts, **prompt for citations / "say I don't know"** (Module 6), **verify outputs + guardrails** (Module 24), **tool use / agents** (Modules 13–16 — let it call a calculator/search/database instead of guessing).
> Hallucination isn't a bug they'll patch away — it's a consequence of *how* the model works. Engineering around it is most of what this course secretly teaches.

---

## 💻 LAB 4 — Run a real (tiny) LLM on your own computer

Proves an LLM is just math running locally. We use **GPT-2** — ancient/tiny, but runs on a laptop CPU. Install once: `pip install transformers torch`.
> ⚠️ `torch` is a big download (~200MB+); GPT-2 downloads ~500MB on first run (cached after). If torch is painful on Windows, paste this file into **Google Colab** (free, nothing to install).

Save as `04_tiny_local_model.py`:

```python
"""
Module 2 - Lab 4: Run a REAL (tiny) LLM on YOUR machine. No cloud, no API key.
"""
from transformers import pipeline, set_seed

print("Loading GPT-2... (first run downloads ~500MB, then cached)")
generator = pipeline("text-generation", model="gpt2")
prompt = "The weather today is"
pad_id = generator.tokenizer.eos_token_id

print("\n=== GREEDY ===")
print(generator(prompt, max_new_tokens=30, do_sample=False,
                truncation=True, pad_token_id=pad_id)[0]["generated_text"])

print("\n=== TEMPERATURE 0.4 (focused) ===")
set_seed(42)
print(generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.4,
                top_k=50, truncation=True, pad_token_id=pad_id)[0]["generated_text"])

print("\n=== TEMPERATURE 1.3 (creative) ===")
set_seed(42)
print(generator(prompt, max_new_tokens=30, do_sample=True, temperature=1.3,
                top_k=50, truncation=True, pad_token_id=pad_id)[0]["generated_text"])
```

### Line by line, in plain English
- `from transformers import pipeline, set_seed` → `pipeline` is an easy wrapper that loads and runs a model for you; `set_seed` fixes the dice so sampling is reproducible.
- `generator = pipeline("text-generation", model="gpt2")` → download (first time) and load **GPT-2**, ready to generate text.
- `prompt = "The weather today is"` → the starting text the model will continue.
- `pad_id = generator.tokenizer.eos_token_id` → a small technical detail that just silences a harmless warning (it tells the model what "end of text" looks like).
- `do_sample=False` → **greedy** (always take the top token; from Lab 3).
- `do_sample=True, temperature=..., top_k=50` → turn on the **dice-roll** and use the exact knobs you learned in Lab 3 — now on a real model.
- `max_new_tokens=30` → generate up to 30 new tokens.
- `[0]["generated_text"]` → the result is a list of dictionaries; this digs out the actual text string.

**Run it:** `python 04_tiny_local_model.py`
**Observe:** after download, a real language model generates text *on your machine* — same family of math as GPT-4, just far smaller and dumber. GPT-2 rambles — **that's the point**; you *feel* the difference between 124M params and the giants. Greedy repeats; high temperature wanders.
**🔧 Break it yourself:** try `model="distilgpt2"` (smaller/dumber); prompt `"def add(a, b):"` and watch it attempt code; crank `temperature=2.0` for near-gibberish.

---

## 🔧 Setup recap (everything Module 2 needs)
```bash
pip install tiktoken numpy transformers torch
```
`tiktoken` → Lab 2 · `numpy` → Lab 3 · `transformers torch` → Lab 4 (or run Lab 4 in Colab).

## 🛠️ Commit to GitHub
```bash
# from your repo root; save the 4 .py files into module-02-how-llms-work/ first
git add .
git commit -m "Module 2: tokenization, attention, sampling labs + local GPT-2 demo"
git push
```

---

## ⚠️ Common Mistakes
1. **Thinking the model "knows" or "looks things up."** It predicts plausible text — internalize this and hallucination stops surprising you.
2. **Confusing characters/words with tokens.** You're billed and limited by *tokens* (~¾ word).
3. **High temperature for factual/coding tasks** → you're *inviting* mistakes. Facts → low temp.
4. **Pasting huge docs and expecting "memory."** That's a context-window violation — the job for RAG.
5. **Assuming bigger is always better.** Often a small fast model + good prompting + RAG beats a giant used naively, at a fraction of the cost.

## 💼 Real Production Example
Coding assistants (Copilot, Cursor) do Part C live: tokenize your code, attend to relevant earlier lines, autoregressively predict tokens — at **low temperature**, because code must be precise. A support bot answering from help docs is **RAG** (Module 9) bolted onto this same loop to fight hallucination.

## 📋 Decision Table — which knob when?
| You want… | Set this |
|---|---|
| Reliable facts, extraction, code | temperature **0–0.3** |
| A balance | temperature **~0.7** |
| Brainstorming, variety | temperature **0.9–1.2** |
| Same answer every time (testing) | temperature **0** / greedy |
| Block absurd words | top-k or top-p |
| Smart, adaptive filtering | top-p **~0.9** (most common) |

## 💬 Plain English Summary
An LLM is a giant next-word predictor. Text is chopped into **tokens** (subword chunks — smarter than words or letters), turned into meaning-vectors, processed by a **Transformer** whose **attention** decides which earlier words matter. It writes one token at a time in a loop; **temperature/top-k/top-p** control how adventurous each pick is. It's built by **pre-training** (knowledge) then **fine-tuning + RLHF/DPO** (behavior). It sees a limited **context window** (in tokens) — which is *why* **RAG** exists. **Bigger models** are smarter but heavier. It **hallucinates** because it predicts plausible text, not verified facts.

## 👨‍🏫 Interview Tips
- **What is a token?** The unit an LLM reads; modern LLMs use subword tokenization (BPE) — common words are single tokens, rare/new words break into known pieces. Middle ground between word-level and char-level. Cost/limits measured in tokens.
- **Temperature, top-k, top-p?** All control next-token choice. Temperature reshapes the distribution (low=sharp/safe, high=flat/creative); top-k keeps the k highest; top-p (nucleus) keeps the smallest set summing to p (adapts to confidence). Top-p is the common default.
- **Why hallucinate, how reduce?** Trained for *plausible* tokens, not truth — no built-in fact-checker. Reduce via RAG, lower temperature, citation prompting, tool use, output validation.
- **Pre-training vs fine-tuning?** Pre-training = expensive base stage (language + world knowledge via next-word prediction). Fine-tuning (+RLHF/DPO) = smaller later stage shaping behavior.
- **Context window?** Max tokens attended to at once (prompt + history + output); exceed it and the earliest content is dropped. Core motivation for RAG.
- **Decoder-only vs encoder-only?** Decoder-only (GPT/Llama/Claude/Gemini) generate left→right (chat/agents); encoder-only (BERT) understand/encode for classification & embeddings; encoder-decoder (T5) transform input→output.

## 🔥 Cross-Questions (test yourself)
1. Your chatbot "forgets" what the user said 30 messages ago. Using only this module, explain why — and name the idea that points to the fix.
2. A teammate says "set temperature to 1.5 so answers are more accurate." What's wrong with that?
3. Why does `"strawberry"` (Lab 2) explain why LLMs struggle to count its "r"s?
4. You need a private, offline assistant on a laptop with no GPU. Which model size, and what trade-off are you accepting?

## ✅ Quick Check
1. In one sentence, what game is an LLM really playing?
2. Why is subword tokenization better than both word-level and character-level?
3. What does attention let the model do? Give the "it" example.
4. Order these and say what each adds: pre-training, RLHF, fine-tuning.
5. Roughly how many words is 1,000 tokens?

## 📚 Best Free Resources
- **Andrej Karpathy — "Let's build GPT from scratch" (YouTube):** builds a tiny GPT in code; best follow-up to this module.
- **Andrej Karpathy — "Intro to Large Language Models" (1 hr):** cleanest big-picture talk.
- **3Blue1Brown — Neural Networks / Transformers series:** the most beautiful *visual* explanation of attention.
- **Jay Alammar — "The Illustrated Transformer" (blog):** classic picture-based walkthrough.
- **Hugging Face LLM Course + Tokenizers docs:** hands-on, matches our tooling.
- **OpenAI Tokenizer (web) + `tiktoken` GitHub README:** play with tokenization in-browser.

---

*End of Module 2. Next up (when you're ready): Module 3 — Free LLM APIs (Groq, Gemini, Ollama), where you build your first real project: a chatbot with memory.*
