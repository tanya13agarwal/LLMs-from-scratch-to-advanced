"""
Project 4: The Model Router.
Sends each question to the best-suited FREE Groq model, with task-appropriate
settings. Two classifiers included: brute-force keywords + smarter LLM.
Run:  python router.py
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Two FREE Groq models with a cost/quality tradeoff
FAST_MODEL  = "llama-3.1-8b-instant"      # cheap & quick - simple stuff + classifying
SMART_MODEL = "llama-3.3-70b-versatile"   # stronger - coding & creative

VALID_LABELS = {"coding", "creative", "general"}

# One place that defines model + system prompt + temperature per category.
# Adding a new category later = add ONE entry here (no new if/elif anywhere).
CONFIG = {
    "coding":   {"model": SMART_MODEL, "system": "You are an expert programmer.",  "temperature": 0.2},
    "creative": {"model": SMART_MODEL, "system": "You are a creative writer.",     "temperature": 0.9},
    "general":  {"model": FAST_MODEL,  "system": "You are a helpful assistant.",   "temperature": 0.7},
}

# ---------- Classifier A: brute-force keywords (free, instant, but brittle) ----------
CODE_KEYWORDS     = ["code", "function", "bug", "python", "error", "script"]
CREATIVE_KEYWORDS = ["poem", "story", "song", "write a", "imagine"]

def classify(question):
    """Label by keyword matching. Fast and free, but only knows exact words."""
    q = question.lower()
    if any(word in q for word in CODE_KEYWORDS):
        return "coding"
    elif any(word in q for word in CREATIVE_KEYWORDS):
        return "creative"
    else:
        return "general"

# ---------- Classifier B: LLM (understands meaning; costs one fast call) ----------
def classify_with_llm(question):
    """Label by asking a fast LLM. Understands paraphrases keywords miss."""
    system = ("You are a classifier. Reply with EXACTLY ONE word - "
              "'coding', 'creative', or 'general' - and nothing else.")
    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        temperature=0,                      # precise, repeatable label - not creative
    )
    label = resp.choices[0].message.content.strip().lower()

    # 🛡️ Only trust known labels; otherwise fall back loudly (never silently misroute)
    if label not in VALID_LABELS:
        print(f"⚠️ Unexpected label '{label}' - defaulting to 'general'")
        label = "general"
    return label

# ---------- Router: look up the config for the category, call Groq ----------
def route(question, category):
    """Pick model/system/temperature via CONFIG, call Groq, return the answer text."""
    cfg = CONFIG.get(category, CONFIG["general"])   # safe default if category is odd
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system"]},
            {"role": "user",   "content": question},
        ],
        temperature=cfg["temperature"],
    )
    return resp.choices[0].message.content

# ---------- Try it ----------
if __name__ == "__main__":
    questions = [
        "Write a Python function to reverse a string.",
        "Write a short poem about the moon.",
        "What is the capital of France?",
    ]
    for q in questions:
        category = classify_with_llm(q)          # swap to classify(q) to compare
        print(f"\n❓ {q}")
        print(f"🧭 Routed to: {category}")
        print(f"🤖 {route(q, category)}")

    # See the two classifiers disagree on tricky inputs
    print("\n" + "=" * 50 + "\nKEYWORD vs LLM on tricky questions:")
    tricky = [
        "Compose a haiku about autumn leaves.",
        "Help me fix this broken loop.",
        "What's the tallest mountain on Earth?",
    ]
    for q in tricky:
        print(f"\n❓ {q}")
        print(f"   keyword : {classify(q)}")
        print(f"   LLM     : {classify_with_llm(q)}")