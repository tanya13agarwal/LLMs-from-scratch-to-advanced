"""
Module 2 - Lab 3: How an LLM PICKS the next word (BRUTE FORCE -> SHORTCUTS).
Tiny fake vocabulary so you SEE the math behind temperature/top-k/top-p.
Install:  pip install numpy
"""
import numpy as np

# Pretend the model just read: "The weather today is ___"
# These 8 words are candidates, with raw scores ("logits"). Higher = liked more.
vocab  = ["sunny", "rainy", "cloudy", "hot", "cold", "banana", "purple", "running"]
logits = np.array([3.0,    2.5,     2.0,     1.5,   1.0,    -1.0,    -2.0,      -3.0])

def softmax(x, temperature=1.0):
    """Turn raw scores into probabilities that sum to 1.0."""
    x = x / temperature
    x = x - np.max(x)            # numerical stability
    e = np.exp(x)
    return e / e.sum()

def show(probs):
    for word, p in sorted(zip(vocab, probs), key=lambda z: -z[1]):
        print(f"  {word:>9}: {p:6.1%} {'#' * int(p * 50)}")

# ----- BRUTE FORCE: greedy -----
print("=== GREEDY: always pick the highest score ===")
print("  Picks:", vocab[int(np.argmax(logits))], "-> EVERY time. Never varies. Repetitive.\n")

# ----- SHORTCUT 1: TEMPERATURE -----
print("=== Temperature 0.5 (cautious/sharper) ===");  show(softmax(logits, 0.5))
print("\n=== Temperature 1.0 (normal) ===");           show(softmax(logits, 1.0))
print("\n=== Temperature 2.0 (wild/flatter) ===");     show(softmax(logits, 2.0))
print("  Low temp -> safe & predictable. High temp -> creative & risky.\n")

# ----- SHORTCUT 2: TOP-K -----
def top_k(logits, k):
    idx = np.argsort(logits)[::-1][:k]
    masked = np.full_like(logits, -np.inf); masked[idx] = logits[idx]
    return softmax(masked)
print("=== Top-k with k=3 (ignore all but top 3) ===");  show(top_k(logits, 3))
print("  'banana','purple','running' now IMPOSSIBLE - good, they were nonsense.\n")

# ----- SHORTCUT 3: TOP-P / NUCLEUS -----
def top_p(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cutoff = np.searchsorted(np.cumsum(probs[order]), p) + 1
    keep = order[:cutoff]
    masked = np.full_like(logits, -np.inf); masked[keep] = logits[keep]
    return softmax(masked)
print("=== Top-p with p=0.9 (keep words until 90% covered) ===");  show(top_p(logits, 0.9))
print("  Keeps MORE words when unsure, FEWER when confident -> why top-p is popular.")