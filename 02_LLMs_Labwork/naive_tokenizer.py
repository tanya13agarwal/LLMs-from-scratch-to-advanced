"""
Module 2 - Lab 1: The NAIVE ways to turn text into tokens (BRUTE FORCE).
Try the obvious approaches, watch them break. Lab 2 shows the real tool.
No installation needed - pure Python.
"""

text = "I'm learning GenAI, and it's amazing!! 🚀"

# ATTEMPT 1: split by spaces
print("=== Attempt 1: Split by spaces ===")
tokens_by_space = text.split(" ")
print(tokens_by_space)
print("Number of tokens:", len(tokens_by_space))
# Problems: "amazing!!" glues punctuation on; "I'm"/"it's" are blobs; emoji stuck on.

# ATTEMPT 2: split by every character
print("\n=== Attempt 2: Split by characters ===")
tokens_by_char = list(text)
print(tokens_by_char)
print("Number of tokens:", len(tokens_by_char))
# Problems: WAY too many tokens; "learning" alone is 8; each char means almost nothing.

# ATTEMPT 3: words AND punctuation kept separate (a bit smarter)
print("\n=== Attempt 3: Crude word + punctuation split (regex) ===")
import re
tokens_regex = re.findall(r"\w+|[^\w\s]", text)
print(tokens_regex)
print("Number of tokens:", len(tokens_regex))
# Better, BUT a brand-new word "GenAI" or typo "amazzzing" needs its own dictionary
# entry, or becomes [UNKNOWN] and loses all meaning.