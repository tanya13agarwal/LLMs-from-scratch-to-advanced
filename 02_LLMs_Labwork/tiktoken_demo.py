"""
Module 2 - Lab 2: The REAL way - subword tokenization with tiktoken.
tiktoken is the EXACT tokenizer OpenAI's GPT models use.
Install:  pip install tiktoken
"""
import tiktoken

# cl100k_base = tokenizer for GPT-3.5 & GPT-4 ; o200k_base = for GPT-4o
enc = tiktoken.get_encoding("cl100k_base")

text = '''
for tid in token_ids:
    print(f"  {tid:>7}  ->  {enc.decode([tid])!r}")
'''

token_ids = enc.encode(text)                 # text -> numbers the model sees
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))

print("\nWhat each token really is:")
for tid in token_ids:
    print(f"  {tid:>7}  ->  {enc.decode([tid])!r}")

print("\nDecoded back to text:", enc.decode(token_ids))   # perfectly reversible

print("\nCharacters in text:", len(text))
print("Tokens in text:    ", len(token_ids))   # sits BETWEEN words and characters

enc_4o = tiktoken.get_encoding("o200k_base")
print("\nSame sentence, different tokenizers:")
print("  GPT-4 (cl100k_base):", len(enc.encode(text)), "tokens")
print("  GPT-4o (o200k_base):", len(enc_4o.encode(text)), "tokens")