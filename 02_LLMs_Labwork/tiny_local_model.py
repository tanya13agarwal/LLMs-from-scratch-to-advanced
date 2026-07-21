"""
Module 2 - Lab 4: Run a REAL (tiny) LLM on YOUR machine. No cloud, no API key.
Install:  pip install transformers torch
(First run downloads GPT-2 ~500MB. If torch is painful, run in Colab.)
"""
from transformers import pipeline, set_seed

print("Loading GPT-2... (first run downloads ~500MB, then cached)")
generator = pipeline("text-generation", model="gpt2")
prompt = "The weather today is"
pad_id = generator.tokenizer.eos_token_id   # silences a harmless warning

# GREEDY (deterministic, often boring/repetitive)
print("\n=== GREEDY ===")
print(generator(prompt, max_new_tokens=30, do_sample=False,
                truncation=True, pad_token_id=pad_id)[0]["generated_text"])

# LOW TEMPERATURE (focused, sensible)
print("\n=== TEMPERATURE 0.4 (focused) ===")
set_seed(42)
print(generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.4,
                top_k=50, truncation=True, pad_token_id=pad_id)[0]["generated_text"])

# HIGH TEMPERATURE (creative, sometimes weird)
print("\n=== TEMPERATURE 1.3 (creative/chaotic) ===")
set_seed(42)
print(generator(prompt, max_new_tokens=30, do_sample=True, temperature=1.3,
                top_k=50, truncation=True, pad_token_id=pad_id)[0]["generated_text"])

print("\nRun a few times. Greedy never changes. Low temp stays sensible. High temp wanders.")