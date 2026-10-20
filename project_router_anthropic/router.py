"""
Question Router - Routes questions to appropriate AI responses based on category.

This router classifies incoming questions into three categories:
- coding: Programming-related questions (uses low temperature for precision)
- creative: Creative writing requests (uses high temperature for creativity)  
- general: General knowledge questions (uses medium temperature)

Two classification methods are provided:
1. Keyword-based (fast, no API call)
2. LLM-based (smarter, uses API call)
"""

import os
from groq import Groq
from dotenv import load_dotenv

# ============================================================================
# SETUP
# ============================================================================

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Two models with cost/quality tradeoff:
FAST_MODEL  = "openai/gpt-oss-20b"   # smaller & faster - for simple stuff
SMART_MODEL = "openai/gpt-oss-120b"  # larger & smarter - for coding & creative


# ============================================================================
# CLASSIFICATION METHODS
# ============================================================================

def classify(question):
    """
    Classify question using keyword matching.
    
    Fast and free (no API call), but less accurate for edge cases.
    
    Args:
        question: The user's question string
        
    Returns:
        One of: 'coding', 'creative', 'general'
    """
    question = question.lower()
    
    coding_keywords = [
        "code", "function", "bug", "python", "error", "script", 
        "programming", "debug", "variable", "class", "method",
        "algorithm", "syntax", "compile", "runtime", "javascript",
        "java", "c++", "html", "css", "api", "database", "sql"
    ]
    
    creative_keywords = [
        "poem", "story", "song", "write a", "imagine", "creative",
        "fiction", "narrative", "compose", "invent", "fantasy",
        "tale", "lyrics", "poetry", "novel", "dream"
    ]
    
    if any(keyword in question for keyword in coding_keywords):
        return "coding"
    elif any(keyword in question for keyword in creative_keywords):
        return "creative"
    else:
        return "general"


def classify_with_llm(question):
    """
    Classify question using LLM.
    
    More accurate than keyword matching, but costs an API call.
    Uses temperature=0 for consistent, deterministic results.
    
    Args:
        question: The user's question string
        
    Returns:
        One of: 'coding', 'creative', 'general'
    """
    # System prompt instructs LLM to act as a classifier
    system = (
        "You are a classifier. Read the user's question and reply with "
        "EXACTLY ONE word - either 'coding', 'creative', or 'general' - "
        "and nothing else."
    )

    # Call Groq API with temperature=0 for precise, repeatable classification
    # Use FAST_MODEL for classification (quick & cheap)
    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    # Extract and clean up the label
    label = resp.choices[0].message.content.strip().lower()
    
    # 🛡️ Bulletproofing: accept ONLY known labels; otherwise fall back loudly
    valid = {"coding", "creative", "general"}
    if label not in valid:
        print(f"⚠️ Unexpected label '{label}' - defaulting to 'general'")
        label = "general"
    
    return label


# ============================================================================
# ROUTING
# ============================================================================

def route(question, category):
    """
    Route question to appropriate AI response based on category.
    
    Each category uses different system prompts and temperatures:
    - coding:   Expert programmer persona, low temp (0.2) for precision
    - creative: Creative writer persona, high temp (0.9) for creativity
    - general:  Helpful assistant persona, medium temp (0.7) for balance
    
    Args:
        question: The user's question string
        category: One of 'coding', 'creative', 'general'
        
    Returns:
        AI-generated response string
    """
    # Configure based on category
    if category == "coding":
        model = SMART_MODEL  # Use smart model for coding
        system = "You are an expert programmer."
        temperature = 0.2
    elif category == "creative":
        model = SMART_MODEL  # Use smart model for creative
        system = "You are a creative writer."
        temperature = 0.9
    else:  # general
        model = FAST_MODEL   # Use fast model for general questions
        system = "You are a helpful assistant."
        temperature = 0.7
    
    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test questions covering all three categories
    questions = [
        "Write a Python function to reverse a string.",  # coding
        "Write a short poem about the moon.",            # creative
        "What is the capital of France?",                # general
    ]

    # Test 1: Keyword-based classification
    print("=" * 60)
    print("TEST 1: Keyword-based Classification")
    print("=" * 60)
    
    for q in questions:
        category = classify(q)
        print(f"\n❓ {q}")
        print(f"🧭 Routed to: {category}")
        answer = route(q, category)
        print(f"🤖 {answer}")

    # Test 2: LLM-based classification
    print("\n" + "=" * 60)
    print("TEST 2: LLM-based Classification")
    print("=" * 60)
    
    for q in questions:
        category = classify_with_llm(q)
        print(f"\n❓ {q}")
        print(f"🧭 Routed to: {category}")
        answer = route(q, category)
        print(f"🤖 {answer}")

    # Test 3: Tricky questions - comparing both classifiers
    # These questions expose limitations of keyword matching
    print("\n" + "=" * 60)
    print("TEST 3: Tricky Questions - Keyword vs LLM Comparison")
    print("=" * 60)
    
    tricky = [
        "Compose a haiku about autumn leaves.",   # keywords MISS this -> wrongly 'general'
        "Help me fix this broken loop.",          # 'fix'/'loop' aren't in keyword lists
        "What's the tallest mountain on Earth?",  # genuinely general
    ]

    for q in tricky:
        print(f"\n❓ {q}")
        print(f"   keyword classifier : {classify(q)}")
        print(f"   LLM classifier     : {classify_with_llm(q)}")
