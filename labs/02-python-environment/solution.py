"""
solution.py - Lab 2 reference solution.

Attempt starter.py first. Then read this for the reasoning, which is explained
in comments and more fully in SOLUTION.md.

    python solution.py
"""


def build_prompt(topic: str, level: str = "beginner", max_words: int = 100) -> str:
    """Build an explanation prompt for a topic.

    Args:
        topic:     What to explain, e.g. "embeddings".
        level:     Audience level. Defaults to "beginner".
        max_words: Word limit for the answer. Defaults to 100.

    Returns:
        A formatted prompt string.
    """
    # An f-string interpolates each variable at the point it appears.
    # Note the parameter ORDER: topic, level, max_words - but the string uses
    # them in a different order. Parameter order and usage order are unrelated.
    return f"Explain {topic} in under {max_words} words. Audience level: {level}."


def estimate_tokens(text: str) -> int:
    """Roughly estimate how many tokens a piece of text uses.

    Uses the ~4-characters-per-token rule of thumb. Module 3 replaces this
    with tiktoken, which counts exactly.

    Args:
        text: The text to measure.

    Returns:
        Estimated token count; 0 for empty text, otherwise at least 1.
    """
    # Handle the empty case first. `not text` is True for "" and for None,
    # which makes this defensive as well as concise.
    if not text:
        return 0

    # round() gives the nearest whole number; max(1, ...) enforces the floor so
    # a 1-character string reports 1 token rather than 0.
    return max(1, round(len(text) / 4))


def estimate_cost(token_count: int, price_per_million: float = 0.15) -> float:
    """Estimate what a number of tokens costs in US dollars.

    Args:
        token_count:       How many tokens.
        price_per_million: Dollars per 1,000,000 tokens.

    Returns:
        Estimated cost in dollars.
    """
    # Convert to millions, then scale by the price.
    # The 1_000_000 underscores are ignored by Python and purely for legibility.
    return (token_count / 1_000_000) * price_per_million


def build_messages(system_prompt: str, user_prompt: str) -> list:
    """Build the messages list that every chat API expects.

    Args:
        system_prompt: Standing instructions that set the model's behaviour.
        user_prompt:   The actual request.

    Returns:
        A two-element list of role/content dictionaries.
    """
    # A LIST (order = chronological) of DICTIONARIES (each a labelled turn).
    # System goes first by convention: it frames everything that follows.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def summarise_batch(prompts: list) -> dict:
    """Summarise a batch of prompts before sending them anywhere.

    Args:
        prompts: A list of prompt strings.

    Returns:
        A dict with keys "count", "total_tokens" and "longest".
    """
    # Guard the empty case up front. Without this, max() below raises
    # ValueError on an empty sequence - a classic edge-case bug.
    if not prompts:
        return {"count": 0, "total_tokens": 0, "longest": None}

    # Accumulate tokens by reusing estimate_tokens rather than reimplementing
    # the rule. If the rule ever changes, it changes in exactly one place.
    total_tokens = 0
    for prompt in prompts:
        total_tokens += estimate_tokens(prompt)

    # max() with key=len compares items by length instead of alphabetically.
    longest = max(prompts, key=len)

    return {
        "count": len(prompts),
        "total_tokens": total_tokens,
        "longest": longest,
    }


# ======================================================================
# Demonstration - everything working together
# ======================================================================

def main():
    print()
    print("=" * 68)
    print("  PROMPT TOOLKIT - demonstration")
    print("=" * 68)

    # --- Build a batch of prompts with a loop (Module 2, section 2.7) ---
    topics = ["embeddings", "attention", "retrieval-augmented generation"]

    prompts = []
    for topic in topics:
        prompts.append(build_prompt(topic, level="beginner", max_words=80))

    print("\nGenerated prompts:")
    # enumerate(..., start=1) gives human-friendly 1-based numbering.
    for index, prompt in enumerate(prompts, start=1):
        print(f"  {index}. {prompt}")

    # --- Summarise the batch ---
    summary = summarise_batch(prompts)

    print("\nBatch summary:")
    for key, value in summary.items():
        print(f"  {key:>13}: {value}")

    # --- Estimate the cost ---
    cost = estimate_cost(summary["total_tokens"])
    print(f"\nEstimated input cost: ${cost:.6f}")
    print("  (input tokens only, at an illustrative $0.15 per million -")
    print("   check your provider's current pricing)")

    # --- Show the messages shape that every API call needs ---
    messages = build_messages(
        system_prompt="You are a concise tutor. Answer in 2 sentences.",
        user_prompt=prompts[0],
    )

    print("\nAPI-ready messages list:")
    for message in messages:
        # Truncate long content so the shape stays readable.
        content = message["content"]
        preview = content if len(content) <= 55 else content[:52] + "..."
        print(f"  {message['role']:>9}: {preview}")

    print()
    print("This 'messages' list is exactly what you pass to the API in Part C.")
    print()


if __name__ == "__main__":
    main()
