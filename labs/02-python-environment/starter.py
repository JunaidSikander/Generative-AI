"""
starter.py - Lab 2: Build Your Workbench

Your task: replace each `TODO` with working code. A self-test at the bottom
checks your work and tells you exactly what is still failing.

    python starter.py

Every function maps to a section of Module 2, noted in its docstring. If you get
stuck, reread that section before opening SOLUTION.md.
"""


# ======================================================================
# TASK 1 - build_prompt
# Module 2, sections 2.5 (f-strings) and 2.7 (functions, default values)
# ======================================================================

def build_prompt(topic: str, level: str = "beginner", max_words: int = 100) -> str:
    """Build an explanation prompt for a topic.

    Args:
        topic:     What to explain, e.g. "embeddings".
        level:     Audience level. Defaults to "beginner".
        max_words: Word limit for the answer. Defaults to 100.

    Returns:
        A prompt string in exactly this shape:
        "Explain {topic} in under {max_words} words. Audience level: {level}."

    Examples:
        >>> build_prompt("embeddings")
        'Explain embeddings in under 100 words. Audience level: beginner.'
        >>> build_prompt("attention", "expert", 50)
        'Explain attention in under 50 words. Audience level: expert.'
    """
    # TODO: return an f-string matching the shape above.
    # Hint: f"Explain {topic} in under ..."
    return ""


# ======================================================================
# TASK 2 - estimate_tokens
# Module 2, sections 2.5 (len) and 2.7 (conditionals)
# ======================================================================

def estimate_tokens(text: str) -> int:
    """Roughly estimate how many tokens a piece of text uses.

    A common rule of thumb is that 1 token is about 4 characters of English.
    That is only an approximation - Module 3 measures it exactly with tiktoken -
    but it is good enough to sanity-check costs before sending a request.

    Args:
        text: The text to measure.

    Returns:
        Estimated token count. Empty text is 0. Any non-empty text is at
        least 1, because even a single character costs one token.

    Examples:
        >>> estimate_tokens("")
        0
        >>> estimate_tokens("abcd")
        1
        >>> estimate_tokens("a" * 40)
        10
    """
    # TODO:
    #   1. If text is empty, return 0.
    #   2. Otherwise divide its length by 4, round to the nearest whole
    #      number, and never return less than 1.
    # Hint: round() rounds, max() enforces a floor.
    return 0


# ======================================================================
# TASK 3 - estimate_cost
# Module 2, section 2.5 (floats)
# ======================================================================

def estimate_cost(token_count: int, price_per_million: float = 0.15) -> float:
    """Estimate what a number of tokens costs in US dollars.

    Providers quote prices per million tokens. The default here is
    illustrative only - always check your provider's current pricing page,
    and note that input and output tokens are usually priced differently.

    Args:
        token_count:       How many tokens.
        price_per_million: Dollars per 1,000,000 tokens.

    Returns:
        Estimated cost in dollars.

    Examples:
        >>> estimate_cost(1_000_000, 0.15)
        0.15
        >>> estimate_cost(1000, 0.15)
        0.00015
    """
    # TODO: convert token_count to millions, then multiply by price_per_million.
    return 0.0


# ======================================================================
# TASK 4 - build_messages
# Module 2, section 2.6 (lists of dictionaries) - THE most important shape
# ======================================================================

def build_messages(system_prompt: str, user_prompt: str) -> list:
    """Build the messages list that every chat API expects.

    Args:
        system_prompt: Standing instructions that set the model's behaviour.
        user_prompt:   The actual request.

    Returns:
        A list of two dictionaries, in this order:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]

    Examples:
        >>> build_messages("Be brief.", "Hi")
        [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}]
    """
    # TODO: return a list containing two dictionaries, each with the
    #       keys "role" and "content".
    return []


# ======================================================================
# TASK 5 - summarise_batch
# Module 2, sections 2.6 (lists, dicts) and 2.7 (loops)
# ======================================================================

def summarise_batch(prompts: list) -> dict:
    """Summarise a batch of prompts before sending them anywhere.

    Args:
        prompts: A list of prompt strings.

    Returns:
        A dictionary with exactly these three keys:
            "count":        how many prompts there are
            "total_tokens": estimated tokens across all of them
            "longest":      the prompt with the most characters
                            (or None if the list is empty)

    Examples:
        >>> summarise_batch(["abcd", "a" * 40])
        {'count': 2, 'total_tokens': 11, 'longest': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
        >>> summarise_batch([])
        {'count': 0, 'total_tokens': 0, 'longest': None}
    """
    # TODO:
    #   1. Handle the empty list first.
    #   2. Loop over prompts, adding up estimate_tokens() for each.
    #   3. Track which prompt is longest (compare len()).
    #   4. Return the dictionary described above.
    # Hint: reuse estimate_tokens - don't rewrite it. max(prompts, key=len)
    #       is a neat way to find the longest, but a plain loop is fine too.
    return {}


# ======================================================================
# SELF-TEST - do not edit. Run this file to see how you are doing.
# ======================================================================

def _run_self_test() -> int:
    """Check each task and print a pass/fail report. Returns the failure count."""

    checks = [
        # (task name, what we ran, what we got, what we expected)
        ("1. build_prompt (defaults)",
         'build_prompt("embeddings")',
         build_prompt("embeddings"),
         "Explain embeddings in under 100 words. Audience level: beginner."),

        ("1. build_prompt (all args)",
         'build_prompt("attention", "expert", 50)',
         build_prompt("attention", "expert", 50),
         "Explain attention in under 50 words. Audience level: expert."),

        ("2. estimate_tokens (empty)",
         'estimate_tokens("")',
         estimate_tokens(""),
         0),

        ("2. estimate_tokens (4 chars)",
         'estimate_tokens("abcd")',
         estimate_tokens("abcd"),
         1),

        ("2. estimate_tokens (40 chars)",
         'estimate_tokens("a" * 40)',
         estimate_tokens("a" * 40),
         10),

        ("2. estimate_tokens (1 char floors to 1)",
         'estimate_tokens("a")',
         estimate_tokens("a"),
         1),

        ("3. estimate_cost (1M tokens)",
         "estimate_cost(1_000_000, 0.15)",
         round(estimate_cost(1_000_000, 0.15), 8),
         0.15),

        ("3. estimate_cost (1k tokens)",
         "estimate_cost(1000, 0.15)",
         round(estimate_cost(1000, 0.15), 8),
         0.00015),

        ("4. build_messages",
         'build_messages("Be brief.", "Hi")',
         build_messages("Be brief.", "Hi"),
         [{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "Hi"}]),

        ("5. summarise_batch (two prompts)",
         'summarise_batch(["abcd", "a" * 40])',
         summarise_batch(["abcd", "a" * 40]),
         {"count": 2, "total_tokens": 11, "longest": "a" * 40}),

        ("5. summarise_batch (empty)",
         "summarise_batch([])",
         summarise_batch([]),
         {"count": 0, "total_tokens": 0, "longest": None}),
    ]

    print()
    print("=" * 68)
    print("  LAB 2 SELF-TEST")
    print("=" * 68)
    print()

    failures = 0
    for name, call, got, expected in checks:
        if got == expected:
            print(f"[ OK ]  {name}")
        else:
            failures += 1
            print(f"[FAIL]  {name}")
            print(f"          ran:      {call}")
            print(f"          got:      {got!r}")
            print(f"          expected: {expected!r}")

    print()
    print("-" * 68)
    if failures == 0:
        print("  All 11 checks passed. Nice work - move on to Part C in the lab brief.")
    else:
        print(f"  {failures} of {len(checks)} checks failing. Keep going.")
        print("  Each failure shows what it expected. Reread the section named")
        print("  in the function's docstring if you are stuck.")
    print("-" * 68)
    print()

    return failures


if __name__ == "__main__":
    _run_self_test()
