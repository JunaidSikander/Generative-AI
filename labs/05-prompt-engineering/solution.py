"""
solution.py - Lab 5 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import json
import re
from dataclasses import dataclass


# ======================================================================
# TASK 1 - build_anatomy_prompt
# ======================================================================

def build_anatomy_prompt(instruction: str, context: str, input_data: str,
                         output_format: str) -> str:
    """Assemble the four parts of a prompt into one string."""
    # The triple-quote delimiters around input_data create a clear boundary
    # between instructions and data. This improves reliability - though it is
    # a mitigation, not a security control (Module 11 covers real defences).
    return (
        f"{instruction}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f'INPUT:\n"""\n{input_data}\n"""\n\n'
        f"{output_format}"
    )


# ======================================================================
# TASK 2 - PromptTemplate
# ======================================================================

@dataclass
class PromptTemplate:
    """A reusable, versioned prompt."""
    name: str
    version: str
    system: str
    user_template: str

    def build(self, **variables) -> list:
        """Render this template into an API-ready messages list.

        Raises:
            KeyError: if a placeholder has no matching variable.
        """
        # str.format raises KeyError on a missing key. We deliberately let it
        # propagate: a prompt shipped with a literal "{ticket}" in it produces
        # plausible-looking nonsense, which is far harder to debug than a crash.
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(**variables)},
        ]


# ======================================================================
# TASK 3 - format_few_shot_messages
# ======================================================================

def format_few_shot_messages(system: str, examples: list, user_input: str) -> list:
    """Build a few-shot messages list using alternating user/assistant turns."""
    messages = [{"role": "system", "content": system}]

    # Each example becomes a completed exchange. The model sees a pattern of
    # "this input produced that output" in exactly the conversational shape
    # it was instruction-tuned on.
    for example_input, example_output in examples:
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": example_output})

    # The real request goes last, so it benefits from recency.
    messages.append({"role": "user", "content": user_input})
    return messages


# ======================================================================
# TASK 4 - extract_json
# ======================================================================

def extract_json(text: str):
    """Pull the first JSON value out of a model response."""
    # Guard the inputs that actually show up: None from a failed API call,
    # and empty strings from a refused or truncated response.
    if not text or not isinstance(text, str):
        return None

    # STRATEGY 1: the whole response is JSON. Fastest and most common when
    # you have used JSON mode or a well-constrained prompt.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # STRATEGY 2: a fenced code block. Very common - chat-tuned models like
    # wrapping code in fences because their training data is full of them.
    #   (?:json)?  optional language tag, non-capturing
    #   (.*?)      the content, non-greedy so we stop at the FIRST closing fence
    #   re.DOTALL  so . matches newlines
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # STRATEGY 3: find the outermost braces or brackets and try that slice.
    # A heuristic, not a parser - see SOLUTION.md for where it breaks.
    for opening, closing in [("{", "}"), ("[", "]")]:
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    # Nothing worked. Returning None (rather than raising) keeps this an
    # ordinary branch in the caller's control flow.
    return None


# ======================================================================
# TASK 5 - validate_ticket
# ======================================================================

VALID_CATEGORIES = ["billing", "technical", "account", "other"]


def validate_ticket(data) -> tuple:
    """Validate a parsed ticket classification against our schema."""
    errors = []

    # Fail fast on the wrong shape. extract_json can legitimately return a
    # list, a string or None, and none of those have .get().
    if not isinstance(data, dict):
        return (False, [f"expected a JSON object, got {type(data).__name__}"])

    # --- category ---
    if "category" not in data:
        errors.append("missing key: category")
    elif data["category"] not in VALID_CATEGORIES:
        errors.append(
            f"invalid category {data['category']!r}; "
            f"expected one of {VALID_CATEGORIES}"
        )

    # --- urgency ---
    if "urgency" not in data:
        errors.append("missing key: urgency")
    else:
        urgency = data["urgency"]
        # isinstance(True, int) is True in Python, so booleans must be excluded
        # explicitly or True would pass as a valid urgency of 1.
        if isinstance(urgency, bool) or not isinstance(urgency, int):
            errors.append(f"urgency must be an integer, got {type(urgency).__name__}")
        elif not 1 <= urgency <= 5:
            errors.append(f"urgency must be 1-5, got {urgency}")

    return (len(errors) == 0, errors)


# ======================================================================
# The prompt library
# ======================================================================

CLASSIFY_ZERO_SHOT = PromptTemplate(
    name="classify_ticket_zero_shot",
    version="1.0",
    system="You are a support-ticket classifier.",
    user_template='Classify this ticket:\n"""\n{ticket}\n"""',
)

CLASSIFY_STRUCTURED = PromptTemplate(
    name="classify_ticket_structured",
    version="1.0",
    system=(
        "You are a support-ticket classifier.\n\n"
        "CATEGORIES:\n"
        "- billing    (payments, invoices, refunds)\n"
        "- technical  (bugs, errors, outages)\n"
        "- account    (login, permissions, profile)\n"
        "- other      (anything else)\n\n"
        "Respond ONLY with JSON: "
        '{"category": "<one of the above>", "urgency": <integer 1-5>}\n'
        "No prose, no code fences."
    ),
    user_template='TICKET:\n"""\n{ticket}\n"""',
)


# ======================================================================
# Demonstrations (no API key needed)
# ======================================================================

def demo_extract_json():
    print("=" * 72)
    print("  extract_json AGAINST REAL RESPONSE SHAPES")
    print("=" * 72)
    print()

    cases = [
        ("bare JSON", '{"category": "billing", "urgency": 4}'),
        ("```json fence", '```json\n{"category": "billing", "urgency": 4}\n```'),
        ("bare ``` fence", '```\n{"category": "billing", "urgency": 4}\n```'),
        ("prose wrapper", 'Here is my classification:\n\n'
                          '{"category": "billing", "urgency": 4}\n\nHope this helps!'),
        ("prose only", 'The category is billing with urgency 4.'),
        ("refusal", 'I cannot classify that ticket.'),
        ("empty", ''),
        ("wrong schema", '{"cat": "billing", "priority": 4}'),
        ("urgency out of range", '{"category": "billing", "urgency": 99}'),
        ("urgency as string", '{"category": "billing", "urgency": "high"}'),
    ]

    for label, response in cases:
        parsed = extract_json(response)
        if parsed is None:
            print(f"  {label:<24} -> NOT PARSED")
        else:
            ok, errors = validate_ticket(parsed)
            verdict = "VALID" if ok else "invalid"
            print(f"  {label:<24} -> parsed, {verdict}")
            for error in errors:
                print(f"  {'':<24}    - {error}")
    print()
    print("  Two distinct failure modes, and they need different fixes:")
    print("   - NOT PARSED   -> constrain the output format harder")
    print("   - parsed, invalid -> validate, then retry with the error fed back")
    print()


def demo_few_shot_shapes():
    print("=" * 72)
    print("  FEW-SHOT: TEXT BLOCK vs MESSAGE TURNS")
    print("=" * 72)
    print()

    examples = [
        ("Best meal I have ever had!", "positive"),
        ("Wrong order, never coming back.", "negative"),
        ("Food was okay, nothing special.", "neutral"),
    ]
    query = "It was delivered cold and tasteless."

    # --- Shape A: one text block ---
    block = "Classify each review as positive, negative, or neutral.\n\n"
    for example_input, example_output in examples:
        block += f'"{example_input}" -> {example_output}\n'
    block += f'\n"{query}" ->'

    print("  SHAPE A - single text block (1 message):")
    for line in block.splitlines():
        print(f"    {line}")
    print()

    # --- Shape B: message turns ---
    messages = format_few_shot_messages(
        "Classify reviews as positive, negative or neutral. One word only.",
        examples, query,
    )
    print(f"  SHAPE B - message turns ({len(messages)} messages):")
    for message in messages:
        print(f"    {message['role']:>9}: {message['content']}")
    print()
    print("  Shape B is generally more reliable on chat models: it matches the")
    print("  conversational format they were instruction-tuned on, and the role")
    print("  tags make the boundary between example and answer unambiguous.")
    print()


def demo_rubric_prompt():
    print("=" * 72)
    print("  VAGUE SCORING vs A RUBRIC")
    print("=" * 72)
    print()

    print("  WEAK:")
    print('    "Score this code out of 100."')
    print("    -> the model picks a number, then invents justification to fit.")
    print()
    print("  STRONG:")
    rubric = """    Evaluate the code against EACH criterion separately.
    For each: state the score, then quote the specific lines that justify it.
    Sum the scores only at the end.

    1. CORRECTNESS (1-5)
       Does it handle empty input, single-element input, and duplicates
       without raising an unhandled exception?

    2. EFFICIENCY (0 or 5)
       Is the primary sort O(n log n) or better? Justify by naming the
       algorithm used.

    3. DOCUMENTATION (0 or 1)
       Does every public function document its parameters, return type,
       and the exceptions it can raise?"""
    print(rubric)
    print()
    print("  Three things make the strong version work: per-criterion scoring,")
    print("  required evidence, and summing LAST so the total is a consequence")
    print("  of the analysis rather than a starting point.")
    print()


if __name__ == "__main__":
    demo_extract_json()
    demo_few_shot_shapes()
    demo_rubric_prompt()
