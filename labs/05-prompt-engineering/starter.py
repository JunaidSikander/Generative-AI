"""
starter.py - Lab 5: The Prompt Workbench

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-5) is pure Python - no API key, no packages, always runs.
PART 2 (the A/B experiment) needs an API key or Ollama, and is optional.
"""

import json
import re
from dataclasses import dataclass


# ======================================================================
# TASK 1 - build_anatomy_prompt
# Module 5, section 5.2
# ======================================================================

def build_anatomy_prompt(instruction: str, context: str, input_data: str,
                         output_format: str) -> str:
    """Assemble the four parts of a prompt into one string.

    Args:
        instruction:   The task, e.g. "Classify the support ticket".
        context:       Background the model needs.
        input_data:    The actual text to process.
        output_format: The required shape of the answer.

    Returns:
        A prompt with each part on its own labelled block, in this EXACT
        format (note the delimiters around the input):

            {instruction}

            CONTEXT:
            {context}

            INPUT:
            \"\"\"
            {input_data}
            \"\"\"

            {output_format}

    Examples:
        >>> print(build_anatomy_prompt("Do X.", "Ctx.", "Data.", "Return Y."))
        Do X.
        <BLANKLINE>
        CONTEXT:
        Ctx.
        <BLANKLINE>
        INPUT:
        \"\"\"
        Data.
        \"\"\"
        <BLANKLINE>
        Return Y.
    """
    # TODO: build the string above. Use an f-string; triple quotes help.
    # The delimiters around input_data are not decoration - they stop the
    # data being read as instructions (Module 5, section 5.2).
    return ""


# ======================================================================
# TASK 2 - PromptTemplate
# Module 5, section 5.10
# ======================================================================

@dataclass
class PromptTemplate:
    """A reusable, versioned prompt.

    Attributes:
        name:          Identifier for logging.
        version:       Bump this when you change the prompt, so a quality
                       regression can be traced to a specific change.
        system:        The stable system prompt.
        user_template: A format string with {placeholders}.
    """
    name: str
    version: str
    system: str
    user_template: str

    def build(self, **variables) -> list:
        """Render this template into an API-ready messages list.

        Args:
            **variables: Values for the placeholders in user_template.

        Returns:
            [{"role": "system", ...}, {"role": "user", ...}]

        Raises:
            KeyError: if a placeholder has no matching variable. Failing
                loudly here is deliberate - a prompt containing a literal
                "{ticket}" produces plausible-looking garbage instead of
                an error, which is far worse to debug.

        Examples:
            >>> t = PromptTemplate("t", "1.0", "Be brief.", "Q: {q}")
            >>> t.build(q="Why?")
            [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Q: Why?'}]
        """
        # TODO:
        #   Return a two-item list: the system message, then the user message
        #   with user_template.format(**variables) as its content.
        #   str.format already raises KeyError on a missing key - do not
        #   suppress it.
        return []


# ======================================================================
# TASK 3 - format_few_shot_messages
# Module 5, section 5.5
# ======================================================================

def format_few_shot_messages(system: str, examples: list, user_input: str) -> list:
    """Build a few-shot messages list using alternating user/assistant turns.

    This shape works better than a text block of examples, because it matches
    the conversational format the model was instruction-tuned on.

    Args:
        system:     The system prompt.
        examples:   List of (input, output) tuples to demonstrate.
        user_input: The real request, appended last.

    Returns:
        A messages list:
            [system]
            [user: example 1 input], [assistant: example 1 output]
            [user: example 2 input], [assistant: example 2 output]
            ...
            [user: user_input]

        So the length is always 2 * len(examples) + 2.

    Examples:
        >>> format_few_shot_messages("S", [("a", "b")], "c")
        [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'a'}, \
{'role': 'assistant', 'content': 'b'}, {'role': 'user', 'content': 'c'}]
    """
    # TODO:
    #   1. Start with the system message.
    #   2. Loop over examples, appending a user message then an assistant
    #      message for each.
    #   3. Append the real user_input last.
    return []


# ======================================================================
# TASK 4 - extract_json
# Module 5, section 5.8 - the function you will reuse in every project
# ======================================================================

def extract_json(text: str):
    """Pull the first JSON value out of a model response.

    Models return JSON in three shapes, and you need to survive all of them:
      1. Bare JSON.
      2. JSON inside a ```json ... ``` fenced block.
      3. JSON buried in explanatory prose.

    Args:
        text: The raw model response.

    Returns:
        The parsed object, or None if nothing valid was found.

        Note it returns None rather than raising: with LLMs, a failed
        extraction is an expected outcome to handle in your control flow,
        not an exceptional one.

    Examples:
        >>> extract_json('{"a": 1}')
        {'a': 1}
        >>> extract_json('```json\\n{"a": 1}\\n```')
        {'a': 1}
        >>> extract_json('Here you go:\\n{"a": 1}\\nHope that helps!')
        {'a': 1}
        >>> extract_json('Sorry, I cannot help.') is None
        True
    """
    # TODO - three strategies, tried in order:
    #
    #   STRATEGY 1: the whole string is JSON.
    #       try json.loads(text.strip()) inside try/except json.JSONDecodeError
    #
    #   STRATEGY 2: a fenced code block.
    #       re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    #       then json.loads on group(1)
    #
    #   STRATEGY 3: the outermost braces or brackets.
    #       for opening, closing in [("{", "}"), ("[", "]")]:
    #           use text.find(opening) and text.rfind(closing)
    #
    #   If all three fail, return None.
    #
    # Guard against text being None or empty first.
    return None


# ======================================================================
# TASK 5 - validate_ticket
# Module 5, section 5.8
# ======================================================================

VALID_CATEGORIES = ["billing", "technical", "account", "other"]


def validate_ticket(data) -> tuple:
    """Validate a parsed ticket classification against our schema.

    Required schema:
        category: one of VALID_CATEGORIES
        urgency:  an integer from 1 to 5 inclusive

    Args:
        data: The parsed model output. May be None or the wrong type.

    Returns:
        (is_valid, errors) where errors is a list of human-readable strings.
        An empty error list means valid.

    Examples:
        >>> validate_ticket({"category": "billing", "urgency": 3})
        (True, [])
        >>> ok, errors = validate_ticket({"category": "billing"})
        >>> ok, len(errors)
        (False, 1)
    """
    errors = []

    # TODO:
    #   1. If data is not a dict, return (False, ["..."]) immediately.
    #   2. Check "category" exists AND is in VALID_CATEGORIES.
    #      Add ONE error if it is missing, ONE if it is present but invalid.
    #   3. Check "urgency" exists, is an int, and is 1-5.
    #      Add exactly ONE error per problem.
    #   4. Return (len(errors) == 0, errors).
    #
    # CAREFUL: isinstance(True, int) is True in Python, so a boolean would
    # sneak through a naive integer check. Exclude bool explicitly.

    return (len(errors) == 0, errors)


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _run_self_test() -> int:
    checks = []

    # --- TASK 1 ---
    expected_prompt = (
        'Do X.\n\n'
        'CONTEXT:\n'
        'Ctx.\n\n'
        'INPUT:\n'
        '"""\n'
        'Data.\n'
        '"""\n\n'
        'Return Y.'
    )
    checks.append(("1. build_anatomy_prompt exact format",
                   build_anatomy_prompt("Do X.", "Ctx.", "Data.", "Return Y."),
                   expected_prompt))

    # --- TASK 2 ---
    template = PromptTemplate("t", "1.0", "Be brief.", "Q: {q}")
    checks.append(("2. PromptTemplate.build returns 2 messages",
                   template.build(q="Why?"),
                   [{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Q: Why?"}]))

    # A missing variable must raise, not silently pass the placeholder through.
    raised = False
    try:
        template.build(wrong_name="x")
    except KeyError:
        raised = True
    except Exception:
        pass
    checks.append(("2. PromptTemplate.build raises on missing variable",
                   raised, True))

    # --- TASK 3 ---
    checks.append(("3. format_few_shot_messages (1 example)",
                   format_few_shot_messages("S", [("a", "b")], "c"),
                   [{"role": "system", "content": "S"},
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"}]))
    checks.append(("3. format_few_shot_messages length = 2n + 2",
                   len(format_few_shot_messages("S", [("a", "b"), ("c", "d"),
                                                      ("e", "f")], "g")),
                   8))
    checks.append(("3. format_few_shot_messages with no examples",
                   format_few_shot_messages("S", [], "c"),
                   [{"role": "system", "content": "S"},
                    {"role": "user", "content": "c"}]))

    # --- TASK 4 ---
    checks.append(("4. extract_json bare object",
                   extract_json('{"category": "billing"}'),
                   {"category": "billing"}))
    checks.append(("4. extract_json ```json fence",
                   extract_json('```json\n{"category": "billing"}\n```'),
                   {"category": "billing"}))
    checks.append(("4. extract_json bare ``` fence",
                   extract_json('```\n{"category": "billing"}\n```'),
                   {"category": "billing"}))
    checks.append(("4. extract_json wrapped in prose",
                   extract_json('Here is the result:\n\n{"category": "billing"}\n\nLet me know!'),
                   {"category": "billing"}))
    checks.append(("4. extract_json array",
                   extract_json('[1, 2, 3]'), [1, 2, 3]))
    checks.append(("4. extract_json nested",
                   extract_json('{"a": {"b": [1, 2]}}'), {"a": {"b": [1, 2]}}))
    checks.append(("4. extract_json returns None on no JSON",
                   extract_json('Sorry, I cannot help with that.'), None))
    checks.append(("4. extract_json returns None on empty string",
                   extract_json(''), None))
    checks.append(("4. extract_json handles None input",
                   extract_json(None), None))

    # --- TASK 5 ---
    checks.append(("5. validate_ticket valid", validate_ticket(
        {"category": "billing", "urgency": 3}), (True, [])))

    for label, payload, expected_count in [
        ("missing urgency", {"category": "billing"}, 1),
        ("missing both", {}, 2),
        ("bad category", {"category": "banana", "urgency": 3}, 1),
        ("urgency too high", {"category": "billing", "urgency": 9}, 1),
        ("urgency wrong type", {"category": "billing", "urgency": "three"}, 1),
        ("urgency is a bool", {"category": "billing", "urgency": True}, 1),
    ]:
        ok, errors = validate_ticket(payload)
        checks.append((f"5. validate_ticket {label}", (ok, len(errors)),
                       (False, expected_count)))

    ok, errors = validate_ticket(None)
    checks.append(("5. validate_ticket rejects None",
                   (ok, len(errors) >= 1), (False, True)))

    print()
    print("=" * 72)
    print("  LAB 5 SELF-TEST - the prompt workbench")
    print("=" * 72)
    print()

    failures = 0
    for name, got, expected in checks:
        if got == expected:
            print(f"[ OK ]  {name}")
        else:
            failures += 1
            print(f"[FAIL]  {name}")
            print(f"          got:      {got!r}")
            print(f"          expected: {expected!r}")

    print()
    print("-" * 72)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  extract_json is the one you will paste into every future project.")
    else:
        print(f"  {failures} of {len(checks)} failing. Keep going.")
    print("-" * 72)
    print()
    return failures


# ======================================================================
# Demo - the prompt library in use (no API needed)
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


def demo_prompt_library():
    print("=" * 72)
    print("  DEMO: two prompts for the same task")
    print("=" * 72)
    print()

    ticket = "I was charged twice for my subscription this month. Please refund."

    for template in [CLASSIFY_ZERO_SHOT, CLASSIFY_STRUCTURED]:
        messages = template.build(ticket=ticket)
        if not messages:
            print("  PromptTemplate.build not implemented yet - finish TASK 2.")
            print()
            return

        total_chars = sum(len(m["content"]) for m in messages)
        # Collapse newlines so a multi-line system prompt stays on one line here.
        preview = messages[0]["content"].replace("\n", " ")[:58]
        print(f"  {template.name}  (v{template.version})")
        print(f"    messages: {len(messages)}   total chars: {total_chars}")
        print(f"    system: {preview}...")
        print()

    print("  The structured version costs more characters on EVERY call.")
    print("  Whether that is worth it is an empirical question - which is")
    print("  exactly what the Part 2 experiment measures.")
    print()


def demo_extract_json_robustness():
    print("=" * 72)
    print("  DEMO: what real model output actually looks like")
    print("=" * 72)
    print()

    real_world_responses = [
        '{"category": "billing", "urgency": 4}',
        '```json\n{"category": "billing", "urgency": 4}\n```',
        'Based on the ticket, here is my classification:\n\n'
        '{"category": "billing", "urgency": 4}\n\nHope this helps!',
        '```\n{"category": "billing", "urgency": 4}\n```',
        'The category is billing with urgency 4.',
    ]

    for i, response in enumerate(real_world_responses, start=1):
        parsed = extract_json(response)
        preview = response.replace("\n", "\\n")[:52]
        status = "parsed  " if parsed is not None else "FAILED  "
        print(f"  {i}. {status} {preview}...")
        if parsed is not None:
            ok, errors = validate_ticket(parsed)
            print(f"     -> {parsed}   valid={ok}"
                  + (f"  errors={errors}" if errors else ""))
        print()

    print("  Response 5 is the important one. It contains the right answer in")
    print("  prose, and no extractor can reliably parse it. That is why you")
    print("  constrain the OUTPUT FORMAT rather than trying to parse anything.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        demo_prompt_library()
        demo_extract_json_robustness()
    else:
        print("  Fix the self-test first, then the demos will run.")
        print()
