"""
solution.py - Lab 11 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import re
import time


# ======================================================================
# TASK 1 - precision_recall_f1
# ======================================================================

def precision_recall_f1(predicted, actual) -> dict:
    """Compute precision, recall and F1 from two collections."""
    # set() both: duplicates in a prediction list would otherwise inflate the
    # denominator and quietly depress precision.
    predicted, actual = set(predicted), set(actual)

    true_positives = len(predicted & actual)

    # Guard both denominators. An empty prediction set or an empty truth set
    # makes the corresponding metric undefined; 0.0 is the usual convention
    # and it avoids ZeroDivisionError on the edge cases that really occur.
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(actual) if actual else 0.0

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": len(predicted - actual),
        "false_negatives": len(actual - predicted),
    }


# ======================================================================
# TASK 2 - mean_reciprocal_rank
# ======================================================================

def mean_reciprocal_rank(rankings: list, relevant_sets: list) -> float:
    """Average of 1 / (rank of the first relevant result)."""
    if not rankings:
        return 0.0

    total = 0.0
    for ranking, relevant in zip(rankings, relevant_sets):
        for position, item in enumerate(ranking, start=1):
            if item in relevant:
                total += 1.0 / position
                # BREAK: only the first relevant result counts. MRR asks
                # "how high was the right answer?", not "how many did we find?"
                break
        # A ranking with no relevant result contributes 0 - no else needed.

    return total / len(rankings)


# ======================================================================
# TASK 3 - evaluate_retrieval
# ======================================================================

def evaluate_retrieval(cases: list, retrieve_fn, top_k: int = 4) -> dict:
    """Run an evaluation set against a retriever and report the metrics."""
    if not cases:
        return {"total": 0, "hits": 0, "recall_at_k": 0.0, "mrr": 0.0, "cases": []}

    case_results = []

    for case in cases:
        chunks = retrieve_fn(case["question"])[:top_k]
        expected = case["expected_contains"].lower()

        rank = None
        for position, chunk in enumerate(chunks, start=1):
            # Case-insensitive substring: a deliberately crude oracle. It
            # tells you the key text was RETRIEVED, not that the chunk
            # answers the question - but it is free and deterministic.
            if expected in chunk.lower():
                rank = position
                break

        case_results.append({"id": case.get("id"), "hit": rank is not None,
                             "rank": rank})

    hits = sum(1 for result in case_results if result["hit"])

    # Misses contribute 0 to the MRR total, but still count in the divisor -
    # otherwise a retriever that finds one case out of fifty scores 1.0.
    reciprocal_total = sum(1.0 / result["rank"]
                           for result in case_results if result["rank"])

    return {
        "total": len(case_results),
        "hits": hits,
        "recall_at_k": hits / len(case_results),
        "mrr": reciprocal_total / len(case_results),
        "cases": case_results,
    }


# ======================================================================
# TASK 4 - redact_pii
# ======================================================================

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[API_KEY]"),
    (re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,15}(?![\w-])"), "[CARD]"),
    (re.compile(r"(?<![\w+])\+?\d(?:[\s().-]{0,2}\d){8,}(?![\w])"), "[PHONE]"),
]


def redact_pii(text: str) -> tuple:
    """Replace obvious PII patterns with labels."""
    if not text:
        return (text, {})

    counts = {}
    for pattern, label in PII_PATTERNS:
        # subn returns (new_string, number_of_substitutions) - exactly the
        # count we want for monitoring, with no second pass needed.
        text, replacements = pattern.subn(label, text)
        if replacements:
            counts[label] = counts.get(label, 0) + replacements

    return (text, counts)


# ======================================================================
# TASK 5 - screen_input
# ======================================================================

MAX_INPUT_CHARS = 10_000

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|the\s+above)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"reveal\s+your\s+(instructions|prompt|system)", re.I),
    re.compile(r"system\s*prompt", re.I),
]


def screen_input(text: str) -> dict:
    """Fast deterministic checks before anything expensive runs."""
    problems = []
    injection_flags = []

    # Guard falsy input FIRST - len(None) raises.
    if not text or not text.strip():
        problems.append("empty input")
    else:
        if len(text) > MAX_INPUT_CHARS:
            problems.append(
                f"too long: {len(text):,} chars (max {MAX_INPUT_CHARS:,})")

        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                injection_flags.append(pattern.pattern)

    # "allowed" depends on `problems` ONLY. Injection flags are telemetry:
    # this is a blocklist, it misses paraphrase and other languages, and it
    # false-positives on legitimate questions about prompting. Blocking on it
    # would break real users while stopping only careless attackers.
    return {"allowed": not problems, "problems": problems,
            "injection_flags": injection_flags}


# ======================================================================
# TASK 6 - backoff_delays
# ======================================================================

def backoff_delays(attempts: int, base: float = 1.0, factor: float = 2.0,
                   max_delay: float = 60.0, jitter=None) -> list:
    """Compute exponential backoff delays."""
    if attempts < 0:
        raise ValueError(f"attempts must be non-negative, got {attempts}")

    delays = []
    for attempt in range(attempts):
        # Cap BEFORE jitter, so jitter can never push a delay above max_delay.
        raw = min(base * (factor ** attempt), max_delay)
        delays.append(raw * jitter(attempt) if jitter else raw)

    return delays


# ======================================================================
# TASK 7 - CircuitBreaker
# ======================================================================

class CircuitBreaker:
    """Stop calling a service that is clearly down."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0,
                 clock=None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        # Injecting the clock is what makes this testable. With
        # time.monotonic hard-coded, testing the recovery timeout would mean
        # actually sleeping for 30 seconds.
        self.clock = clock or time.monotonic

        self.state = "closed"
        self.consecutive_failures = 0
        self.opened_at = None

    def allow_request(self) -> bool:
        """May a request proceed right now?"""
        if self.state == "open":
            if self.clock() - self.opened_at >= self.recovery_timeout:
                # Transition to half_open and permit exactly one trial.
                self.state = "half_open"
                return True
            return False

        # closed, or half_open with a trial pending.
        return True

    def record_success(self) -> None:
        """Record a successful call. Resets everything to healthy."""
        self.state = "closed"
        # Resetting the COUNTER is the important half. Without it, a service
        # failing 1% of the time trips the breaker after a few hundred
        # requests despite being perfectly healthy.
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        """Record a failed call. May open the circuit."""
        self.consecutive_failures += 1

        # A failed trial re-opens IMMEDIATELY. We just proved the service is
        # still down, so waiting for the threshold again would send more
        # doomed traffic at it.
        if (self.state == "half_open"
                or self.consecutive_failures >= self.failure_threshold):
            self.state = "open"
            self.opened_at = self.clock()


class FakeClock:
    """A controllable clock, so the circuit breaker is actually testable."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ======================================================================
# BONUS - the pieces the tasks do not cover
# ======================================================================

FAITHFULNESS_JUDGE_PROMPT = """You are evaluating whether an ANSWER is faithful
to its CONTEXT. Faithful means every factual claim is supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate in this exact order:
1. List every factual claim the ANSWER makes.
2. For each claim, quote the CONTEXT text that supports it, or write
   NOT SUPPORTED.
3. Only then, give a verdict.

Return JSON:
{{
  "claims": [{{"claim": "...", "support": "..." or null}}],
  "unsupported_count": <integer>,
  "verdict": "faithful" | "partially_faithful" | "unfaithful"
}}

Do not be generous. A claim that is plausible but absent from the context is
NOT SUPPORTED.
"""


def judge_agreement(judge_verdicts: list, human_verdicts: list) -> dict:
    """Measure how often an LLM judge agrees with human labels.

    The step almost everyone skips. An unvalidated judge produces numbers
    that FEEL like measurement and are not - and you will optimise against
    them (Module 11, section 11.9).

    Returns:
        {"agreement": float, "n": int, "interpretation": str}
    """
    if not judge_verdicts:
        return {"agreement": 0.0, "n": 0, "interpretation": "no data"}

    matches = sum(1 for judged, human in zip(judge_verdicts, human_verdicts)
                  if judged == human)
    agreement = matches / len(judge_verdicts)

    if agreement > 0.85:
        interpretation = "trustworthy for tracking trends"
    elif agreement >= 0.70:
        interpretation = "directional only - do not make fine decisions on it"
    else:
        interpretation = "MEASURING SOMETHING ELSE - fix the judge prompt first"

    return {"agreement": agreement, "n": len(judge_verdicts),
            "interpretation": interpretation}


def generate_with_retry(call_model, validate, max_attempts: int = 3,
                        fallback: str = "I can't help with that right now."):
    """Generate, validate, and retry with the SPECIFIC failure fed back.

    Args:
        call_model: Callable(list_of_messages) -> str.
        validate:   Callable(text) -> (is_valid, problems).
        max_attempts: Hard cap. Beyond ~3, retrying rarely helps.
        fallback:   Returned when every attempt fails.

    Returns:
        {"ok", "content", "attempts", "problems"}
    """
    messages = []
    problems = []

    for attempt in range(1, max_attempts + 1):
        response = call_model(messages)
        is_valid, problems = validate(response)

        if is_valid:
            return {"ok": True, "content": response,
                    "attempts": attempt, "problems": []}

        # Feed back the SPECIFIC failure. A generic "try again" wastes a call;
        # "field 'total' must be a number" is actionable.
        messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user",
             "content": f"That response was invalid: {'; '.join(problems)}. "
                        f"Return a corrected response only."},
        ]

    # Exhausted retries is an EXPECTED outcome, not an exception. Return a
    # defined failure with a fallback that cannot itself fail.
    return {"ok": False, "content": fallback,
            "attempts": max_attempts, "problems": problems}


# ======================================================================
# Demonstrations
# ======================================================================

def demo_precision_recall_tradeoff():
    print("=" * 76)
    print("  THE PRECISION / RECALL TRADE-OFF IS A PRODUCT DECISION")
    print("=" * 76)
    print()

    # A moderation classifier at various thresholds.
    actual_abusive = set(range(20))
    total = 1000

    thresholds = {
        "0.9 (very strict)": set(range(6)),
        "0.7": set(range(14)) | {500, 501},
        "0.5 (balanced)": set(range(18)) | set(range(500, 510)),
        "0.3": set(range(20)) | set(range(500, 540)),
        "0.1 (very lax)": set(range(20)) | set(range(500, 700)),
    }

    print(f"  1000 requests, {len(actual_abusive)} genuinely abusive.")
    print()
    print(f"  {'threshold':<20}{'precision':>11}{'recall':>9}{'F1':>7}"
          f"{'missed':>9}{'false alarms':>15}")
    print("  " + "-" * 71)

    for label, predicted in thresholds.items():
        m = precision_recall_f1(predicted, actual_abusive)
        print(f"  {label:<20}{m['precision']:>11.2f}{m['recall']:>9.2f}"
              f"{m['f1']:>7.2f}{m['false_negatives']:>9}"
              f"{m['false_positives']:>15}")

    print()
    print("  F1 peaks somewhere in the middle - but F1 weights the two errors")
    print("  EQUALLY, which is almost never what you want.")
    print()
    print("  For content moderation, a missed violation is usually worse than")
    print("  a false block, so you want the high-recall end even though F1")
    print("  says otherwise. For auto-approving expense claims you want the")
    print("  opposite end.")
    print()
    print("  Decide which error costs more FIRST. Then pick the metric that")
    print("  reflects that decision - do not let F1 decide for you.")
    print()


def demo_judge_validation():
    print("=" * 76)
    print("  VALIDATING AN LLM JUDGE AGAINST HUMANS")
    print("=" * 76)
    print()

    scenarios = {
        "well-designed judge": (
            ["faithful"] * 17 + ["unfaithful"] * 3,
            ["faithful"] * 16 + ["unfaithful"] * 4,
        ),
        "judge with a mild bias": (
            ["faithful"] * 16 + ["unfaithful"] * 4,
            ["faithful"] * 11 + ["unfaithful"] * 9,
        ),
        "judge measuring the wrong thing": (
            ["faithful"] * 18 + ["unfaithful"] * 2,
            ["faithful"] * 8 + ["unfaithful"] * 12,
        ),
    }

    for label, (judged, human) in scenarios.items():
        result = judge_agreement(judged, human)
        print(f"  {label:<36} {result['agreement']:>6.0%}  "
              f"{result['interpretation']}")

    print()
    print("  The third judge produces confident numbers on every run. They")
    print("  are not measurement - they correlate with something other than")
    print("  what a human considers faithful.")
    print()
    print("  A confident WRONG metric is worse than no metric, because you")
    print("  will optimise against it and think you are improving.")
    print()
    print("  30-50 human-labelled cases is an afternoon. Do it before you")
    print("  trust a single number the judge produces.")
    print()


def demo_retry_with_feedback():
    print("=" * 76)
    print("  RETRY: SPECIFIC FEEDBACK vs A GENERIC NUDGE")
    print("=" * 76)
    print()

    import json

    def validate_json_total(text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return (False, [f"not valid JSON: {exc.msg}"])
        if "total" not in data:
            return (False, ["missing required field: total"])
        if not isinstance(data["total"], (int, float)):
            return (False, [f"field 'total' must be a number, "
                            f"got {type(data['total']).__name__}"])
        return (True, [])

    # A model that improves when told exactly what was wrong.
    responses_with_feedback = [
        'Here you go: {"total": "twelve"}',
        '{"total": "twelve"}',
        '{"total": 12.0}',
    ]
    # A model that gets only "try again" and flails.
    responses_generic = [
        'Here you go: {"total": "twelve"}',
        'Sure! {"total": "twelve pounds"}',
        'Let me try again: {"amount": 12}',
    ]

    for label, responses in [("specific feedback", responses_with_feedback),
                             ("generic 'try again'", responses_generic)]:
        queue = list(responses)
        result = generate_with_retry(
            call_model=lambda messages: queue.pop(0) if queue else "{}",
            validate=validate_json_total,
        )
        status = "SUCCEEDED" if result["ok"] else "FELL BACK"
        print(f"  {label:<24} {status} after {result['attempts']} attempts")
        if not result["ok"]:
            print(f"    last problems: {result['problems']}")
            print(f"    returned     : {result['content']!r}")

    print()
    print("  Both models were equally capable. The difference is that one was")
    print("  told 'field total must be a number, got str' and the other was")
    print("  told 'try again'.")
    print()
    print("  Note also that the failing case returns a FALLBACK, not an")
    print("  exception. Exhausted retries is an expected outcome, and a")
    print("  pre-written safe message beats a stack trace.")
    print()


def demo_layer_costs():
    print("=" * 76)
    print("  WHERE TO SPEND YOUR LATENCY BUDGET")
    print("=" * 76)
    print()

    # Illustrative orders of magnitude - measure your own.
    layers = [
        ("output schema validation", 1, "catches failures from ANY cause"),
        ("input length / type screen", 1, "bounds cost per request"),
        ("PII redaction", 2, "backstop, plus a monitoring signal"),
        ("injection regex", 2, "telemetry only - it is a blocklist"),
        ("moderation API (input)", 150, "policy categories"),
        ("moderation API (output)", 150, "protects users from your output"),
        ("injection classifier", 400, "genuinely better than regex"),
        ("LLM-as-a-judge", 3000, "OFFLINE ONLY - never in the hot path"),
    ]

    base_call = 800
    print(f"  base model call: {base_call} ms")
    print()
    print(f"  {'layer':<28}{'ms':>7}{'cumulative':>13}   note")
    print("  " + "-" * 74)

    cumulative = base_call
    for label, cost, note in layers:
        if "OFFLINE" in note:
            print(f"  {label:<28}{cost:>7}{'--':>13}   {note}")
            continue
        cumulative += cost
        print(f"  {label:<28}{cost:>7}{cumulative:>12}ms   {note}")

    print()
    print(f"  Full runtime stack: {base_call} ms -> {cumulative} ms "
          f"({cumulative/base_call:.1f}x)")
    print()
    print("  Note the ordering. The first four layers cost SIX milliseconds")
    print("  combined and catch a large share of real failures. The two")
    print("  moderation calls cost fifty times that.")
    print()
    print("  So start at the top. Output schema validation is nearly free and")
    print("  catches failures you never anticipated - including ones no")
    print("  input-side check could have seen.")
    print()


if __name__ == "__main__":
    demo_precision_recall_tradeoff()
    demo_judge_validation()
    demo_retry_with_feedback()
    demo_layer_costs()
