"""
starter.py - Lab 12: Decide, Then Prepare

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-6) is pure standard library - no packages, no GPU, no API key.
You will implement the decision framework as code, validate a training dataset
against the failure modes that silently ruin a fine-tune, compute LoRA
parameter counts, and run the break-even arithmetic.

PART 2 (in the lab brief) is an optional real QLoRA fine-tune on free Colab.
"""

import json
import random
from collections import Counter, defaultdict


# ======================================================================
# TASK 1 - recommend_approach
# Module 12, section 12.1
# ======================================================================

def recommend_approach(requirements: dict) -> dict:
    """Recommend prompting, RAG or fine-tuning from stated requirements.

    Encodes Module 12, section 12.1's decision procedure. The ordering is the
    point: prompting, then RAG, then fine-tuning, and stop at the first rung
    that meets the need.

    Args:
        requirements: Any of these boolean keys (missing = False):
            needs_private_knowledge  must know documents it was not trained on
            needs_current_info       must know things after the training cutoff
            needs_citations          must attribute claims to a source
            facts_change_often       the knowledge base updates regularly
            needs_house_style        must match a specific tone or voice
            needs_specialist_task    a task the base model simply cannot do
            high_volume_cost_matters cost per request is a binding constraint
          And these:
            n_examples: int          how many labelled examples you have
            tried_prompting: bool    have you already tried a good prompt?

    Returns:
        {
          "recommendation": "prompting" | "rag" | "fine-tuning" | "rag + fine-tuning",
          "reasons": list[str],       why
          "blockers": list[str],      why fine-tuning is NOT viable yet
        }

    The rules, in priority order:
      1. If prompting has not been tried -> "prompting". Nothing else matters
         yet; most fine-tuning problems are prompt problems.
      2. Work out whether RAG is indicated: any of needs_private_knowledge,
         needs_current_info, needs_citations, facts_change_often.
      3. Work out whether fine-tuning is indicated: any of needs_house_style,
         needs_specialist_task, high_volume_cost_matters.
      4. Fine-tuning is BLOCKED if n_examples < 50 (use few-shot instead).
      5. Combine: both indicated and not blocked -> "rag + fine-tuning".
         RAG only, or fine-tuning blocked -> "rag". Fine-tuning only -> it.
         Neither -> "prompting".

    Examples:
        >>> recommend_approach({"tried_prompting": False})["recommendation"]
        'prompting'
        >>> r = recommend_approach({"tried_prompting": True,
        ...                         "needs_citations": True})
        >>> r["recommendation"]
        'rag'
    """
    # TODO:
    #   1. If not requirements.get("tried_prompting"), return immediately with
    #      "prompting" and a reason saying so.
    #   2. rag_signals  = the four RAG keys that are True
    #      tune_signals = the three fine-tuning keys that are True
    #   3. blockers: if requirements.get("n_examples", 0) < 50 and tune_signals,
    #      add a blocker naming the count.
    #   4. Apply rule 5 to pick the recommendation.
    #   5. Return the dict. Put the matched signal names into "reasons".
    return {"recommendation": "prompting", "reasons": [], "blockers": []}


# ======================================================================
# TASK 2 - validate_training_example
# Module 12, section 12.7
# ======================================================================

VALID_ROLES = {"system", "user", "assistant"}


def validate_training_example(example: dict) -> tuple:
    """Check one training example against the chat-format rules.

    Args:
        example: A dict expected to hold a "messages" list.

    Returns:
        (is_valid, problems)

    The rules:
      1. "messages" must be a non-empty list.
      2. Every message must be a dict with a valid role and non-empty content.
      3. The LAST message must have role "assistant" - it is the training
         target. Without it there is nothing to learn.
      4. A "system" message may only appear at index 0.

    Examples:
        >>> validate_training_example({"messages": [
        ...     {"role": "user", "content": "hi"},
        ...     {"role": "assistant", "content": "hello"}]})
        (True, [])
        >>> ok, problems = validate_training_example({"messages": []})
        >>> ok
        False
    """
    problems = []

    # TODO:
    #   1. messages = example.get("messages"). If it is not a list, or it is
    #      empty, return (False, ["missing or empty 'messages' list"]) at once -
    #      every check below assumes a non-empty list.
    #   2. Loop with enumerate. For each message:
    #        - not a dict            -> "message {i} is not an object", continue
    #        - role not in VALID_ROLES -> "message {i}: invalid role {role!r}"
    #        - content missing/blank   -> "message {i}: empty content"
    #      Use str(message.get("content", "")).strip() so a non-string
    #      content does not raise.
    #   3. If the last message is a dict and its role is not "assistant",
    #      add "last message must have role 'assistant'".
    #   4. For messages[1:], any role == "system" is a problem:
    #      "message {i}: system message must come first".
    #   5. Return (not problems, problems).

    return (not problems, problems)


# ======================================================================
# TASK 3 - validate_dataset
# Module 12, section 12.7
# ======================================================================

MIN_EXAMPLES = 50


def validate_dataset(examples: list, label_fn=None) -> dict:
    """Check a whole dataset for the failures that silently ruin a fine-tune.

    Per-example validation is not enough. These problems only exist at the
    dataset level, and every one of them trains happily without erroring.

    Args:
        examples: The training examples.
        label_fn: Optional callable(example) -> label, for the balance check.
                  If None, the balance check is skipped.

    Returns:
        {
          "total": int,
          "valid": int,                   examples passing task 2
          "invalid_indices": list[int],
          "duplicates": int,              exact duplicate examples
          "warnings": list[str],
          "label_counts": dict,           empty when label_fn is None
        }

    Warnings raised:
      - fewer than MIN_EXAMPLES examples
      - any invalid examples
      - any exact duplicates
      - class imbalance: the largest class is more than 10x the smallest

    Examples:
        >>> ex = [{"messages": [{"role": "user", "content": "a"},
        ...                     {"role": "assistant", "content": "b"}]}]
        >>> validate_dataset(ex)["valid"]
        1
    """
    # TODO:
    #   1. Run validate_training_example over each; collect invalid indices.
    #   2. Duplicates: serialise each example with
    #        json.dumps(example, sort_keys=True)
    #      and count how many serialisations appear more than once. The
    #      duplicate COUNT is total minus the number of distinct
    #      serialisations.
    #   3. Labels: if label_fn is given, build a Counter of labels.
    #   4. Warnings, in this order: too few, invalid present, duplicates
    #      present, imbalance (max count > 10 * min count, and at least two
    #      distinct labels).
    #   5. Return the dict above.
    return {"total": 0, "valid": 0, "invalid_indices": [], "duplicates": 0,
            "warnings": [], "label_counts": {}}


# ======================================================================
# TASK 4 - stratified_split
# Module 12, section 12.7
# ======================================================================

def stratified_split(examples: list, label_fn, validation_fraction: float = 0.2,
                     seed: int = 42) -> tuple:
    """Split into train/validation while preserving the label distribution.

    Args:
        examples:            The full dataset.
        label_fn:            Callable(example) -> hashable label.
        validation_fraction: Share of EACH label group held out.
        seed:                Fixed, so the split is reproducible.

    Returns:
        (train, validation)

    Stratified, not random: a random split of imbalanced data can put every
    example of a rare class on one side, making its validation score
    meaningless.

    Seeded: comparing two training runs across different splits tells you
    nothing.

    Raises:
        ValueError: if validation_fraction is not between 0 and 1 exclusive.

    Examples:
        >>> data = [{"y": "a"}] * 10 + [{"y": "b"}] * 10
        >>> train, val = stratified_split(data, lambda e: e["y"], 0.2)
        >>> len(train), len(val)
        (16, 4)
    """
    # TODO:
    #   1. Raise ValueError unless 0 < validation_fraction < 1.
    #   2. Group examples by label_fn into a defaultdict(list).
    #   3. rng = random.Random(seed)   -- a LOCAL generator, so this does not
    #      disturb global random state that a caller may depend on.
    #   4. For each label (iterate in a STABLE order - sort by str(label) -
    #      so the result does not depend on dict ordering):
    #        shuffle a copy, cut = int(len(group) * validation_fraction),
    #        first `cut` to validation, the rest to train.
    #   5. Return (train, validation).
    return ([], [])


# ======================================================================
# TASK 5 - LoRA parameter arithmetic
# Module 12, section 12.4
# ======================================================================

def lora_parameter_count(d_in: int, d_out: int, rank: int) -> int:
    """Trainable parameters LoRA adds for ONE weight matrix.

    LoRA approximates the update dW (d_in x d_out) as B @ A, where
    B is (d_in x rank) and A is (rank x d_out).

    Examples:
        >>> lora_parameter_count(4096, 4096, 8)
        65536
        >>> lora_parameter_count(768, 768, 8)
        12288
    """
    # TODO: return rank * (d_in + d_out).
    # Note this grows LINEARLY with d, while a full fine-tune grows with d^2 -
    # which is why LoRA saves proportionally more on bigger models.
    return 0


def full_parameter_count(d_in: int, d_out: int) -> int:
    """Parameters a FULL fine-tune would update for the same matrix.

    Examples:
        >>> full_parameter_count(4096, 4096)
        16777216
    """
    # TODO: return d_in * d_out.
    return 0


def lora_config_summary(d_model: int, n_layers: int, rank: int,
                        targets_per_layer: int = 2,
                        base_params: int = 7_000_000_000,
                        bytes_per_param: int = 2) -> dict:
    """Summarise a whole LoRA configuration.

    Args:
        d_model:           Model dimension (assumes square weight matrices).
        n_layers:          Transformer layers getting adapters.
        rank:              LoRA rank.
        targets_per_layer: Weight matrices adapted per layer, e.g. 2 for
                           q_proj and v_proj.
        base_params:       Total parameters in the base model.
        bytes_per_param:   2 for fp16.

    Returns:
        {
          "trainable_params": int,
          "base_params": int,
          "trainable_percent": float,     as a percentage, not a fraction
          "adapter_bytes": int,
          "adapter_mb": float,
        }

    Examples:
        >>> s = lora_config_summary(4096, 32, 8)
        >>> s["trainable_params"]
        4194304
        >>> round(s["adapter_mb"], 1)
        8.4
    """
    # TODO:
    #   trainable = lora_parameter_count(d_model, d_model, rank)
    #               * targets_per_layer * n_layers
    #   adapter_bytes = trainable * bytes_per_param
    #   adapter_mb    = adapter_bytes / 1e6      (megabytes, decimal)
    #   trainable_percent = 100 * trainable / base_params
    return {"trainable_params": 0, "base_params": base_params,
            "trainable_percent": 0.0, "adapter_bytes": 0, "adapter_mb": 0.0}


# ======================================================================
# TASK 6 - break_even_volume
# Module 12, section 12.11
# ======================================================================

def break_even_volume(base_prompt_tokens: int, tuned_prompt_tokens: int,
                      base_price_per_million: float,
                      tuned_price_per_million: float,
                      finetuning_cost: float) -> dict:
    """When does fine-tuning to save cost actually pay back?

    The saving comes from a SHORTER PROMPT and often a smaller model. But a
    fine-tuned model frequently costs MORE per token, so the two effects can
    cancel - which is exactly why you run this before starting.

    Args:
        base_prompt_tokens:      Prompt length with the base model (long,
                                 because it carries instructions and examples).
        tuned_prompt_tokens:     Prompt length with the fine-tuned model.
        base_price_per_million:  Base model input price.
        tuned_price_per_million: Fine-tuned model input price.
        finetuning_cost:         One-off cost of producing the fine-tune.

    Returns:
        {
          "base_cost_per_request": float,
          "tuned_cost_per_request": float,
          "saving_per_request": float,    NEGATIVE if tuning costs more
          "break_even_requests": int or None,   None if it never pays back
          "pays_back": bool,
        }

    Examples:
        >>> r = break_even_volume(2000, 50, 0.15, 0.30, 100.0)
        >>> r["pays_back"]
        True
    """
    # TODO:
    #   1. cost = (tokens / 1_000_000) * price, for each side.
    #   2. saving = base_cost - tuned_cost.
    #   3. If saving <= 0: break_even_requests is None and pays_back is False -
    #      no volume ever recovers a per-request LOSS.
    #   4. Otherwise break_even_requests = ceil(finetuning_cost / saving).
    #      Use math.ceil, and int() the result.
    #   5. Return the dict.
    return {"base_cost_per_request": 0.0, "tuned_cost_per_request": 0.0,
            "saving_per_request": 0.0, "break_even_requests": None,
            "pays_back": False}


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _example(user: str, assistant: str, system: str = None) -> dict:
    """Build a well-formed training example."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def _run_self_test() -> int:
    checks = []

    def check(name, got, expected):
        checks.append((name, got, expected))

    def check_raises(name, fn, exception=ValueError):
        try:
            fn()
            checks.append((name, "did not raise", f"raised {exception.__name__}"))
        except exception:
            checks.append((name, f"raised {exception.__name__}",
                           f"raised {exception.__name__}"))
        except Exception as exc:
            checks.append((name, f"raised {type(exc).__name__}",
                           f"raised {exception.__name__}"))

    # --- TASK 1 ---
    check("1. untried prompting always wins",
          recommend_approach({"tried_prompting": False,
                              "needs_house_style": True,
                              "n_examples": 5000})["recommendation"],
          "prompting")
    check("1. citations -> rag",
          recommend_approach({"tried_prompting": True,
                              "needs_citations": True})["recommendation"], "rag")
    check("1. private knowledge -> rag",
          recommend_approach({"tried_prompting": True,
                              "needs_private_knowledge": True})["recommendation"],
          "rag")
    check("1. house style with enough data -> fine-tuning",
          recommend_approach({"tried_prompting": True, "needs_house_style": True,
                              "n_examples": 500})["recommendation"],
          "fine-tuning")
    check("1. both needs -> rag + fine-tuning",
          recommend_approach({"tried_prompting": True, "needs_citations": True,
                              "needs_house_style": True,
                              "n_examples": 500})["recommendation"],
          "rag + fine-tuning")
    check("1. too few examples BLOCKS fine-tuning",
          recommend_approach({"tried_prompting": True, "needs_house_style": True,
                              "n_examples": 10})["recommendation"],
          "prompting")
    check("1. and records the blocker",
          len(recommend_approach({"tried_prompting": True,
                                  "needs_house_style": True,
                                  "n_examples": 10})["blockers"]), 1)
    check("1. blocked fine-tuning still falls back to rag when rag applies",
          recommend_approach({"tried_prompting": True, "needs_citations": True,
                              "needs_house_style": True,
                              "n_examples": 10})["recommendation"], "rag")
    check("1. no special needs -> prompting",
          recommend_approach({"tried_prompting": True,
                              "n_examples": 500})["recommendation"], "prompting")
    check("1. reasons are populated",
          len(recommend_approach({"tried_prompting": True,
                                  "needs_citations": True})["reasons"]) >= 1, True)

    # --- TASK 2 ---
    check("2. a well-formed example is valid",
          validate_training_example(_example("hi", "hello")), (True, []))
    check("2. a system message first is valid",
          validate_training_example(_example("hi", "hello", "Be brief."))[0], True)

    for label, example, expected_count in [
        ("empty messages", {"messages": []}, 1),
        ("missing messages", {}, 1),
        ("messages not a list", {"messages": "nope"}, 1),
    ]:
        ok, problems = validate_training_example(example)
        check(f"2. rejects: {label}", (ok, len(problems)), (False, expected_count))

    ok, problems = validate_training_example(
        {"messages": [{"role": "user", "content": "hi"}]})
    check("2. rejects an example not ending with assistant",
          (ok, len(problems)), (False, 1))

    ok, problems = validate_training_example({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "wizard", "content": "hello"}]})
    # invalid role AND does not end with assistant
    check("2. rejects an invalid role", (ok, len(problems)), (False, 2))

    ok, problems = validate_training_example({"messages": [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "hello"}]})
    check("2. rejects whitespace-only content", (ok, len(problems)), (False, 1))

    ok, problems = validate_training_example({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
        {"role": "assistant", "content": "hello"}]})
    check("2. rejects a system message that is not first",
          (ok, len(problems)), (False, 1))

    ok, problems = validate_training_example({"messages": [
        "not a dict", {"role": "assistant", "content": "hello"}]})
    check("2. rejects a non-object message", (ok, len(problems)), (False, 1))

    # --- TASK 3 ---
    good = [_example(f"question {i}", "billing") for i in range(60)]
    result = validate_dataset(good)
    check("3. counts total and valid", (result["total"], result["valid"]), (60, 60))
    check("3. no warnings on a clean dataset", result["warnings"], [])

    small = [_example(f"q{i}", "billing") for i in range(10)]
    check("3. warns on too few examples",
          any("50" in w or "few" in w.lower()
              for w in validate_dataset(small)["warnings"]), True)

    with_dupes = good + [good[0], good[0]]
    result = validate_dataset(with_dupes)
    check("3. counts duplicates", result["duplicates"], 2)
    check("3. warns on duplicates",
          any("duplicat" in w.lower() for w in result["warnings"]), True)

    with_invalid = good + [{"messages": []}]
    result = validate_dataset(with_invalid)
    check("3. counts invalid examples", result["valid"], 60)
    check("3. records invalid indices", result["invalid_indices"], [60])

    imbalanced = ([_example(f"q{i}", "billing") for i in range(100)]
                  + [_example(f"r{i}", "technical") for i in range(5)])
    result = validate_dataset(
        imbalanced, label_fn=lambda e: e["messages"][-1]["content"])
    check("3. counts labels", result["label_counts"],
          {"billing": 100, "technical": 5})
    check("3. warns on class imbalance",
          any("imbalance" in w.lower() or "balance" in w.lower()
              for w in result["warnings"]), True)

    balanced = ([_example(f"q{i}", "billing") for i in range(30)]
                + [_example(f"r{i}", "technical") for i in range(30)])
    check("3. no imbalance warning when balanced",
          any("imbalance" in w.lower()
              for w in validate_dataset(
                  balanced,
                  label_fn=lambda e: e["messages"][-1]["content"])["warnings"]),
          False)

    # --- TASK 4 ---
    data = [{"y": "a"} for _ in range(10)] + [{"y": "b"} for _ in range(10)]
    train, validation = stratified_split(data, lambda e: e["y"], 0.2)
    check("4. split sizes", (len(train), len(validation)), (16, 4))
    check("4. nothing is lost or duplicated", len(train) + len(validation), 20)
    check("4. validation preserves the label ratio",
          sorted(Counter(e["y"] for e in validation).items()),
          [("a", 2), ("b", 2)])

    imbalanced_data = ([{"y": "common"} for _ in range(90)]
                       + [{"y": "rare"} for _ in range(10)])
    _, validation = stratified_split(imbalanced_data, lambda e: e["y"], 0.2)
    check("4. a rare class still appears in validation",
          Counter(e["y"] for e in validation)["rare"], 2)

    train_a, validation_a = stratified_split(data, lambda e: e["y"], 0.2, seed=1)
    train_b, validation_b = stratified_split(data, lambda e: e["y"], 0.2, seed=1)
    check("4. the same seed gives the same split",
          validation_a == validation_b, True)

    check_raises("4. rejects a validation fraction of 0",
                 lambda: stratified_split(data, lambda e: e["y"], 0.0))
    check_raises("4. rejects a validation fraction of 1",
                 lambda: stratified_split(data, lambda e: e["y"], 1.0))

    # --- TASK 5 ---
    check("5. lora params 4096x4096 rank 8",
          lora_parameter_count(4096, 4096, 8), 65_536)
    check("5. lora params 768x768 rank 8",
          lora_parameter_count(768, 768, 8), 12_288)
    check("5. lora params scale linearly with rank",
          lora_parameter_count(4096, 4096, 16),
          2 * lora_parameter_count(4096, 4096, 8))
    check("5. full params 4096x4096", full_parameter_count(4096, 4096), 16_777_216)
    check("5. lora is under 0.4% of full at rank 8",
          round(100 * lora_parameter_count(4096, 4096, 8)
                / full_parameter_count(4096, 4096), 3), 0.391)
    check("5. lora saves MORE on a bigger matrix",
          (lora_parameter_count(8192, 8192, 8) / full_parameter_count(8192, 8192))
          < (lora_parameter_count(768, 768, 8) / full_parameter_count(768, 768)),
          True)

    summary = lora_config_summary(4096, 32, 8)
    check("5. 7B config trainable params", summary["trainable_params"], 4_194_304)
    check("5. 7B config trainable percent",
          round(summary["trainable_percent"], 4), 0.0599)
    check("5. 7B config adapter size in MB", round(summary["adapter_mb"], 1), 8.4)
    check("5. more targets means more params",
          lora_config_summary(4096, 32, 8, targets_per_layer=4)["trainable_params"],
          2 * summary["trainable_params"])

    # --- TASK 6 ---
    result = break_even_volume(2000, 50, 0.15, 0.30, 100.0)
    check("6. base cost per request",
          round(result["base_cost_per_request"], 8), 0.0003)
    check("6. tuned cost per request",
          round(result["tuned_cost_per_request"], 8), 0.000015)
    check("6. saving per request",
          round(result["saving_per_request"], 8), 0.000285)
    check("6. break-even volume", result["break_even_requests"], 350_878)
    check("6. pays back", result["pays_back"], True)

    # A fine-tune that costs MORE per request never pays back.
    result = break_even_volume(100, 100, 0.15, 0.60, 100.0)
    check("6. a per-request loss never pays back",
          (result["pays_back"], result["break_even_requests"]), (False, None))
    check("6. and reports a negative saving",
          result["saving_per_request"] < 0, True)

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 12 SELF-TEST - decide, then prepare")
    print("=" * 76)
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
    print("-" * 76)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  You can now answer the prompt-vs-RAG-vs-fine-tune question with")
        print("  a procedure rather than an instinct - and you have the dataset")
        print("  checks that catch the failures which train perfectly happily.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1, 2, 3 (uses 2), 4, 5, 6.")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_decision_framework():
    print("=" * 76)
    print("  EXPERIMENT 1: the decision framework on real scenarios")
    print("=" * 76)
    print()

    scenarios = [
        ("Support bot over our help centre",
         {"tried_prompting": True, "needs_private_knowledge": True,
          "needs_citations": True, "facts_change_often": True, "n_examples": 0}),
        ("Match our brand's writing voice",
         {"tried_prompting": True, "needs_house_style": True, "n_examples": 800}),
        ("Same, but we only have 12 examples",
         {"tried_prompting": True, "needs_house_style": True, "n_examples": 12}),
        ("Classify 2M tickets/month, cost is critical",
         {"tried_prompting": True, "high_volume_cost_matters": True,
          "n_examples": 3000}),
        ("Cite our policies AND sound like us",
         {"tried_prompting": True, "needs_citations": True,
          "needs_house_style": True, "n_examples": 600}),
        ("We think we need fine-tuning (prompt untried)",
         {"tried_prompting": False, "needs_specialist_task": True,
          "n_examples": 5000}),
    ]

    for label, requirements in scenarios:
        result = recommend_approach(requirements)
        if not result["reasons"] and result["recommendation"] == "prompting" \
                and requirements.get("tried_prompting"):
            pass
        print(f"  {label}")
        print(f"    -> {result['recommendation'].upper()}")
        for reason in result["reasons"]:
            print(f"       because: {reason}")
        for blocker in result["blockers"]:
            print(f"       BLOCKED: {blocker}")
        print()

    print("  The last scenario is the one worth dwelling on. Five thousand")
    print("  examples and a genuinely specialist task - and the answer is still")
    print("  'try a prompt first', because nobody has.")
    print()
    print("  That ordering is not pedantry. A large share of 'we need to")
    print("  fine-tune' turns out to be 'our prompt was vague', and finding")
    print("  that out costs an afternoon instead of a month.")
    print()


def experiment_lora_scaling():
    print("=" * 76)
    print("  EXPERIMENT 2: why LoRA gets BETTER on bigger models")
    print("=" * 76)
    print()

    print(f"  {'matrix':>14}{'rank':>6}{'full params':>15}{'LoRA params':>14}"
          f"{'% of full':>11}")
    print("  " + "-" * 60)

    for d in [768, 1024, 4096, 8192]:
        full = full_parameter_count(d, d)
        lora = lora_parameter_count(d, d, 8)
        if full == 0:
            print("  not implemented yet.")
            print()
            return
        print(f"  {f'{d}x{d}':>14}{8:>6}{full:>15,}{lora:>14,}"
              f"{100 * lora / full:>10.3f}%")

    print()
    print("  The percentage FALLS as the model grows: 2.08% at 768, 0.20% at")
    print("  8192. Full fine-tuning costs scale with d squared; LoRA scales")
    print("  with d. So every increase in model size makes LoRA a better deal.")
    print()

    print("  Realistic whole-model configurations:")
    print()
    print(f"  {'model':>10}{'layers':>8}{'rank':>6}{'trainable':>14}"
          f"{'% of base':>12}{'adapter':>11}")
    print("  " + "-" * 61)

    configurations = [
        ("1.5B", 1536, 28, 8, 1_500_000_000),
        ("7B", 4096, 32, 8, 7_000_000_000),
        ("7B", 4096, 32, 32, 7_000_000_000),
        ("13B", 5120, 40, 8, 13_000_000_000),
        ("70B", 8192, 80, 8, 70_000_000_000),
    ]

    for name, d_model, layers, rank, base in configurations:
        summary = lora_config_summary(d_model, layers, rank, base_params=base)
        print(f"  {name:>10}{layers:>8}{rank:>6}"
              f"{summary['trainable_params']:>14,}"
              f"{summary['trainable_percent']:>11.4f}%"
              f"{summary['adapter_mb']:>10.1f}M")

    print()
    print("  A 70B model - roughly 140 GB of weights - is specialised by a")
    print("  42 MB file. You can put that in Git.")
    print()
    print("  Compare the two 7B rows. Quadrupling the rank quadruples the")
    print("  trainable parameters, and it is still under a quarter of one")
    print("  percent of the base. Rank is cheap; the reason not to raise it")
    print("  is overfitting, not size.")
    print()


def experiment_dataset_problems():
    print("=" * 76)
    print("  EXPERIMENT 3: dataset failures that train perfectly happily")
    print("=" * 76)
    print()

    label_of = lambda e: e["messages"][-1]["content"]

    datasets = {
        "clean, balanced, 60 examples":
            [_example(f"ticket {i}", "billing" if i % 2 else "technical")
             for i in range(60)],

        "only 20 examples":
            [_example(f"ticket {i}", "billing" if i % 2 else "technical")
             for i in range(20)],

        "with 5 exact duplicates":
            [_example(f"ticket {i}", "billing" if i % 2 else "technical")
             for i in range(60)]
            + [_example("ticket 0", "technical")] * 5,

        "90/10 class imbalance":
            [_example(f"a{i}", "billing") for i in range(90)]
            + [_example(f"b{i}", "technical") for i in range(9)],

        "3 malformed examples":
            [_example(f"ticket {i}", "billing" if i % 2 else "technical")
             for i in range(60)]
            + [{"messages": []},
               {"messages": [{"role": "user", "content": "no answer"}]},
               {"messages": [{"role": "user", "content": "x"},
                             {"role": "wizard", "content": "y"}]}],
    }

    for label, examples in datasets.items():
        result = validate_dataset(examples, label_fn=label_of)
        if result["total"] == 0:
            print("  validate_dataset not implemented yet.")
            print()
            return
        status = "OK" if not result["warnings"] else "PROBLEMS"
        print(f"  [{status:>8}] {label}")
        print(f"             {result['valid']}/{result['total']} valid, "
              f"{result['duplicates']} duplicates, "
              f"labels {dict(result['label_counts'])}")
        for warning in result["warnings"]:
            print(f"             - {warning}")
        print()

    print("  Every one of these datasets TRAINS. The loss goes down, the run")
    print("  completes, and you get an adapter file at the end.")
    print()
    print("  The 90/10 case is the quietest failure: the model learns to")
    print("  predict 'billing' almost always, scores 90% accuracy on a matching")
    print("  test set, and is useless for the class you actually cared about.")
    print()
    print("  This is Module 11's lesson arriving early - accuracy on imbalanced")
    print("  data measures how common the majority class is.")
    print()


def experiment_break_even():
    print("=" * 76)
    print("  EXPERIMENT 4: does fine-tuning for cost actually pay back?")
    print("=" * 76)
    print()
    print("  A classifier. The base model needs a long prompt with instructions")
    print("  and few-shot examples; the fine-tuned model needs almost none.")
    print()

    scenarios = [
        ("2000 -> 50 tokens, tuned costs 2x", 2000, 50, 0.15, 0.30, 100.0),
        ("2000 -> 50 tokens, tuned costs 8x", 2000, 50, 0.15, 1.20, 100.0),
        ("400 -> 50 tokens, tuned costs 2x", 400, 50, 0.15, 0.30, 100.0),
        ("no prompt saving, tuned costs 2x", 500, 500, 0.15, 0.30, 100.0),
        ("2000 -> 50, but a 5000 dollar fine-tune", 2000, 50, 0.15, 0.30, 5000.0),
    ]

    print(f"  {'scenario':<40}{'saving/req':>13}{'break-even':>16}")
    print("  " + "-" * 69)

    for label, base_tokens, tuned_tokens, base_price, tuned_price, cost in scenarios:
        result = break_even_volume(base_tokens, tuned_tokens, base_price,
                                   tuned_price, cost)
        if result["break_even_requests"] is None:
            volume = "NEVER"
        else:
            volume = f"{result['break_even_requests']:,} req"
        print(f"  {label:<40}${result['saving_per_request']:>12.8f}{volume:>16}")

    print()
    print("  Compare rows 1 and 2. Making the tuned model EIGHT TIMES more")
    print("  expensive per token only moves break-even from 351k to 417k - an")
    print("  18% penalty. That is the surprise: when the prompt shrinks 40x,")
    print("  the per-token price barely matters.")
    print()
    print("  Row 3 shows what DOES matter. A smaller prompt reduction, 400 to")
    print("  50 instead of 2000 to 50, needs SIX TIMES the volume. The saving")
    print("  is driven almost entirely by how many tokens you stop sending.")
    print()
    print("  Row 4 is the case with no cost argument at all: a fine-tune that")
    print("  does not shorten the prompt costs more per request forever, and")
    print("  no volume recovers a per-request loss.")
    print()
    print("  So the question to ask is not 'is the tuned model cheaper per")
    print("  token?' - it usually is not. It is 'how much prompt can I delete?'")
    print()
    print("  Run this arithmetic BEFORE you start. If break-even is 350,000")
    print("  requests and you serve 10,000 a month, that is three years to")
    print("  recover a hundred dollars.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_decision_framework()
        experiment_lora_scaling()
        experiment_dataset_problems()
        experiment_break_even()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
