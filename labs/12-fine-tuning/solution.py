"""
solution.py - Lab 12 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import json
import math
import random
from collections import Counter, defaultdict


# ======================================================================
# TASK 1 - recommend_approach
# ======================================================================

RAG_SIGNALS = [
    ("needs_private_knowledge", "must know documents outside its training data"),
    ("needs_current_info", "must know things after the training cutoff"),
    ("needs_citations", "must attribute claims to a source"),
    ("facts_change_often", "the knowledge base updates regularly"),
]

FINETUNE_SIGNALS = [
    ("needs_house_style", "must match a specific tone or voice"),
    ("needs_specialist_task", "a task the base model cannot do at all"),
    ("high_volume_cost_matters", "cost per request is a binding constraint"),
]

MIN_EXAMPLES = 50


def recommend_approach(requirements: dict) -> dict:
    """Recommend prompting, RAG or fine-tuning from stated requirements."""
    # RULE 1, and it outranks everything. A large share of "we need to
    # fine-tune" turns out to be "our prompt was vague", and finding that out
    # costs an afternoon rather than a month.
    if not requirements.get("tried_prompting"):
        return {
            "recommendation": "prompting",
            "reasons": ["prompting has not been tried yet - always start here"],
            "blockers": [],
        }

    reasons = [why for key, why in RAG_SIGNALS if requirements.get(key)]
    tune_reasons = [why for key, why in FINETUNE_SIGNALS if requirements.get(key)]

    wants_rag = bool(reasons)
    wants_tune = bool(tune_reasons)

    blockers = []
    n_examples = requirements.get("n_examples", 0)
    if wants_tune and n_examples < MIN_EXAMPLES:
        blockers.append(
            f"only {n_examples} examples; fine-tuning needs at least "
            f"{MIN_EXAMPLES}. Use few-shot prompting instead.")
        wants_tune = False

    reasons.extend(tune_reasons)

    if wants_rag and wants_tune:
        recommendation = "rag + fine-tuning"
    elif wants_rag:
        recommendation = "rag"
    elif wants_tune:
        recommendation = "fine-tuning"
    else:
        recommendation = "prompting"

    return {"recommendation": recommendation, "reasons": reasons,
            "blockers": blockers}


# ======================================================================
# TASK 2 - validate_training_example
# ======================================================================

VALID_ROLES = {"system", "user", "assistant"}


def validate_training_example(example: dict) -> tuple:
    """Check one training example against the chat-format rules."""
    messages = example.get("messages")

    # Fail fast: every check below assumes a non-empty list of messages.
    if not isinstance(messages, list) or not messages:
        return (False, ["missing or empty 'messages' list"])

    problems = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"message {index} is not an object")
            continue

        if message.get("role") not in VALID_ROLES:
            problems.append(
                f"message {index}: invalid role {message.get('role')!r}")

        # str(...) so a non-string content does not raise on .strip().
        if not str(message.get("content", "")).strip():
            problems.append(f"message {index}: empty content")

    # The LAST message is the training target. Without an assistant turn
    # there is literally nothing for the model to learn to produce.
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") != "assistant":
        problems.append("last message must have role 'assistant'")

    # A system message may only appear first. One buried mid-conversation
    # teaches the model a message ordering it will never see in production.
    for index, message in enumerate(messages[1:], start=1):
        if isinstance(message, dict) and message.get("role") == "system":
            problems.append(f"message {index}: system message must come first")

    return (not problems, problems)


# ======================================================================
# TASK 3 - validate_dataset
# ======================================================================

def validate_dataset(examples: list, label_fn=None) -> dict:
    """Check a whole dataset for the failures that silently ruin a fine-tune."""
    total = len(examples)

    invalid_indices = []
    for index, example in enumerate(examples):
        is_valid, _ = validate_training_example(example)
        if not is_valid:
            invalid_indices.append(index)

    valid = total - len(invalid_indices)

    # Exact duplicates. sort_keys makes the serialisation canonical, so two
    # dicts differing only in key order still compare equal.
    serialised = [json.dumps(example, sort_keys=True, default=str)
                  for example in examples]
    duplicates = total - len(set(serialised))

    label_counts = {}
    if label_fn is not None:
        counter = Counter()
        for example in examples:
            try:
                counter[label_fn(example)] += 1
            except Exception:  # noqa: BLE001 - a malformed example has no label
                continue
        label_counts = dict(counter)

    warnings = []

    if total < MIN_EXAMPLES:
        warnings.append(
            f"only {total} examples; fewer than {MIN_EXAMPLES} is usually "
            f"better served by few-shot prompting")

    if invalid_indices:
        warnings.append(
            f"{len(invalid_indices)} invalid example(s) at indices "
            f"{invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}")

    if duplicates:
        warnings.append(
            f"{duplicates} exact duplicate(s); duplicates over-weight whatever "
            f"they contain")

    # Imbalance. The model learns to predict the majority class, scores well
    # on a matching test set, and is useless for the class you cared about.
    if len(label_counts) >= 2:
        largest = max(label_counts.values())
        smallest = min(label_counts.values())
        if largest > 10 * smallest:
            warnings.append(
                f"class imbalance: largest class {largest}, smallest {smallest} "
                f"({largest / smallest:.0f}x)")

    return {"total": total, "valid": valid, "invalid_indices": invalid_indices,
            "duplicates": duplicates, "warnings": warnings,
            "label_counts": label_counts}


# ======================================================================
# TASK 4 - stratified_split
# ======================================================================

def stratified_split(examples: list, label_fn, validation_fraction: float = 0.2,
                     seed: int = 42) -> tuple:
    """Split into train/validation while preserving the label distribution."""
    if not 0 < validation_fraction < 1:
        raise ValueError(
            f"validation_fraction must be between 0 and 1 exclusive, "
            f"got {validation_fraction}")

    by_label = defaultdict(list)
    for example in examples:
        by_label[label_fn(example)].append(example)

    # A LOCAL generator: seeding the global random module would disturb any
    # other randomness the caller depends on.
    rng = random.Random(seed)

    train, validation = [], []

    # Sort by str(label) so the iteration order is stable and the split is
    # reproducible regardless of insertion order.
    for label in sorted(by_label, key=str):
        shuffled = list(by_label[label])
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * validation_fraction)
        validation.extend(shuffled[:cut])
        train.extend(shuffled[cut:])

    return (train, validation)


# ======================================================================
# TASK 5 - LoRA parameter arithmetic
# ======================================================================

def lora_parameter_count(d_in: int, d_out: int, rank: int) -> int:
    """Trainable parameters LoRA adds for ONE weight matrix."""
    # B is (d_in x rank), A is (rank x d_out). Note this grows LINEARLY with
    # d, while a full fine-tune grows with d^2.
    return rank * (d_in + d_out)


def full_parameter_count(d_in: int, d_out: int) -> int:
    """Parameters a FULL fine-tune would update for the same matrix."""
    return d_in * d_out


def lora_config_summary(d_model: int, n_layers: int, rank: int,
                        targets_per_layer: int = 2,
                        base_params: int = 7_000_000_000,
                        bytes_per_param: int = 2) -> dict:
    """Summarise a whole LoRA configuration."""
    per_matrix = lora_parameter_count(d_model, d_model, rank)
    trainable = per_matrix * targets_per_layer * n_layers

    adapter_bytes = trainable * bytes_per_param

    return {
        "trainable_params": trainable,
        "base_params": base_params,
        "trainable_percent": 100 * trainable / base_params if base_params else 0.0,
        "adapter_bytes": adapter_bytes,
        "adapter_mb": adapter_bytes / 1e6,
    }


# ======================================================================
# TASK 6 - break_even_volume
# ======================================================================

def break_even_volume(base_prompt_tokens: int, tuned_prompt_tokens: int,
                      base_price_per_million: float,
                      tuned_price_per_million: float,
                      finetuning_cost: float) -> dict:
    """When does fine-tuning to save cost actually pay back?"""
    base_cost = (base_prompt_tokens / 1_000_000) * base_price_per_million
    tuned_cost = (tuned_prompt_tokens / 1_000_000) * tuned_price_per_million

    saving = base_cost - tuned_cost

    if saving <= 0:
        # No volume recovers a per-request LOSS. This happens more often than
        # people expect, because fine-tuned models frequently cost more per
        # token - so a shorter prompt can be entirely cancelled out.
        return {"base_cost_per_request": base_cost,
                "tuned_cost_per_request": tuned_cost,
                "saving_per_request": saving,
                "break_even_requests": None,
                "pays_back": False}

    return {"base_cost_per_request": base_cost,
            "tuned_cost_per_request": tuned_cost,
            "saving_per_request": saving,
            "break_even_requests": int(math.ceil(finetuning_cost / saving)),
            "pays_back": True}


# ======================================================================
# BONUS - things the tasks do not cover
# ======================================================================

def detect_length_correlation(examples: list, label_fn,
                              threshold: float = 1.5) -> list:
    """Warn when input length correlates with the label.

    The spurious-correlation failure from Module 12, section 12.6 - and from
    Lab 1's Teachable Machine stretch, one layer up the stack. If every
    "approve" example is long and every "reject" is short, the model learns
    LENGTH, scores beautifully on your test set, and fails in production.

    Returns:
        A list of warning strings. Empty means no strong correlation found.
    """
    lengths_by_label = defaultdict(list)

    for example in examples:
        try:
            user_text = " ".join(
                message.get("content", "")
                for message in example["messages"]
                if message.get("role") == "user")
            lengths_by_label[label_fn(example)].append(len(user_text))
        except Exception:  # noqa: BLE001
            continue

    if len(lengths_by_label) < 2:
        return []

    means = {label: sum(lengths) / len(lengths)
             for label, lengths in lengths_by_label.items() if lengths}
    if len(means) < 2:
        return []

    longest_label = max(means, key=means.get)
    shortest_label = min(means, key=means.get)

    if means[shortest_label] == 0:
        return []

    ratio = means[longest_label] / means[shortest_label]
    if ratio < threshold:
        return []

    return [
        f"input length correlates with label: {longest_label!r} averages "
        f"{means[longest_label]:.0f} chars vs {shortest_label!r} at "
        f"{means[shortest_label]:.0f} ({ratio:.1f}x). The model may learn "
        f"LENGTH rather than the task."
    ]


def check_split_leakage(train: list, validation: list) -> list:
    """Detect examples appearing in BOTH train and validation.

    Leakage makes your validation score meaningless in the most flattering
    possible direction - the model has already seen the answers.
    """
    train_keys = {json.dumps(e, sort_keys=True, default=str) for e in train}
    validation_keys = {json.dumps(e, sort_keys=True, default=str)
                       for e in validation}

    overlap = train_keys & validation_keys
    if not overlap:
        return []

    return [f"{len(overlap)} example(s) appear in BOTH train and validation; "
            f"validation scores will be inflated"]


def to_jsonl(examples: list, path: str) -> int:
    """Write examples as JSONL - one JSON object per line.

    Returns the number of lines written.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


# ======================================================================
# Demonstrations
# ======================================================================

def _example(user: str, assistant: str, system: str = None) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def demo_memory_arithmetic():
    print("=" * 76)
    print("  WHY FULL FINE-TUNING NEEDS A CLUSTER AND LoRA DOES NOT")
    print("=" * 76)
    print()
    print("  Training memory is roughly:")
    print("    weights + gradients + optimiser state (2 values per param, Adam)")
    print("    ~= 16 bytes per TRAINABLE parameter, at fp32 optimiser state.")
    print()

    print(f"  {'model':>8}{'full fine-tune':>18}{'LoRA r=8':>14}"
          f"{'QLoRA r=8':>14}")
    print("  " + "-" * 54)

    models = [
        ("1.5B", 1_500_000_000, 1536, 28),
        ("7B", 7_000_000_000, 4096, 32),
        ("13B", 13_000_000_000, 5120, 40),
        ("70B", 70_000_000_000, 8192, 80),
    ]

    for name, base, d_model, layers in models:
        # Full: every parameter is trainable.
        full_gb = base * 16 / 1e9

        summary = lora_config_summary(d_model, layers, 8, base_params=base)
        trainable = summary["trainable_params"]

        # LoRA: frozen base at fp16, plus optimiser state for the adapter only.
        lora_gb = (base * 2 + trainable * 16) / 1e9
        # QLoRA: frozen base at 4 bits.
        qlora_gb = (base * 0.5 + trainable * 16) / 1e9

        print(f"  {name:>8}{f'{full_gb:,.0f} GB':>18}{f'{lora_gb:,.1f} GB':>14}"
              f"{f'{qlora_gb:,.1f} GB':>14}")

    print()
    print("  These are rough figures - activations and gradient checkpointing")
    print("  shift them - but the ORDERS OF MAGNITUDE are the point.")
    print()
    print("  A 7B full fine-tune wants ~112 GB, so multiple high-end GPUs.")
    print("  The same model with QLoRA wants about 3.6 GB, which is a free")
    print("  Colab session.")
    print()
    print("  And note WHERE the saving comes from. LoRA barely reduces the")
    print("  weights - it removes the gradients and optimiser state, which is")
    print("  most of the cost. QLoRA then shrinks the weights too.")
    print()


def demo_spurious_correlation():
    print("=" * 76)
    print("  THE PATTERN YOUR DATASET TEACHES BY ACCIDENT")
    print("=" * 76)
    print()

    label_of = lambda e: e["messages"][-1]["content"]

    # A dataset where the LABEL correlates with input length. Entirely
    # plausible: detailed complaints get escalated, terse ones do not.
    biased = (
        [_example("I am extremely unhappy with the service I received last "
                  "Tuesday and would like this escalated to a manager please",
                  "escalate") for _ in range(30)]
        + [_example("thanks", "close") for _ in range(30)]
    )

    balanced = (
        [_example("Escalate this please", "escalate") for _ in range(15)]
        + [_example("I am extremely unhappy with the service I received and "
                    "would like this escalated to a manager", "escalate")
           for _ in range(15)]
        + [_example("thanks", "close") for _ in range(15)]
        + [_example("Thank you very much for resolving this so quickly, I am "
                    "happy with the outcome and have nothing further", "close")
           for _ in range(15)]
    )

    for label, dataset in [("length correlates with label", biased),
                           ("length varied within each label", balanced)]:
        warnings = detect_length_correlation(dataset, label_of)
        status = "PROBLEM" if warnings else "OK"
        print(f"  [{status:>7}] {label}")
        for warning in warnings:
            print(f"            {warning}")
        print()

    print("  Both datasets are perfectly valid. Both train. Both would score")
    print("  well on a test set drawn the same way.")
    print()
    print("  The first has learned to escalate LONG messages. A terse but")
    print("  furious 'This is unacceptable. Manager. Now.' gets closed.")
    print()
    print("  This is the same failure as Lab 1's Teachable Machine stretch -")
    print("  the model learns whatever most easily predicts the label, and it")
    print("  cannot know what you meant.")
    print()


def demo_the_checklist():
    print("=" * 76)
    print("  THE HONEST CHECKLIST")
    print("=" * 76)
    print()

    projects = [
        ("Team A - two weeks in", {
            "tried a good prompt and measured it": False,
            "tried RAG, or it is not a facts problem": False,
            "has an evaluation set and a baseline number": False,
            "has 50+ high-quality examples": True,
            "can state 'better' as a number": False,
            "has done the break-even arithmetic": False,
        }),
        ("Team B - ready to start", {
            "tried a good prompt and measured it": True,
            "tried RAG, or it is not a facts problem": True,
            "has an evaluation set and a baseline number": True,
            "has 50+ high-quality examples": True,
            "can state 'better' as a number": True,
            "has done the break-even arithmetic": True,
        }),
    ]

    for name, checklist in projects:
        unticked = [item for item, done in checklist.items() if not done]
        print(f"  {name}")
        for item, done in checklist.items():
            print(f"    [{'x' if done else ' '}] {item}")
        if unticked:
            print(f"    -> NOT READY. Next task: {unticked[0]}")
        else:
            print("    -> ready to fine-tune")
        print()

    print("  Team A has the examples, which is the part that feels like")
    print("  progress. They have none of the things that would tell them")
    print("  whether the fine-tune helped.")
    print()
    print("  If any box is unticked, that box is your next task - not")
    print("  fine-tuning.")
    print()


if __name__ == "__main__":
    demo_memory_arithmetic()
    demo_spurious_correlation()
    demo_the_checklist()
