# 🧪 Lab 12: Decide, Then Prepare

**Module:** [12 — Fine-Tuning & Model Customization](../../modules/12-fine-tuning.md)

---

## Objective

Most of fine-tuning is deciding whether to, and preparing data. This lab is both — and only optionally the training itself.

By the end you will have:

1. **Encoded the decision framework as code** — prompt, RAG or fine-tune, with the ordering enforced
2. **Built a dataset validator** that catches the failures which train perfectly happily
3. **Computed LoRA parameter counts** and discovered why bigger models are a better deal
4. **Run the break-even arithmetic** and found what actually drives it
5. *(Optional)* **Fine-tuned a real model** with QLoRA on a free Colab GPU

## Expected outcome

`python starter.py` reports **54 of 54 checks passing**, then four experiments — including a table showing a 70B model specialised by a 42 MB file.

## Requirements

| | |
|---|---|
| **Part 1** | **Standard library only.** No packages, no GPU, no API key. |
| **Part 2** | Optional. A free Google Colab account. |
| **Cost** | **Free**, including the optional real fine-tune |
| **Time** | ~70 minutes (+60 for the optional training run) |

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 6 tasks, 54-check self-test, 4 experiments. |
| `solution.py` | Reference solution + length-correlation and leakage detectors + 3 demos. |
| `SOLUTION.md` | The reasoning, including what the break-even numbers really say. |

---

## Part 1 — Decide and prepare (50 min)

```powershell
python labs/12-fine-tuning/starter.py
```

| Task | Function | Key idea | Module 12 § |
|---|---|---|---|
| 1 | `recommend_approach` | The decision procedure, as code | 12.1 |
| 2 | `validate_training_example` | Chat-format rules | 12.7 |
| 3 | `validate_dataset` | **Dataset-level failures** | 12.7 |
| 4 | `stratified_split` | Reproducible, balance-preserving splits | 12.7 |
| 5 | `lora_parameter_count` + summary | The arithmetic behind an 8 MB adapter | 12.4 |
| 6 | `break_even_volume` | Does the cost case hold? | 12.11 |

### Task 1 — the ordering is the content

There's a test that looks almost unfair:

```
[ OK ]  1. untried prompting always wins
```

Five thousand examples, a genuinely specialist task — and the recommendation is still `"prompting"`, because nobody has tried one.

That's not a quirk of the implementation. **A large share of "we need to fine-tune" turns out to be "our prompt was vague"**, and finding that out costs an afternoon instead of a month. The rule outranks everything else deliberately.

Note also that a fine-tuning *blocker* (too few examples) doesn't fall back to nothing — if RAG also applies, you get RAG. There's a test for it.

### Task 3 — the failures that don't error

Per-example validation isn't enough. These four problems exist only at the dataset level, and **every one of them trains happily**:

| Problem | What it does |
|---|---|
| Too few examples | Overfits immediately |
| Duplicates | Over-weight whatever they contain |
| Class imbalance | Teaches the model to always predict the majority |
| Invalid examples | Silently skipped, or worse, learned from |

The duplicate check uses `json.dumps(example, sort_keys=True)` so two dicts differing only in key order still compare equal.

### Task 4 — two properties, both tested

**Stratified**, so a rare class survives the split:

```
[ OK ]  4. a rare class still appears in validation
```

A random split of 90/10 data can put every rare example on one side, and then its validation score is meaningless.

**Seeded**, so runs are comparable:

```
[ OK ]  4. the same seed gives the same split
```

Comparing two training runs across different splits tells you nothing about the training.

### Task 5 — the arithmetic worth internalising

```python
lora_parameter_count(d_in, d_out, rank) == rank * (d_in + d_out)
```

Linear in `d`, where a full fine-tune is quadratic. There's a test asserting LoRA saves proportionally **more** on a bigger matrix — which is the property that makes the technique scale in the right direction.

**✅ Part 1 complete at `All 54 checks passed.`**

---

## The four experiments

### Experiment 1 — the framework on real scenarios

Six scenarios, including one designed to sting: 5,000 examples, a specialist task, and the answer is still "try a prompt first".

### Experiment 2 — LoRA gets better on bigger models

```
       model  layers  rank     trainable   % of base    adapter
        1.5B      28     8     1,376,256     0.0918%       2.8M
          7B      32     8     4,194,304     0.0599%       8.4M
          7B      32    32    16,777,216     0.2397%      33.6M
         70B      80     8    20,971,520     0.0300%      41.9M
```

**A 70B model — roughly 140 GB of weights — is specialised by a 42 MB file.** And the percentage *falls* as models grow, because full cost scales with `d²` while LoRA scales with `d`.

Compare the two 7B rows: quadrupling the rank still leaves you under a quarter of one percent. **Rank is cheap; the reason not to raise it is overfitting, not size.**

### Experiment 3 — datasets that train happily and fail anyway

Five datasets, four broken. Every one of them completes a training run and produces an adapter.

The 90/10 imbalance is the quietest: the model learns to predict the majority class, scores 90% on a matching test set, and is useless for the class you cared about. **That's Module 11's lesson arriving early** — accuracy on imbalanced data measures how common the majority class is.

### Experiment 4 — what actually drives the cost case

```
  scenario                                   saving/req      break-even
  2000 -> 50 tokens, tuned costs 2x       $  0.00028500     350,878 req
  2000 -> 50 tokens, tuned costs 8x       $  0.00024000     416,667 req
  400 -> 50 tokens, tuned costs 2x        $  0.00004500   2,222,223 req
  no prompt saving, tuned costs 2x        $ -0.00007500           NEVER
```

The surprise is in rows 1 and 2: making the tuned model **eight times more expensive per token** only worsens break-even by 18%. When the prompt shrinks 40×, the per-token price barely matters.

Row 3 shows what does: a smaller prompt reduction needs **six times the volume**.

> **The question to ask is not "is the tuned model cheaper per token?" — it usually isn't. It's "how much prompt can I delete?"**

---

## Part 2 — An actual fine-tune (optional, 60 min)

Free, on Colab's GPU. **Do this only if Part 1 concluded that fine-tuning is warranted** — or purely to see the mechanics.

### Step 1: open a Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com/), new notebook, then **Runtime → Change runtime type → T4 GPU**.

### Step 2: install and load

```python
!pip install -q transformers peft datasets accelerate bitsandbytes trl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # small enough to iterate quickly

# QLoRA: quantize the FROZEN base to 4 bits (Module 12, section 12.5).
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quantization, device_map="auto")
model = prepare_model_for_kbit_training(model)
```

### Step 3: attach LoRA adapters

```python
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

**Compare the printed number against your `lora_config_summary`.** Qwen2.5-0.5B has `d_model=896` and 24 layers — work out what you expect before you look.

### Step 4: build a dataset with your own validator

```python
import sys
# Upload starter.py to Colab, or paste your functions in.
from starter import validate_dataset, stratified_split

CATEGORIES = ["billing", "technical", "account", "other"]

# Build 100+ examples. Vary length and phrasing WITHIN each category
# (Module 12, section 12.6) so the model cannot learn a shortcut.
raw_examples = [
    {"messages": [
        {"role": "system", "content": "Classify the support ticket."},
        {"role": "user", "content": "I was charged twice this month"},
        {"role": "assistant", "content": "billing"},
    ]},
    # ... add many more, across all four categories
]

label_of = lambda e: e["messages"][-1]["content"]

report = validate_dataset(raw_examples, label_fn=label_of)
print(report["warnings"])          # FIX THESE BEFORE TRAINING
assert not report["warnings"], "clean the dataset first"

train, validation = stratified_split(raw_examples, label_of, 0.2)
print(f"{len(train)} train, {len(validation)} validation")
```

### Step 5: train

```python
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

arguments = SFTConfig(
    output_dir="./adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     # not the last epoch - the BEST one
)

trainer = SFTTrainer(
    model=model,
    args=arguments,
    train_dataset=Dataset.from_list(train),
    eval_dataset=Dataset.from_list(validation),
)

trainer.train()
model.save_pretrained("./adapter")
```

### Step 6: evaluate against the baseline that matters

```python
# The comparison people skip (Module 12, section 12.9).
# 1. base model, ZERO-shot        - the floor
# 2. base model, FEW-SHOT prompted - the REAL baseline
# 3. fine-tuned model              - your candidate
#
# Run all three over your validation set and compute accuracy for each.
```

### Then answer these

1. **What did `print_trainable_parameters()` report?** Does it match your `lora_config_summary` for `d_model=896`, 24 layers, rank 8, 2 targets?
2. **How large is `./adapter`?** Compare against the base model's download size.
3. **Did validation loss turn up before training ended?** If so, at which epoch — and did `load_best_model_at_end` save you?
4. **Fine-tuned versus few-shot prompted.** How big is the gap, really?
5. **Ask it something unrelated to classification.** Did general capability survive? (§12.2 — catastrophic forgetting.)
6. **Was it worth it** compared to the few-shot prompt, given the effort?

Question 4 is the honest one, and question 6 is the one that matters. **A common outcome is that a good few-shot prompt is within a couple of points** — which is a genuinely useful finding, not a failed lab.

---

## 🚀 Stretch Challenges

### 1. Add the length-correlation detector

`solution.py` has `detect_length_correlation`. Build your own, then deliberately construct a dataset where "escalate" examples are long and "close" examples are short.

**Train on it if you did Part 2.** Then test with a *terse but furious* message. Does it close the ticket? That's the spurious correlation from Lab 1's Teachable Machine stretch, now in a language model.

### 2. Add leakage detection

`check_split_leakage` in `solution.py` finds examples appearing in both train and validation. Deliberately introduce leakage and watch validation accuracy become meaninglessly high.

### 3. Sweep the rank

If you did Part 2, train at `r=4`, `r=8`, `r=32`. Plot validation loss against rank.

**You will probably find little difference** — which is the point of §12.4's advice to start at 8 and raise only when training loss plateaus too high.

### 4. Measure catastrophic forgetting properly

Build a 20-question general-capability set (arithmetic, general knowledge, summarising). Run it **before and after**.

A fine-tune that improves your task by 5% and loses 20% on general reasoning is usually a bad trade — **and you cannot see that from task metrics at all** (§12.9).

### 5. Do the full comparison from §12.9

All five rows: zero-shot, few-shot, RAG, fine-tuned, fine-tuned + RAG. Use the evaluation harness from Lab 11.

**This is the table that should exist before any fine-tuning decision**, and building it is the single most valuable thing in this lab.

### 6. Write your project's honest checklist

Take §12.12's six-item checklist and fill it in for something you actually want to build. `solution.py`'s third demo shows two teams doing this.

**If any box is unticked, that box is your next task** — not fine-tuning.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including the memory arithmetic showing where LoRA's saving actually comes from.
3. Run `python solution.py` for three demos: training-memory arithmetic across model sizes, a spurious-correlation detector, and the honest checklist.

**Next:** Module 13 — Deployment Basics. You've built a RAG bot, an agent, a vision extractor and an evaluation harness. Module 13 puts a UI on one and makes it shareable — the course's second portfolio milestone.
