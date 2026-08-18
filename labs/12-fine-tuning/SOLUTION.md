# Lab 12 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `recommend_approach`

```python
if not requirements.get("tried_prompting"):
    return {"recommendation": "prompting",
            "reasons": ["prompting has not been tried yet - always start here"],
            "blockers": []}
```

### Why that early return outranks everything

There's a test that reads almost as a provocation:

```
[ OK ]  1. untried prompting always wins
```

It passes 5,000 examples and `needs_house_style: True` — every signal pointing at fine-tuning — and expects `"prompting"`.

**That's deliberate, and it's the whole content of the task.** A large share of "we need to fine-tune" turns out to be "our prompt was vague". Finding that out costs an afternoon; discovering it after a month of data curation costs a month.

Encoding it as an unconditional early return means the framework can't be talked out of it by a sufficiently exciting-sounding requirement.

### The blocker falls back rather than failing

```python
if wants_tune and n_examples < MIN_EXAMPLES:
    blockers.append(...)
    wants_tune = False
```

Setting `wants_tune = False` means the *rest* of the logic still runs. So:

| Situation | Result |
|---|---|
| Style needed, 10 examples, no RAG signals | `prompting` |
| Style needed, 10 examples, **citations also needed** | `rag` |

A blocked fine-tune doesn't leave the user with nothing — it falls through to whatever else applies. There's a test for it.

### Why signals are (key, explanation) pairs

```python
RAG_SIGNALS = [
    ("needs_private_knowledge", "must know documents outside its training data"),
    ...
]
```

The recommendation alone isn't useful — **you need to be able to defend it to whoever asked for fine-tuning.** Pairing each key with its explanation means `reasons` is populated for free, and the output is a rationale rather than a verdict.

---

## Task 2 — `validate_training_example`

```python
messages = example.get("messages")
if not isinstance(messages, list) or not messages:
    return (False, ["missing or empty 'messages' list"])
```

Fail fast — every check below indexes into `messages`.

### The rule people forget

```python
last = messages[-1]
if isinstance(last, dict) and last.get("role") != "assistant":
    problems.append("last message must have role 'assistant'")
```

**The final assistant message is the training target.** It is the thing the model learns to produce. An example ending on a `user` turn has nothing to learn from — and depending on the library it will either error obscurely or train on nothing.

### Why one bad message can produce two problems

There's a test expecting **2** problems from:

```python
{"messages": [{"role": "user", "content": "hi"},
              {"role": "wizard", "content": "hello"}]}
```

The role is invalid *and* the example doesn't end with an assistant message. Both are true and both are reported — **a validator that stops at the first problem makes you fix errors one round-trip at a time.**

### `str(...)` before `.strip()`

```python
if not str(message.get("content", "")).strip():
```

Content can arrive as a number, `None`, or a nested object from a bad export. `str()` makes the check total rather than raising `AttributeError` on the first malformed row of a 5,000-line file.

---

## Task 3 — `validate_dataset`

### Why per-example validation isn't enough

Every problem this catches is invisible at the single-example level, and **every one of them trains successfully**:

| Problem | Symptom during training | Symptom in production |
|---|---|---|
| Too few examples | Loss drops fast | Overfitted, brittle |
| Duplicates | Nothing | Over-weighted patterns |
| Class imbalance | **Great accuracy** | Useless on the minority class |
| Invalid examples | Nothing | Fewer examples than you think |

> **🔑 The dangerous thing about all four is that the run completes and you get an adapter file.** There is no error to notice.

### The duplicate check

```python
serialised = [json.dumps(example, sort_keys=True, default=str)
              for example in examples]
duplicates = total - len(set(serialised))
```

`sort_keys=True` makes the serialisation canonical, so two dicts differing only in key order still compare equal — which matters because JSON exporters don't guarantee key order.

`default=str` stops the whole check crashing on an unexpected type.

### The imbalance threshold

```python
if largest > 10 * smallest:
```

10× is a judgement call, not a law. The purpose is to flag the case where the model can score well by ignoring a class entirely.

`len(label_counts) >= 2` guards the single-class case — one label isn't imbalanced, it's a different problem (you have nothing to classify).

### The label extraction is guarded

```python
try:
    counter[label_fn(example)] += 1
except Exception:
    continue
```

`label_fn` typically does `example["messages"][-1]["content"]`, which raises on exactly the malformed examples the validator is supposed to be *reporting*. Guarding it means one bad row doesn't prevent you seeing the report.

---

## Task 4 — `stratified_split`

```python
rng = random.Random(seed)

for label in sorted(by_label, key=str):
    shuffled = list(by_label[label])
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * validation_fraction)
    validation.extend(shuffled[:cut])
    train.extend(shuffled[cut:])
```

### Three decisions, all tested

**1. A local `random.Random(seed)`**, not `random.seed(seed)`. Seeding the global module changes randomness for the entire process — including anything the caller depends on. A local generator is contained.

**2. `sorted(by_label, key=str)`** for a stable iteration order. Dict ordering is insertion order in modern Python, so the split would otherwise depend on the order examples happened to arrive in. `key=str` handles mixed label types without raising.

**3. Stratified, not random.** The test that matters:

```
[ OK ]  4. a rare class still appears in validation
```

With 90 common and 10 rare examples, a random 20% split can easily produce a validation set containing zero rare examples. Your validation score for that class is then undefined — and if your metric quietly treats it as 1.0, you'll ship believing it works.

### Why seeding matters more than it looks

```
[ OK ]  4. the same seed gives the same split
```

If you change the LoRA rank and retrain, you need the *only* difference to be the rank. An unseeded split means you're comparing rank-8-on-split-A against rank-32-on-split-B, and any difference could be either.

This is the same principle as Module 11's injected clock: **push non-determinism to the boundary and control it.**

---

## Task 5 — the LoRA arithmetic

```python
def lora_parameter_count(d_in, d_out, rank):
    return rank * (d_in + d_out)
```

One line, and the whole economic argument for LoRA is in its shape.

### Linear versus quadratic

| | Growth in `d` |
|---|---|
| Full fine-tune | `d × d` — **quadratic** |
| LoRA | `r × 2d` — **linear** |

So the ratio `LoRA / full` is proportional to `1/d`. **Every increase in model size makes LoRA a proportionally better deal**, and there's a test asserting exactly that:

```
[ OK ]  5. lora saves MORE on a bigger matrix
```

Experiment 2 shows it: 2.08% of full at 768 dimensions, 0.195% at 8192.

**That's the property that made LoRA matter.** A technique that saved 50% would have been a footnote; one that saves proportionally more as models grow is what let fine-tuning survive the scaling era.

### The configuration summary

```
          7B      32     8     4,194,304     0.0599%       8.4M
         70B      80     8    20,971,520     0.0300%      41.9M
```

**A 70B model — around 140 GB of weights — specialised by a 42 MB file.** Small enough to commit to Git, email, or ship in a container layer.

Note the two 7B rows in the experiment. Quadrupling the rank from 8 to 32 quadruples the trainable parameters and still leaves you under a quarter of one percent.

> **🔑 Rank is cheap. The reason not to raise it is overfitting, not size.** Raise it when training loss plateaus too high — that's the signal the adapter is capacity-limited. Raising it because "more is better" adds overfitting risk and buys nothing.

### Where the memory saving actually comes from

`solution.py`'s first demo makes a point the parameter count alone doesn't:

```
     model    full fine-tune      LoRA r=8     QLoRA r=8
        7B            112 GB        14.1 GB        3.6 GB
       70B          1,120 GB       140.3 GB       35.3 GB
```

**LoRA barely reduces the weight memory** — the frozen base is still 14 GB at fp16 for a 7B model. What it removes is the **gradients and optimiser state**, which at ~16 bytes per trainable parameter is most of the cost of training.

QLoRA then attacks the remaining piece by quantizing the frozen base to 4 bits. The two techniques target different halves of the problem, which is why they compose so well.

---

## Task 6 — `break_even_volume`

```python
saving = base_cost - tuned_cost

if saving <= 0:
    return {..., "break_even_requests": None, "pays_back": False}

return {..., "break_even_requests": int(math.ceil(finetuning_cost / saving))}
```

`None` rather than infinity or a huge number: **no volume recovers a per-request loss**, and `None` says that unambiguously.

`math.ceil` because a partial request isn't a thing.

### What the experiment actually shows

I expected this table to demonstrate that an expensive fine-tuned model kills the cost case. **It doesn't**, and the real finding is more useful:

```
  scenario                                   saving/req      break-even
  2000 -> 50 tokens, tuned costs 2x       $  0.00028500     350,878 req
  2000 -> 50 tokens, tuned costs 8x       $  0.00024000     416,667 req
  400 -> 50 tokens, tuned costs 2x        $  0.00004500   2,222,223 req
  no prompt saving, tuned costs 2x        $ -0.00007500           NEVER
```

**Rows 1 and 2:** making the tuned model **eight times more expensive per token** only worsens break-even by 18%. When you delete 97% of the prompt, the per-token rate on what remains barely registers.

**Row 3:** a smaller prompt reduction — 400 to 50 instead of 2000 to 50 — needs **six times the volume**.

**Row 4:** no prompt reduction, no cost case, ever.

> **🔑 So the question is not "is the tuned model cheaper per token?" It usually isn't. It's "how much prompt can I delete?"**
>
> The saving is dominated by the token-count ratio, not the price ratio. That reframes the whole cost argument: fine-tuning for cost is worthwhile when your prompt is *long* — heavy instructions, many few-shot examples — and pointless when it's already short.

Row 5 in the experiment adds the obvious multiplier: a $5,000 fine-tune needs 50× the volume of a $100 one. Break-even scales linearly with upfront cost, so a cheap experiment can be worth running when an expensive one isn't.

---

## The experiments — discussion

### Experiment 1: the scenario designed to sting

```
  We think we need fine-tuning (prompt untried)
    -> PROMPTING
       because: prompting has not been tried yet - always start here
```

Five thousand examples. A genuinely specialist task. Still "try a prompt".

**People find this annoying, and that's the point.** The framework is encoding a discipline that's easy to agree with in the abstract and hard to follow when you're excited about a fine-tune.

### Experiment 3: the quiet failure

Five datasets, four broken, all of them trainable.

The 90/10 imbalance deserves attention because of *how* it fails:

1. Training completes normally
2. Loss looks fine
3. Test accuracy is ~90%
4. The model has learned to always answer "billing"
5. It is useless for exactly the class you built it to catch

**This is Module 11 §11.10 arriving a module early** — accuracy on imbalanced data measures how common the majority class is. The dataset validator catches it *before* you spend the GPU hours.

---

## Part 2 — Discussion

### 1. The parameter count

For Qwen2.5-0.5B (`d_model=896`, 24 layers, rank 8, targeting `q_proj` and `v_proj`):

```
  8 x (896 + 896) x 2 x 24 = 688,128
```

If `print_trainable_parameters()` reports something different, the likely reasons are that the model uses grouped-query attention (so `v_proj` is smaller than `q_proj`), or that the library adapted different modules than you assumed.

**Both are worth investigating rather than shrugging at.** Reading the actual module list is how you find out what you're really training.

### 3. Validation loss turning up

Common on a small dataset by epoch 2 or 3. `load_best_model_at_end=True` is what saves you from shipping the overfitted final epoch — and if you didn't set it, you shipped the worst checkpoint of the run.

**A training loss near zero is a warning.** It means memorisation, and it will reproduce your training examples perfectly while generalising poorly.

### 4. Fine-tuned versus few-shot prompted

**The honest question, and often an uncomfortable answer.** On a straightforward classification task with a well-constructed few-shot prompt, the gap is frequently small.

That is a **useful result, not a failed lab.** It tells you:

- Your task didn't need fine-tuning
- You've saved yourself the ongoing operational burden
- You now know *with evidence* rather than assumption

Module 12 §12.9 warns that comparing against a *zero-shot* baseline flatters a fine-tune enormously. This experiment is where you feel why.

### 5. General capability

Test something unrelated — arithmetic, general knowledge, summarising. On a small LoRA fine-tune with few epochs, general capability usually survives, because the base weights are untouched (§12.3).

**That's a genuine advantage of PEFT over full fine-tuning**, and it's worth confirming rather than assuming. Push the epochs up and you can still degrade it.

### 6. Was it worth it?

The question the whole module builds toward. Consider:

| | Few-shot prompt | Fine-tune |
|---|---|---|
| Time to build | An afternoon | Days (mostly data) |
| Change requirements | Edit a string | Retrain |
| Base model deprecated | Migrate in an hour | Redo the pipeline |
| Cost per request | Higher (long prompt) | Lower (short prompt) |
| Quality | Often within a couple of points | Usually the best |

**Fine-tuning is infrastructure.** It's the right call when the quality gap is real, the volume justifies it, and the task is stable. It's the wrong call for an experiment.

---

## 🚀 Stretch — Discussion

### 1. Length correlation

`solution.py`'s second demo builds a dataset where "escalate" examples are long and "close" examples are short — an entirely plausible accident, since detailed complaints do tend to get escalated.

The model learns **length**. A terse but furious *"This is unacceptable. Manager. Now."* gets closed.

This is the same failure as Lab 1's Teachable Machine stretch (the model that learned position instead of the object) and Module 5 §5.5's few-shot warning. **It recurs at every level of the stack** because it isn't a bug in any technique — it's a consequence of models learning whatever most easily predicts the label.

The detector is crude (a mean-length ratio) and it catches the obvious cases, which are most of them.

### 3. The rank sweep

You'll likely find little difference between `r=4`, `r=8` and `r=32` on a small, straightforward task.

**That's the expected result**, and it's why §12.4 says to start at 8. Rank matters when the adapter is genuinely capacity-limited — a complex task, a large dataset, a big behavioural shift. For classification with 100 examples it isn't the bottleneck.

### 5. The full five-row comparison

**The single most valuable thing in this lab**, and it belongs *before* any fine-tuning decision rather than after.

Building it with Lab 11's harness takes an hour and answers the question the whole module is about. Most teams skip it, fine-tune, and then have no idea whether it helped.

### 6. The honest checklist

`solution.py`'s third demo contrasts two teams. Team A has the examples — the part that *feels* like progress — and none of the things that would tell them whether the fine-tune worked.

> **If any box is unticked, that box is your next task.** Not fine-tuning.

---

## Ready for Module 13?

- [ ] You can state the ordering: prompt, then RAG, then fine-tune
- [ ] You can explain why fine-tuning on documents makes hallucination worse
- [ ] You know the test: can you write down what correct looks like, but not why?
- [ ] You can explain why LoRA saves proportionally more on bigger models
- [ ] You know where LoRA's memory saving actually comes from (optimiser state, not weights)
- [ ] You can name four dataset failures that train perfectly happily
- [ ] You know why the split must be stratified *and* seeded
- [ ] You can say what really drives the break-even calculation
- [ ] You'd compare against a few-shot baseline, not a zero-shot one

**Next: Module 13 — Deployment Basics.** You've built a RAG bot, an agent, a vision extractor and an evaluation harness, and they all run on your laptop. Module 13 puts a UI on one, ships it, and covers the cost and rate-limit realities of letting other people use it — the course's second portfolio milestone.

---

<div align="center">

**[⬅ Back to Lab 12](README.md)** · **[📖 Module 12](../../modules/12-fine-tuning.md)** · **[🏠 README](../../README.md)**

</div>
