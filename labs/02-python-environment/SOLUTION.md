# Lab 2 — Solutions & Discussion

> **Attempt `starter.py` first.** The self-test tells you exactly what's expected, which is most of the help you need. Read on for the reasoning.

Complete runnable code is in [`solution.py`](solution.py). This file explains *why*.

---

## Task 1 — `build_prompt`

```python
def build_prompt(topic: str, level: str = "beginner", max_words: int = 100) -> str:
    return f"Explain {topic} in under {max_words} words. Audience level: {level}."
```

**One line.** The whole task is the f-string.

**Worth noticing:** the parameters are declared in the order `topic, level, max_words`, but the string uses them in the order `topic, max_words, level`. **Declaration order and usage order are unrelated.** Parameter order matters only to callers passing positional arguments.

**Why `level` and `max_words` have defaults:** most calls want a beginner-level, 100-word answer, so callers shouldn't have to repeat that. Defaults capture the common case; parameters allow the exception.

> **⚠️ The rule for defaults:** never use a mutable value (`[]`, `{}`) as a default. `def f(items=[])` creates **one** list shared across every call, so items accumulate between calls. It's a genuinely famous Python trap. Use `def f(items=None)` and build the list inside.

**Why this prompt shape avoids an article problem.** An earlier draft read `"Explain {topic} to a {level}."` — which produces *"to a expert"*. Rather than adding a/an logic, the phrasing moves `level` into a labelled field: `Audience level: expert.` Simpler code *and* a clearer prompt. Structured, labelled prompt fields turn out to be a real technique — it's most of Module 5.

---

## Task 2 — `estimate_tokens`

```python
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))
```

**Three decisions in two lines.**

**1. `if not text` rather than `if len(text) == 0`.** In Python, empty strings, empty lists and `None` are all "falsy". `not text` catches all of them, so this survives being handed `None` — which happens constantly when reading optional API fields.

**2. `round()` not `int()`.** `int()` truncates: `int(3.9)` is `3`. `round(3.9)` is `4`. For an estimate, rounding is more honest.

> **A Python quirk worth knowing:** `round()` uses *banker's rounding* — exact halves go to the nearest **even** number. `round(2.5)` is `2`, not `3`; `round(3.5)` is `4`. Surprising, standard in financial computing, and it's why the self-test avoids `.5` boundaries.

**3. `max(1, ...)` as a floor.** Without it, a 1-character string gives `round(0.25)` = `0` tokens — which is wrong, because even one character costs a token. `max(1, ...)` says "never below 1".

**Order matters:** the empty check must come *before* the `max(1, ...)`, or `estimate_tokens("")` would return 1 instead of 0.

### How good is 4 characters per token?

Decent for English prose, and poor elsewhere:

| Text | Estimated | Actual (roughly) | Why |
|---|---|---|---|
| Ordinary English | close | — | The rule was derived from it |
| Code | underestimates | more tokens | Punctuation and symbols often tokenise individually |
| Non-English | badly underestimates | far more | Poorly represented in English-trained vocabularies |
| Long numbers | underestimates | more | Digits split into small groups |
| Rare/technical words | underestimates | more | Fragment into sub-pieces |

That non-English row has a real-world consequence: **the same sentence in Hindi or Arabic can cost several times as many tokens as in English**, so identical requests cost different amounts depending on language. It's an equity issue as much as a billing one, and Module 3 measures it exactly with `tiktoken`.

---

## Task 3 — `estimate_cost`

```python
def estimate_cost(token_count: int, price_per_million: float = 0.15) -> float:
    return (token_count / 1_000_000) * price_per_million
```

**The underscores in `1_000_000` are ignored by Python** — purely a readability feature. `1_000_000` and `1000000` are identical. Use them for any large number; `10_000_000` versus `10000000` is the difference between glancing and counting zeros.

**Two things this function deliberately does not model:**

1. **Input and output tokens are priced differently** — output typically costs several times more. A real cost function needs both rates.
2. **Prices change**, which is why the rate is a parameter with an illustrative default rather than a hardcoded constant.

**Why estimate at all?** Because a loop over 10,000 documents costs 10,000 times one call. Estimating first turns "why is my bill $340?" into a number you saw before spending it. Module 13 covers cost control properly.

---

## Task 4 — `build_messages`

```python
def build_messages(system_prompt: str, user_prompt: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
```

**This is the most important function in the lab.** Not because it's hard — it's the easiest — but because this exact structure is what you pass to every chat model API for the remaining twelve modules.

Read the shape carefully:

- The outer **list** is ordered, because a conversation is chronological
- Each inner **dictionary** is one turn, labelled with `role` and `content`
- **`system` goes first** by convention — it frames everything that follows

**The three roles:**

| Role | Who it represents | Purpose |
|---|---|---|
| `system` | You, the developer | Standing instructions: persona, rules, output format |
| `user` | The person | The actual request |
| `assistant` | The model | Its previous replies, when you're continuing a conversation |

Adding conversation history means appending more dictionaries in order. That's all "memory" is at the API level — you re-send the transcript every time. Module 6 automates it, and Module 5 is entirely about what goes in that `system` message.

> **📌 The trailing comma** after the last dictionary is legal Python and good practice: adding a line later produces a one-line diff instead of two.

---

## Task 5 — `summarise_batch`

```python
def summarise_batch(prompts: list) -> dict:
    if not prompts:
        return {"count": 0, "total_tokens": 0, "longest": None}

    total_tokens = 0
    for prompt in prompts:
        total_tokens += estimate_tokens(prompt)

    longest = max(prompts, key=len)

    return {
        "count": len(prompts),
        "total_tokens": total_tokens,
        "longest": longest,
    }
```

### The empty-list guard is the actual lesson

Without the first two lines:

```python
>>> summarise_batch([])
ValueError: max() iterable argument is empty
```

`max()` cannot return the largest of nothing, so it raises. If you hit this, **you found a real bug class, not a puzzle.**

This is the **empty-collection edge case**, and it is one of the most common sources of production crashes in data pipelines. Your code works perfectly on every test you ran, then falls over at 3am when a query legitimately matches zero documents. In Module 8 you'll write retrieval code that must handle exactly this: *what does the pipeline do when nothing is found?*

The habit worth forming: **whenever you write a function that takes a collection, ask what it does when the collection is empty** — before you write the body.

### `max(prompts, key=len)`

`max()` normally compares items directly, which for strings means alphabetically. `key=len` tells it to compare `len(item)` instead, so it returns the longest string rather than the last alphabetically.

The equivalent explicit loop, if that reads more clearly to you:

```python
longest = prompts[0]
for prompt in prompts:
    if len(prompt) > len(longest):
        longest = prompt
```

Both are correct. `max(..., key=...)` is more idiomatic; the loop is more obvious. **Pick whichever you'd rather read in six months.**

### Reusing `estimate_tokens`

The loop calls `estimate_tokens()` instead of repeating `len(p) / 4`. If the estimation rule changes — and in Module 3 it does, to exact `tiktoken` counting — it changes in **one place** and every caller improves.

Duplicating the formula would mean hunting down every copy. This is the single most valuable habit in programming, and it has an unglamorous name: **don't repeat yourself**.

---

## Part C — Discussion

### 1. Estimate versus actual

Your estimate was probably **lower** than the actual count. Two reasons:

- **The 4-chars-per-token rule is approximate** (see Task 2's table)
- **The API adds tokens you didn't count** — every message carries a few tokens of role and formatting overhead, so a conversation costs slightly more than the sum of its text

The gap widens with many short messages, because per-message overhead is charged repeatedly.

### 2. Running it twice

**Different answers**, unless you set `temperature=0`. This is exactly Lab 1's Experiment 1, now visible in your own code — and now you can see the knob: `temperature`.

```python
temperature=0      # near-deterministic: same input, essentially same output
temperature=0.7    # balanced (the usual default)
temperature=1.5    # wild, often incoherent
```

**Practical guidance:** use low temperature for extraction, classification and anything you'll parse; higher for brainstorming and creative writing. Module 3 covers the sampling mechanics; Module 11 covers why non-determinism makes testing hard.

### 3. The pirate system prompt

**What changed:** tone, vocabulary, style.
**What didn't:** the two-sentence limit, and the factual content about tokens.

That's the system prompt doing its job — steering *how* the model responds without changing *what* it knows. It's a strong steering mechanism and not an absolute one, which is the entire subject of Module 5 (and, from the security angle, Module 11: if a system prompt were an absolute guarantee, prompt injection wouldn't exist).

---

## 🚀 Stretch — Discussion

### Comparing models

You'll typically find a small cheap model is **indistinguishable** from a large one on simple tasks — definitions, formatting, classification, extraction — and clearly worse on multi-step reasoning.

**The practical consequence:** defaulting to the most capable model for everything can cost 10–50× more than necessary. Matching model size to task difficulty is one of the highest-leverage cost optimisations available, and Module 13 covers routing between tiers.

### Retry with backoff

Two details in that snippet are worth extracting:

**Exponential backoff** — `2 ** attempt` gives 1s, 2s, 4s. Growing waits give a struggling service room to recover; hammering it every 100ms makes an outage worse.

**`for ... else`** — the `else` block runs only if the loop completed *without* `break`. It's an unusual construct, and this "retry N times, then give up" pattern is what it's genuinely good for.

Production code adds **jitter** (a small random offset) so that thousands of clients retrying simultaneously don't synchronise into a thundering herd. Module 11 covers the full pattern.

### Exact token counting

Running `tiktoken` against your approximation is the best possible preparation for Module 3. You should have found the approximation worst on:

- **Code** — `{`, `}`, `->`, `::` often tokenise individually
- **Non-English text** — sometimes 2–4× more tokens than the estimate
- **Long numbers** — `1234567` splits into several tokens
- **Emoji** — frequently several tokens each

If you did this, you've already answered Module 3's opening question. It starts by asking why `"tokenizing"` becomes `["token", "izing"]` — and you'll have seen it happen.

---

## Ready for Module 3?

Check yourself:

- [ ] Your environment activates, and you know how to tell (`(.venv)` in the prompt)
- [ ] You can explain why virtual environments exist, in one sentence
- [ ] `git status` does **not** list `.env`, and you know what to do if a key leaks
- [ ] You can write a function with a default parameter, a type hint and a docstring
- [ ] You can build a `messages` list from memory, with the right role names
- [ ] You know why `summarise_batch([])` needed a guard

Any gaps, reread that section of [Module 2](../../modules/02-python-and-environment.md).

**Next up: Module 3 — Tokens, Embeddings & Similarity.** You'll count tokens exactly, turn text into vectors, and build a working semantic search over a small set of documents — the foundation everything from Module 7 onward is built on.

---

<div align="center">

**[⬅ Back to Lab 2](README.md)** · **[📖 Module 2](../../modules/02-python-and-environment.md)** · **[🏠 README](../../README.md)**

</div>
