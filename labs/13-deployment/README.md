# 🧪 Lab 13: Ship It Safely

**Module:** [13 — Deployment Basics](../../modules/13-deployment.md)

> ### 🚀 This is the course's second portfolio milestone.
> By the end you'll have a public URL you can send to anyone — running your own pipeline, with the protections that make leaving it up overnight a reasonable thing to do.

---

## Objective

Build the four components that stand between a public URL and an unbounded bill, then deploy for real.

By the end you will have:

1. **A token-bucket rate limiter** that permits bursts while bounding sustained rate
2. **An LRU cache with TTL** — and the cache-key discipline that stops it leaking data
3. **A budget guard** that degrades gracefully instead of breaking
4. **A pre-launch checker** encoding §13.12's checklist
5. **A deployed app** at a public URL

## Expected outcome

`python starter.py` reports **46 of 46 checks passing**, then four experiments. Part 2 gives you a live URL.

## Requirements

| | |
|---|---|
| **Part 1** | **Standard library only.** No packages, no API key. |
| **Part 2** | `gradio`, plus a free Hugging Face account |
| **Cost** | **Free hosting.** ~$0.10 of API usage. |
| **Time** | ~75 minutes |

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 5 tasks, 46-check self-test, 4 experiments. |
| `solution.py` | Reference solution + a `ProtectedEndpoint` wiring all four together + 3 demos. |
| `SOLUTION.md` | The reasoning, including the cache-key mistake that leaks data. |

---

## Part 1 — The protections (45 min)

```powershell
python labs/13-deployment/starter.py
```

Every component takes an **injected clock**, so the tests control time instead of sleeping. Same pattern as Lab 11's circuit breaker.

| Task | Component | Key idea | Module 13 § |
|---|---|---|---|
| 1 | `TokenBucket` | Bursts allowed, sustained rate bounded | 13.8 |
| 2 | `make_cache_key` | **Stable, and includes the user** | 13.7 |
| 3 | `ResponseCache` | LRU + TTL, two independent rules | 13.7 |
| 4 | `BudgetGuard` | Degrade instead of break | 13.6 |
| 5 | `preflight_checks` | The checklist as code | 13.12 |

### Task 1 — two details that matter

**Start full.** A brand-new user shouldn't be rate-limited on their first request. There's a test that the first `capacity` requests all succeed.

**Cap the refill.** Without `min(capacity, ...)`, an idle bucket accumulates unlimited burst — so a user quiet for an hour could then fire thousands of requests at once. There's a test that advances the clock 1000 seconds and confirms the burst is still just `capacity`.

### Task 2 — the check that isn't about correctness

```
[ OK ]  2. THE SAFETY ONE: a different user gives a different key
```

**If responses are personalised and the user isn't in the key, you will serve one user's data to another.** That's a data breach, not a bug — and nothing errors, because the cache is working exactly as designed with the wrong idea of what makes two requests identical.

`solution.py`'s second demo shows it happening with an account-balance question.

### Task 3 — LRU means *least recently used*, not oldest

```
[ OK ]  3. evicts the LEAST RECENTLY USED entry
```

The test inserts `a`, `b`, `c`, then **reads `a`**, then inserts `d`. The evicted key must be `b` — `a` was just used, so it's no longer the least recent. `OrderedDict.move_to_end` on every hit is what makes this work.

### Task 5 — blockers versus warnings

Seven blockers, seven warnings, and the split is deliberate:

```python
return {"ready": not blockers, ...}     # warnings do NOT block
```

**A checker that blocks on everything gets bypassed entirely**, and then it protects nothing. Blockers are the things that bound your worst case; warnings are things that make the product better.

**✅ Part 1 complete at `All 46 checks passed.`**

---

## The four experiments

### Experiment 1 — what a rate limiter saves

```
  requests attempted : 300
  allowed            : 69
  blocked            : 231
  saved              : $0.46 (77%)
```

Five minutes of a script firing once per second. **And the limiter didn't block the first ten requests** — a real user clicking around is unaffected. That's the property that makes bursts survivable while still bounding sustained volume.

### Experiment 2 — why caching pays more than you expect

```
  requests    : 500
  API calls   : 40
  hit rate    : 92%
```

Real traffic isn't uniform. A handful of questions dominate, because people ask the obvious things and demo visitors all click the same example. **That skew is exactly what a cache exploits.**

### Experiment 3 — degrading gracefully, plus a floating-point lesson

The budget guard refuses requests once the daily limit is reached, while the app stays *up* — serving a cached answer or a polite message rather than a 500.

Then look closely: **49 requests served, not 50, with $0.02 remaining.** That's floating-point drift — adding `0.02` to itself 49 times gives `0.9800000000000005`, not `0.98`.

Harmless here, and **not harmless in a billing system.** The standard fix is to never store money as a float: integer minor units (cents) or `decimal.Decimal`.

### Experiment 4 — the pre-launch check

Three configurations. The first — hardcoded key, no cap, no rate limit, debug on, stack traces returned — **is not a straw man.** Every one of those is the *default*: you get a hardcoded key by pasting it in, no spending cap by not visiting billing, and stack traces by not catching anything.

**Nothing on that list requires effort to get wrong.**

---

## Part 2 — Deploy it (30 min)

### Step 1: build the app

Create `app.py` in the repo root. This wires Part 1's components around whichever app you built in Modules 8, 9 or 10.

```powershell
pip install gradio
```

```python
"""app.py - a public, protected Gradio app."""

import os
import sys
import uuid

import gradio as gr

sys.path.append("labs/13-deployment")
from starter import TokenBucket, ResponseCache, BudgetGuard, make_cache_key

from rag import DocumentQA, load_documents        # your Module 8 bot

# --- Fail at STARTUP, not on the first request (section 13.5) ---
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Missing OPENAI_API_KEY")

MODEL = "gpt-4o-mini"
COST_PER_REQUEST = 0.002

qa = DocumentQA()
qa.index(load_documents("documents"))

cache = ResponseCache(max_size=500, ttl_seconds=3600.0)
budget = BudgetGuard(daily_limit=1.00)
buckets = {}


def get_bucket(session_id: str) -> TokenBucket:
    if session_id not in buckets:
        # 5 burst, then one every 10 seconds.
        buckets[session_id] = TokenBucket(capacity=5, refill_per_second=0.1)
    return buckets[session_id]


def answer(question: str, session_id: str) -> tuple[str, str]:
    """Run a question through every protection layer, in cost order."""
    # LAYER 0: input validation. Free.
    if not question or not question.strip():
        return ("Please enter a question.", "")
    if len(question) > 2000:
        return ("Question too long (max 2000 characters).", "")

    # LAYER 1: rate limit. Still free.
    bucket = get_bucket(session_id)
    if not bucket.allow():
        wait = round(bucket.time_until_available())
        return (f"Too many requests. Please wait {wait} seconds.", "")

    # LAYER 2: cache. Note session_id is IN the key.
    key = make_cache_key(MODEL, [{"role": "user", "content": question}],
                         0.0, session=session_id)
    cached = cache.get(key)
    if cached is not None:
        return (cached[0] + "\n\n_(cached)_", cached[1])

    # LAYER 3: budget. The last check before we spend anything.
    if not budget.can_spend(COST_PER_REQUEST):
        return ("Daily usage limit reached. Please try again tomorrow.", "")

    # LAYER 4: the expensive call.
    try:
        result = qa.answer(question)
    except Exception:
        # Never leak internals to the user (section 13.4).
        return ("Something went wrong. Please try again.", "")

    budget.record(COST_PER_REQUEST)

    sources = "\n".join(
        f"[{i}] {chunk['metadata']['source']}"
        for i, chunk in enumerate(result["chunks"], start=1)
    )
    cache.set(key, (result["answer"], sources))
    return (result["answer"], sources)


with gr.Blocks(title="Document Q&A") as demo:
    gr.Markdown(
        "# Document Q&A\n"
        "Answers come **only** from the indexed documents, with citations. "
        "If the documents don't cover it, the bot will say so."
    )

    # A per-browser-session id, so rate limits and cache are per user.
    session = gr.State(lambda: str(uuid.uuid4()))

    question_box = gr.Textbox(label="Question",
                              placeholder="Ask about the documents...")
    ask_button = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=6)
    sources_box = gr.Textbox(label="Sources", lines=3)

    # Examples matter - a blank box gets abandoned (section 13.2).
    gr.Examples(
        examples=["What is the refund policy?",
                  "Who approves large expenses?",
                  "What are the office hours?"],
        inputs=question_box,
    )

    ask_button.click(answer, inputs=[question_box, session],
                     outputs=[answer_box, sources_box])
    question_box.submit(answer, inputs=[question_box, session],
                        outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
```

```powershell
python app.py
```

### Step 2: test the protections locally

Before deploying, confirm each layer actually fires:

1. **Ask the same question twice.** Does the second say `(cached)`?
2. **Click Ask six times rapidly.** Does the sixth get rate-limited?
3. **Set `daily_limit=0.01` and ask a few questions.** Does it refuse gracefully?
4. **Paste 3,000 characters.** Is it rejected before any API call?
5. **Break `documents/` (rename it) and restart.** Does startup fail loudly, or does it boot and fail per request?

### Step 3: run the pre-launch check

```python
from starter import preflight_checks

result = preflight_checks({
    "api_key_source": "env",
    "env_in_gitignore": True,
    "env_in_dockerignore": True,
    "provider_spending_cap": 10.0,      # SET THIS IN YOUR BILLING DASHBOARD
    "daily_budget_limit": 1.0,
    "rate_limit_enabled": True,
    "max_tokens": 500,
    "max_input_chars": 2000,
    "health_endpoint": False,           # Gradio provides its own
    "structured_logging": False,
    "logs_user_input": False,
    "dependencies_pinned": True,
    "debug_mode": False,
    "returns_stack_traces": False,
})

print("READY" if result["ready"] else "NOT READY")
for blocker in result["blockers"]:
    print("  BLOCKER:", blocker)
```

> **⚠️ Answer honestly.** `provider_spending_cap: 10.0` is a lie unless you have actually set it in your billing dashboard. **Go and do that now if you haven't** — it's the one control that bounds your worst case absolutely.

### Step 4: deploy to Hugging Face Spaces

1. Create an account at [huggingface.co](https://huggingface.co)
2. **New → Space**, SDK **Gradio**, visibility **Public**
3. Add three files:

```
app.py             your app
requirements.txt   PINNED versions
README.md          with the Spaces YAML header
```

```yaml
---
title: Document Q&A
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

Ask questions about the indexed documents. Answers include citations,
and the bot says "I don't know" rather than guessing.
```

4. **Settings → Repository secrets → New secret**: `OPENAI_API_KEY`

   Never commit it. The Space injects it as an environment variable, so your `os.getenv` call is unchanged (§13.5).

5. Push. Watch the build log. Your app is live at `huggingface.co/spaces/<you>/<name>`.

### Step 5: send it to someone

**That's the milestone.** A URL, running your pipeline, that a stranger can use without costing you an unbounded amount.

### Then answer these

1. **How long from `git push` to a working URL?**
2. **Ask it something your documents don't cover.** Does it say "I don't know" (Module 8 §8.9)?
3. **Have someone else use it.** Did they ask what you expected? Did the examples help?
4. **Check the Space logs.** What can you see? What can't you?
5. **Ask the same question from two different browsers.** Do they share a cache entry? Should they?
6. **What would break first at a thousand users a day?**

Question 5 is worth thinking through carefully — the answer depends entirely on whether your responses are personalised.

**✅ Part 2 complete when you have a public URL you're comfortable sharing.**

---

## 🚀 Stretch Challenges

### 1. Add structured logging

Implement §13.10's `log_request`. Log length and a hash of the question, **not the text** — then justify that decision in a comment.

Run for a day and compute your `said_dont_know` rate. That's Module 11 §11.12's cheapest early-warning signal.

### 2. Wire all four layers with `ProtectedEndpoint`

`solution.py` has one. Build your own, then instrument it: what fraction of requests were rate-limited, cached, or budget-blocked?

### 3. Build the FastAPI version

Use §13.4's skeleton. Then compare: which was easier to build, and which would you rather hand to a frontend developer?

Add `/health` and check what your platform does when it fails.

### 4. Containerise it

Write the `Dockerfile` and `.dockerignore` from §13.11. Then verify the key isn't in the image:

```powershell
docker build -t myapp .
docker run --rm myapp cat /app/.env      # should fail - no such file
docker history myapp                      # inspect the layers
```

**If `.env` made it in, delete the image and fix `.dockerignore`.** A later `RM` layer does not remove it.

### 5. Add streaming

Convert to `gr.ChatInterface` with a generator (§13.9). Time to first token versus total time — which changed?

Then consider: **how do you validate output you haven't finished generating?**

### 6. Break your own app

- Paste 50,000 characters
- Click Ask twenty times in two seconds
- Ask the same thing from two browsers simultaneously
- Ask something designed to make it hallucinate
- Try a prompt injection (Module 11 §11.3)

**Write down what happened for each.** That list is your next sprint.

### 7. Swap in the free stack

Point it at Ollama instead of a hosted API. Cost per request drops to zero, and the budget guard becomes unnecessary — but latency rises and quality drops.

See `appendix/A-local-stack.md`. **Worth doing once**, because a deployment where data never leaves your infrastructure is the right answer for some problems.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including the measured trade-off between a per-user cache and a shared one.
3. Run `python solution.py` for three demos: all four layers on one request path, the cache-key leak, and what each layer is worth in money.

**Next:** Module 14 — Ethics & Limitations. The final module, and the one that asks what you *should* build rather than what you can.
