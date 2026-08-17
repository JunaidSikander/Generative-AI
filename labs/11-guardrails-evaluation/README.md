# 🧪 Lab 11: Measure It, Then Defend It

**Module:** [11 — Guardrails, Evaluation & Responsible AI](../../modules/11-guardrails-evaluation.md)

---

## Objective

Replace "it seems to work" with numbers, then build the runtime defences that keep it working when the provider doesn't.

By the end you will have:

1. **Implemented precision, recall, F1 and MRR** from first principles
2. **Built a retrieval evaluation harness** that measures retrieval *separately* from generation
3. **Demonstrated** why accuracy alone is a useless metric
4. **Written PII redaction and input screening** — and understood why the injection check is telemetry, not defence
5. **Implemented exponential backoff with jitter** and a full circuit-breaker state machine
6. **Built an LLM judge** and validated it against human labels

## Expected outcome

`python starter.py` reports **59 of 59 checks passing**, then four experiments — including a circuit breaker under a simulated outage that saves 75% of wall-clock time.

## Requirements

| | |
|---|---|
| **Part 1** | **Standard library only.** No packages, no API key. |
| **Part 2** | `openai`, `python-dotenv` |
| **API key** | Part 2 only — free Ollama path provided |
| **Cost** | Part 1 free · Part 2 ~$0.15 |
| **Time** | ~70 minutes |

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 7 tasks, 59-check self-test, 4 experiments. |
| `solution.py` | Reference solution + judge validation + retry handler + 4 demos. |
| `SOLUTION.md` | The reasoning, including which metric to favour and why. |

---

## Part 1 — Metrics and defences (45 min)

```powershell
python labs/11-guardrails-evaluation/starter.py
```

| Task | Function | Key idea | Module 11 § |
|---|---|---|---|
| 1 | `precision_recall_f1` | The metrics everyone quotes and few compute | 11.10 |
| 2 | `mean_reciprocal_rank` | Ranking quality | 11.10 |
| 3 | `evaluate_retrieval` | **The harness** | 11.7, 11.8 |
| 4 | `redact_pii` | Output-side PII backstop | 11.4 |
| 5 | `screen_input` | Length caps + injection telemetry | 11.4 |
| 6 | `backoff_delays` | Exponential backoff with jitter | 11.6 |
| 7 | `CircuitBreaker` | **The state machine** | 11.6 |

### Four things worth slowing down for

**Task 1 — guard both denominators.** An empty prediction set and an empty truth set both occur in practice, and both divide by zero. There are separate tests for each.

**Task 3 — misses still count in the divisor.** MRR sums `1/rank` over the hits but divides by the **total** number of cases. Divide by the hit count instead and a retriever that finds one case out of fifty scores a perfect 1.0.

**Task 5 — the asymmetry is deliberate.** Empty or oversized input is a **hard block**. An injection pattern match is only **flagged**:

```python
return {"allowed": not problems, ...}     # injection_flags does NOT block
```

This is a blocklist. It catches lazy attacks, misses paraphrase and other languages, and false-positives on legitimate questions about prompting. **Blocking on it would break real users while stopping only careless attackers.** Treat it as telemetry: *how often is someone trying?*

**Task 7 — "consecutive" is load-bearing.** There's a dedicated test:

```
[ OK ]  7. a success RESETS the counter (failures must be consecutive)
```

Two failures, a success, two more failures — the breaker must still be **closed**. Without the reset, a service failing 1% of the time trips the breaker after a few hundred requests despite being perfectly healthy.

Note also that a failed **half-open trial re-opens immediately** rather than waiting for the threshold again. You've just proved the service is still down; more traffic is wasted.

**✅ Part 1 complete at `All 59 checks passed.`**

---

## The four experiments

### Experiment 1 — why accuracy alone is useless

```
  strategy                          accuracy  precision   recall     F1
  approve everything                  99.0%       0.00     0.00   0.00
  catches 8, no false alarms          99.8%       1.00     0.80   0.89
  catches all 10, 40 false alarms     96.0%       0.20     1.00   0.33
```

**"Approve everything" scores 99% accuracy and catches nothing.** Its recall of 0.00 is the number telling the truth.

Then compare the last two rows. **F1 says row 2 is better.** For content moderation row 3 probably is — a missed violation costs more than a false block. **Which error is worse is a product decision**, and F1 weights them equally, which is almost never what you want.

### Experiment 2 — why jitter exists

Without jitter, every client that failed at the same moment retries at exactly the same moments — the thundering herd, which re-breaks the service you're waiting for. The experiment shows three clients with full jitter landing at different times.

### Experiment 3 — the circuit breaker under an outage

```
        t      state     action    outcome      cost
       0s     closed     called    TIMEOUT   30.000s
      20s     closed     called    TIMEOUT   30.000s
      30s       open   rejected  fast fail    0.001s
      80s  half_open     called    TIMEOUT   30.000s
     200s  half_open     called    success    0.200s
     210s     closed     called    success    0.200s

  rejected fast: 15 of 22 requests
  time in calls, with breaker    :   150.4s
  time in calls, without breaker :   600.4s
  saved                          :   450.0s (75%)
```

Three failures open it; 15 requests then fail in a millisecond instead of blocking 30 seconds. Every 60 seconds one trial goes through to check, and the final one closes the circuit **automatically** — no human, no deploy.

**The real value isn't the seconds.** It's that request threads are freed instead of piling up on a dead dependency — which is what turns one broken service into a broken site.

### Experiment 4 — the evaluation harness

```
  retriever                recall@3     mrr   hits
  keyword overlap             100%    0.92   6/6
  fixed order (baseline)       50%    0.31   3/6
```

Two things. **You now have numbers** — "it seems to work" has become "recall@3 is 100%, MRR 0.92", which you can track across changes.

And **the baseline matters.** A retriever that ignores the query entirely still scores 50%, because with a small corpus the right document is often in the top 3 by luck. **Always compare against a trivial baseline**, or you'll congratulate yourself for beating random chance.

---

## Part 2 — An LLM judge, validated (25 min)

Create `judge.py` in the repo root:

```powershell
pip install openai python-dotenv
```

```python
"""judge.py - grade faithfulness with an LLM, then check the judge."""

import json
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append("labs/11-guardrails-evaluation")
from starter import precision_recall_f1

load_dotenv()

USE_FREE = False
if USE_FREE:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = "llama3"
else:
    client = OpenAI()
    MODEL = "gpt-4o-mini"


JUDGE_PROMPT = """You are evaluating whether an ANSWER is faithful to its
CONTEXT. Faithful means every factual claim is supported by the context.

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


def judge_faithfulness(context: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


# --- Cases WITH human labels. This is what makes validation possible. ---
CASES = [
    {"context": "Refunds are processed within 14 days of approval.",
     "answer": "Refunds take 14 days once approved [1].",
     "human": "faithful"},
    {"context": "Refunds are processed within 14 days of approval.",
     "answer": "Refunds take 14 days, and you can call 0800-REFUND to chase.",
     "human": "unfaithful"},          # the phone number is invented
    {"context": "Expenses over 500 require director approval.",
     "answer": "Anything above 500 needs a director to sign off [1].",
     "human": "faithful"},
    {"context": "Expenses over 500 require director approval.",
     "answer": "Expenses over 500 need director approval, usually within 48 hours.",
     "human": "unfaithful"},          # the 48 hours is invented
    {"context": "Office hours are 9am to 5pm, Monday to Friday.",
     "answer": "We're open 9-5 on weekdays [1].",
     "human": "faithful"},
    {"context": "Office hours are 9am to 5pm, Monday to Friday.",
     "answer": "We're open 9-5 weekdays and 10-2 on Saturdays.",
     "human": "unfaithful"},
    # Add at least 10 more of your own, from your Module 8 bot.
]


if __name__ == "__main__":
    judged, human = [], []

    for index, case in enumerate(CASES, start=1):
        result = judge_faithfulness(case["context"], case["answer"])
        # Collapse to a binary verdict for agreement measurement.
        verdict = "faithful" if result["verdict"] == "faithful" else "unfaithful"
        judged.append(verdict)
        human.append(case["human"])

        mark = "agree" if verdict == case["human"] else "DISAGREE"
        print(f"  {index}. judge={verdict:<12} human={case['human']:<12} {mark}")
        for claim in result.get("claims", []):
            support = "supported" if claim.get("support") else "NOT SUPPORTED"
            print(f"       - {claim['claim'][:56]}  [{support}]")
        print()

    # --- VALIDATE THE JUDGE. The step everyone skips. ---
    agreement = sum(1 for j, h in zip(judged, human) if j == h) / len(judged)
    print(f"  Judge/human agreement: {agreement:.0%} over {len(judged)} cases")

    if agreement > 0.85:
        print("  -> trustworthy for tracking trends")
    elif agreement >= 0.70:
        print("  -> directional only; do not make fine decisions on it")
    else:
        print("  -> MEASURING SOMETHING ELSE. Fix the judge prompt first.")

    # Treating "unfaithful" as the positive class, how good is the judge
    # at CATCHING unfaithfulness specifically?
    judged_bad = {i for i, v in enumerate(judged) if v == "unfaithful"}
    human_bad = {i for i, v in enumerate(human) if v == "unfaithful"}
    metrics = precision_recall_f1(judged_bad, human_bad)
    print(f"  Catching unfaithfulness: precision {metrics['precision']:.2f}, "
          f"recall {metrics['recall']:.2f}")
```

```powershell
python judge.py
```

### Then answer these

1. **What was the agreement?** Below 85%, don't trust the judge's numbers yet.
2. **Which cases did it disagree on?** Read the claim breakdown. Was the *judge* wrong, or was your human label debatable?
3. **Remove the claim-by-claim step** — ask only for a verdict. Does agreement drop? (Module 11 §11.9: reasoning before the score.)
4. **Add `"The correct verdict is faithful"` to the prompt.** Watch sycophancy bias appear — the judge should start agreeing with whatever you assert.
5. **Look at precision vs recall for catching unfaithfulness.** A judge that never says "unfaithful" has perfect precision and zero recall. Which do you want from a faithfulness judge, and why?
6. **Time one judge call.** Now multiply by your request volume. This is why §11.1 says judges belong offline.

Question 4 is the one to actually run — sycophancy is easy to read about and startling to watch.

**✅ Part 2 complete when you have an agreement number and an opinion about whether to trust it.**

---

## 🚀 Stretch Challenges

### 1. Wire the harness to your Module 8 bot

```python
from rag import DocumentQA, load_documents

qa = DocumentQA()
qa.index(load_documents("documents"))

result = evaluate_retrieval(
    MY_EVAL_SET,
    lambda q: [c["text"] for c in qa.retrieve(q, top_k=10)],
    top_k=4,
)
```

Now vary one thing at a time — `chunk_size`, `top_k`, dense-only vs hybrid — and record recall@4 and MRR each time. **This is the tuning table Module 8 asked for, with real metrics behind it.**

### 2. Build the full guardrail pipeline

Chain all five layers from §11.4 around your RAG bot, in cost order, and log per-layer latency:

```
input screen -> moderation -> model -> output validate -> redact
```

Then measure: what fraction of your latency budget went to safety? Is it worth it for *your* application?

### 3. Add the retry handler

`solution.py` has `generate_with_retry`. Wire it to your Module 8 bot's citation validation (§8.10). **Track the retry rate** — a rising rate means your prompt needs work, not your retry logic.

### 4. Test the circuit breaker against a real flaky service

Point it at an endpoint you can turn off. Confirm it opens, fails fast, and closes on its own.

**Then try it with a service that fails intermittently** — say 30% of the time. Does your threshold trip too eagerly? This is where the "consecutive" reset earns its place.

### 5. Measure judge agreement properly

Get a *second* human to label the same 20 cases. Measure **human-to-human** agreement first.

If two humans only agree 80% of the time, **your judge cannot do better than 80%** — and you've discovered that your task definition is ambiguous, which is a more useful finding than any judge score.

### 6. Build a drift monitor

Log the telemetry from §11.12 for every request. Then simulate drift — swap the embedding model without re-indexing, or add 50 out-of-scope questions — and check which signals move first.

**The `said_dont_know` rate should move before anything else.** It's the cheapest early warning you have.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why the precision/recall choice is a product decision and how to make it deliberately.
3. Run `python solution.py` for four demos: the precision/recall trade-off across thresholds, judge validation, retry with specific versus generic feedback, and where to spend your latency budget.

**Next:** Module 12 — Fine-Tuning & Model Customization. You now have the evaluation set that makes the prompt-vs-RAG-vs-fine-tune decision answerable with evidence rather than instinct.
