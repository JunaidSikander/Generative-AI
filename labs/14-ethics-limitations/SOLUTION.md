# Lab 14 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `selection_rates`

```python
if len(predictions) != len(groups):
    raise ValueError(...)
```

### Why the length check earns its place

`zip()` silently truncates to the shorter list. Given 1,000 predictions and 999 group labels, you get an answer — a wrong one, computed over 999 rows, with no error.

In most code that's a bug. **In a fairness audit it's worse than a bug**, because the output is a number someone will use to justify shipping a system. A silently wrong fairness figure is more dangerous than no figure at all.

There's a test for it.

### The zero-total guard

```python
return {group: positives[group] / totals[group]
        for group in totals if totals[group] > 0}
```

`totals` is a `defaultdict`, and iterating it after `positives[group]` lookups could otherwise introduce keys with zero totals. The guard makes the function total over its inputs.

---

## Task 2 — `disparate_impact_ratio`

```python
if highest == 0:
    return 1.0
```

### Why "nobody selected" returns 1.0

If no group had anyone selected, every group was treated identically — equally badly, but identically. There's no *disparity* to report.

More practically: without this, an ordinary input (a filter that matched nothing, an empty batch, a quiet hour) divides by zero and crashes your monitoring job. There's a test:

```
[ OK ]  2. nobody selected anywhere gives 1.0, not a crash
```

### The four-fifths rule

```python
FOUR_FIFTHS = 0.8
```

Not an arbitrary number. It comes from US employment-discrimination practice, where a selection rate for one group below four-fifths of the highest group's rate is treated as evidence warranting investigation.

> **⚠️ It is a screening threshold, not a verdict.** A ratio of 0.79 doesn't prove discrimination and 0.81 doesn't prove fairness. It's a flag that says *look harder here* — and treating it as a pass/fail gate is how systems get certified as fair at 0.81.

There's a test at exactly 0.8, because that's where an off-by-one in the comparison would show.

---

## Task 3 — `true_positive_rates`

```python
if actual:
    should_be_positive[group] += 1
    if prediction:
        correctly_positive[group] += 1
```

**Only rows that should be positive contribute.** Negatives are irrelevant here — that's precisely what makes this "equal opportunity" rather than "equal outcome". It asks: *among people who deserved a positive outcome, did each group get it equally often?*

### Omit, don't zero

```python
return {group: correctly_positive[group] / should_be_positive[group]
        for group in should_be_positive if should_be_positive[group] > 0}
```

There's a dedicated test:

```
[ OK ]  3. a group with no positives is OMITTED, not scored 0.0
```

If a group contains nobody who *should* have been selected, there is no rate to compute. Recording `0.0` would:

- Read as **total failure** rather than **no data**
- Poison the `max - min` in `equal_opportunity_difference`, producing a large fabricated gap
- Make a group you know nothing about look like the worst-treated one

> **🔑 "No data" and "zero" are different values.** This is the third time this distinction has mattered in the course — Module 2's `None`-versus-absent, Module 10's `legible` flag, and here. Conflating them produces confidently wrong numbers, which in a fairness context is the worst possible failure.

---

## Task 4 — `fairness_report`

### The demonstration this function exists for

```
    selection rates    : {'A': 0.5, 'B': 0.5}
    disparate impact   : 1.00  <- looks PERFECT
    true positive rates: {'A': 0.9, 'B': 0.5}
    opportunity gap    : 0.40  <- badly unfair
```

The test data is constructed carefully: two groups of 100, exactly 50 genuinely qualified in each, and **both selected at exactly 50%**.

Group A: 45 of its 50 selections are qualified people (TPR 0.90).
Group B: 25 of its 50 selections are qualified people (TPR 0.50).

**Both groups get 50 slots. Group B's slots go to the wrong people.** A qualified person in group B is far less likely to be picked than a qualified person in group A — while every headline metric reports perfect parity.

### Why this isn't an engineering shortfall

It is **mathematically impossible** to satisfy demographic parity, equal opportunity and calibration simultaneously, except when base rates are identical across groups or the classifier is perfect. This is a proven impossibility result, not a limitation of anyone's implementation.

> **🔑 So choosing which fairness definition to optimise is a values decision, not a technical one.**
>
> It belongs to whoever owns the product and its consequences. It should be **written down and defended** — not left implicit in whichever metric a library computes by default, and certainly not chosen after the fact because it produced a favourable number.

### The guard nobody expects

```
[ OK ]  4. and disagreement cannot be claimed without measuring it
```

Without `actuals`, `metrics_disagree` must be `False` — not because there's no disagreement, but because **you haven't measured it**.

That's the honest behaviour. The alternative — defaulting to "no disagreement detected" and letting a reader infer "fair" — is exactly how a system gets a clean bill of health from metrics that were never computed.

**Absence of evidence gets reported as absence of evidence.**

---

## Task 5 — `energy_break_even`

```python
break_even_queries = int(math.ceil(training_kwh / inference_kwh_per_query))
```

Straightforward arithmetic, and the result is the part worth internalising:

```
       queries/day      break-even
            10,000    100,000 days     (~274 years)
         1,000,000      1,000 days     (~2.7 years)
       100,000,000         10 days
```

**At any meaningful deployment scale, inference overtakes training quickly** — and then keeps going, forever, while training happened once.

That's the opposite of the usual framing, where training energy is the headline number. It matters because it changes who has leverage:

| | Training footprint | Inference footprint |
|---|---|---|
| Who controls it | The model provider | **You** |
| Your lever | Choose a smaller model | Caching, prompts, retrieval, image size |

Every Module 13 optimisation was already motivated by cost. **They're environmental levers too**, which is a convenient alignment rather than a trade-off.

### The unit conversions

`1 MWh = 1,000 kWh` and `1 kWh = 1,000 Wh`. Getting these wrong by a factor of a thousand is easy and the result still *looks* plausible, which is why the test pins exact values.

### Why the figures are labelled illustrative

Published energy and carbon estimates for models vary by **orders of magnitude**, because they:

- Are estimates, since providers rarely publish measurements
- Go stale fast as efficiency improves
- Compare different things — training only, or including all failed experiments
- Omit grid carbon intensity, which changes the answer by 26× (see below)

**The shape of the argument is robust. The numbers are not.** Use the shape.

---

## Task 6 — `impact_assessment`

### The rubber-stamp rule

```python
if has_review and not system.get("human_review_has_time"):
    score += 1
    concerns.append("human review exists but has no time to be real...")
```

There's a test:

```
[ OK ]  6. rubber-stamp review scores worse than real review
```

And notice what it implies: claiming review *without time* scores **worse than claiming no review at all**.

That's deliberate, and it's the most opinionated line in the lab. A reviewer approving a hundred confident recommendations an hour is not reviewing. They are:

- Providing **false assurance** to everyone downstream
- Absorbing accountability that the system's designers should hold
- Subject to well-documented **automation bias** — people defer to confident systems, especially under time pressure

**Honest "no human review" is safer than dishonest "human review", because at least the risk is visible** and someone might address it.

### Why invisible errors score double

```python
if affects_people and not system.get("errors_visible_to_affected_person"):
    score += 2
```

Module 14 §14.9: **the dangerous shape is errors that are invisible to the person affected and costly to them.**

If a candidate is screened out by a model and never learns a model was involved, they cannot notice the error, cannot challenge it, and cannot even know there was something to challenge. The system's accuracy is almost beside the point — the *feedback loop is broken*, so errors never surface and never get fixed.

### Why `required_before_building` only fires at high risk

A checklist that demands an evaluation set for a personal note summariser gets ignored, and then it's ignored for the CV screener too.

Same principle as Lab 13's blockers-versus-warnings split: **a check that fires on everything protects nothing.**

---

## The demonstrations in `solution.py`

### 1. Removing the protected attribute doesn't help

The demo builds a population where a neutral-looking feature correlates with group membership — as postcode, school, employment gaps and writing style all do in practice. Then it builds a model that **never sees the group**:

```
    selection rates       : {'B': 0.253, 'A': 0.798}
    disparate impact ratio: 0.32
    passes four-fifths    : NO
```

**The model has no access to group membership and produces a 3× disparity.**

> **🔑 Removing the protected attribute removes your ability to MEASURE the disparity. It does not remove the disparity.**

And there's a second-order problem the demo points out: **had group membership not been recorded, these numbers couldn't have been produced at all** — and the system would look clean because nothing was being checked.

That creates a genuine tension with data-minimisation principles: you may need to collect a sensitive attribute in order to audit for discrimination against it. There's no clean resolution, and it's worth knowing the tension exists rather than discovering it mid-project.

### 2. Intersectionality

```
  Checking ONE attribute at a time:
    attribute 1: {'X': 0.6, 'Y': 0.55}  ratio 0.92  passes
    attribute 2: {'P': 0.6, 'Q': 0.55}  ratio 0.92  passes

  Checking the COMBINATION:
    ('X', 'P'): 0.80   ('X', 'Q'): 0.40
    ('Y', 'P'): 0.40   ('Y', 'Q'): 0.70
    ratio 0.50  FAILS
```

**Each attribute passes comfortably at 0.92. Two intersectional cells are selected at half the rate of the best cell.**

Auditing attributes one at a time misses it entirely — and one at a time is how it's usually done, because it's what the tooling makes easy.

*(The first version of this demo used cell values where both single attributes actually failed, which didn't demonstrate anything. Finding values where each attribute genuinely passes while the combination fails took a search — which is itself informative about how non-obvious this failure mode is.)*

### 3. Carbon depends on the grid

```
  grid                                 gCO2/kWh   tonnes CO2/yr
  very low carbon (hydro/nuclear)            30             110
  world average, roughly                    400           1,460
  coal-heavy                                800           2,920
```

**Identical energy. A 26× spread in emissions.**

So "this model emits X tonnes" is not a property of the model. It's a property of the model *and* where it runs *and* when it runs — grid intensity varies by region, season and time of day.

This is why published figures diverge so wildly, and why quoting one without its assumptions is close to meaningless. **It also means workload placement is a real lever**, independent of any efficiency work.

---

## 🚀 Stretch — Discussion

### 1. False positive rate parity

The mirror of equal opportunity, and it's the one that matters when the harm is being **wrongly flagged** rather than wrongly excluded.

| Harm | Metric |
|---|---|
| Wrongly excluded from an opportunity | Equal opportunity (TPR parity) |
| Wrongly flagged as a risk | **FPR parity** |
| Unequal access to a resource | Demographic parity |

A fraud-detection system with equal TPR but unequal FPR catches fraud equally well across groups while **falsely accusing one group far more often**. That's a serious harm invisible to the metrics in tasks 1–3.

Now you have three metrics that can each disagree with the other two. That's not a defect in the metrics — it reflects genuinely different ideas about what fairness means.

### 3. The measurement paradox

The most uncomfortable finding in this lab: **you often need to collect a sensitive attribute in order to detect discrimination against it.**

Data minimisation says collect less. Fairness auditing says you cannot measure what you don't record. Both are right, and the resolution is usually procedural — collect it, restrict access to it tightly, use it only for aggregate auditing, never as a model feature.

**Worth deciding deliberately rather than discovering when someone asks for a fairness report you can't produce.**

### 4. Honest comparisons

You'll find carbon comparisons are hard to make well. A flight's emissions depend on route, aircraft, occupancy and altitude effects; a household's on climate and grid; a model's on all the factors above.

**The honest version usually has more caveats than the comparison is worth**, which is a useful thing to learn before quoting one.

### 5. Disclosure

The test is showing it to someone who hasn't used the app. Common findings:

- They didn't read it
- They read it and didn't understand the scope
- They understood "AI" and missed "may be wrong"

**Disclosure that isn't understood isn't disclosure.** §14.10's point that good disclosure is a product feature rather than a legal disclaimer is testable, and this is the test.

### 7. Revisiting earlier labs

Most of what you built comes out low or medium risk — a personal tool, a documentation Q&A bot.

**The interesting part is noticing which assumptions made it so.** Your Module 8 bot is low risk *because* it's informational, cites sources, says "I don't know", and nobody's employment depends on it. Change any of those and the assessment changes.

That's the transferable skill: **risk is a property of the deployment, not the technology.** The same RAG pipeline is a harmless documentation search or a consequential decision system depending entirely on what you point it at and who it affects.

---

## 🎓 That's the course

Fourteen modules, fourteen labs, two portfolio pieces.

### What you can now do that you couldn't

- Explain how these systems work, from tokens to attention
- Write prompts that produce parseable output
- Build a RAG system with citations that says "I don't know"
- Give a model tools without giving it a path to arbitrary code execution
- Extract validated structured data from images
- **Measure** retrieval quality, extraction accuracy, judge agreement, and fairness
- Ship to a public URL without exposing an unbounded bill
- Decide what not to build

### The thread

Every technique in this course exists to compensate for something the model cannot do. RAG because it doesn't know your data. Citations because it can't be trusted. Schemas because its output isn't structured. Guardrails because it can be persuaded. Evaluation because it can't tell you when it's wrong. Fairness metrics because it will reproduce whatever was in its training data, accurately.

**That framing is what makes the next technique legible when it arrives** — and there will be a next technique, probably soon.

### The habit

Measure it, don't assume it.

You now know how to produce a number for every claim you might want to make about a system you built. Systems built by people who measure are better than systems built by people who are confident.

**Go and build something.**

---

<div align="center">

**[⬅ Back to Lab 14](README.md)** · **[📖 Module 14](../../modules/14-ethics-limitations.md)** · **[🏠 README](../../README.md)**

**🎓 Well done.**

</div>
