# 🧪 Lab 14: Measure It, Then Decide

**Module:** [14 — Ethics & Limitations](../../modules/14-ethics-limitations.md)

> ### 🎓 The final lab.
> Ethics discussions usually stay abstract. This one is arithmetic — because "is our system fair?" has a measurable answer, and the interesting part is that the measurements disagree.

---

## Objective

Turn three ethical questions into computations you can run.

By the end you will have:

1. **Implemented three fairness metrics** and demonstrated for yourself that they disagree
2. **Computed the energy break-even** between training and inference — and found which dominates
3. **Built a structured impact assessment** and run it on something you actually want to build
4. **Seen** that removing a protected attribute doesn't remove the disparity

## Expected outcome

`python starter.py` reports **43 of 43 checks passing**, then four experiments — including one showing a system with *perfect* demographic parity and a 40-point gap in true positive rates.

## Requirements

| | |
|---|---|
| **Packages** | **None.** Pure standard library. |
| **API key** | None |
| **Cost** | Free |
| **Time** | ~45 minutes |

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 6 tasks, 43-check self-test, 4 experiments. |
| `solution.py` | Reference solution + intersectional and false-positive metrics + 3 demos. |
| `SOLUTION.md` | The reasoning, including why fairness metrics provably conflict. |

---

## Part 1 — The measurements (45 min)

```powershell
python labs/14-ethics-limitations/starter.py
```

| Task | Function | Key idea | Module 14 § |
|---|---|---|---|
| 1 | `selection_rates` | Positive-outcome rate per group | 14.2 |
| 2 | `disparate_impact_ratio` | The four-fifths rule | 14.2 |
| 3 | `true_positive_rates` | Equal opportunity — needs ground truth | 14.2 |
| 4 | `fairness_report` | **Detect when the metrics disagree** | 14.2 |
| 5 | `energy_break_even` | Training versus inference | 14.4 |
| 6 | `impact_assessment` | Should you build it? | 14.9 |

### Task 1 — check the lengths

`zip()` silently truncates to the shorter list. Given 1,000 predictions and 999 group labels, you'd get an answer — a wrong one, with no error. There's a test that expects `ValueError`.

**In a fairness audit specifically, a silently wrong answer is worse than a crash.**

### Task 3 — omit, don't zero

```
[ OK ]  3. a group with no positives is OMITTED, not scored 0.0
```

If a group has nobody who *should* have been selected, there is no true positive rate to compute. Recording `0.0` would read as total failure rather than no data — and would then poison the max/min in the gap calculation.

**"No data" and "zero" are different**, and conflating them produces confidently wrong fairness numbers.

### Task 4 is the point of the lab

```
[ OK ]  4. selection rates are IDENTICAL
[ OK ]  4. disparate impact looks PERFECT
[ OK ]  4. and it passes the four-fifths rule
[ OK ]  4. but the true positive rates differ badly
[ OK ]  4. THE POINT: the metrics disagree, and it is detected
```

The test data: two groups, 100 people each, exactly 50 qualified in each, and **both selected at exactly 50%**. Group A's picks are mostly qualified people; group B's are mostly not.

Also note:

```
[ OK ]  4. and disagreement cannot be claimed without measuring it
```

Without ground truth, `metrics_disagree` must be `False` — not because there's no disagreement, but because **you haven't measured it.** Claiming a clean bill of health from metrics you couldn't compute is the failure mode this guards against.

### Task 6 — the rubber-stamp rule

```
[ OK ]  6. rubber-stamp review scores worse than real review
```

A system claiming human review *without time to review* scores **worse** than one honestly admitting it has none.

That's deliberate. A reviewer approving a hundred confident recommendations an hour isn't reviewing — they're providing false assurance, and everyone downstream believes a human checked it. **Honest "no review" is safer than dishonest "review", because at least the risk is visible.**

**✅ Part 1 complete at `All 43 checks passed.`**

---

## The four experiments

### Experiment 1 — the metrics disagree

```
  scenario                          DI ratio   4/5?   EO gap     verdict
  obviously unequal                     0.60     NO     0.30      unfair
  equal rates, unequal accuracy         1.00    yes     0.40    DISAGREE
  genuinely fair                        1.00    yes     0.00        fair
```

Row 2 in detail:

```
    selection rates    : {'A': 0.5, 'B': 0.5}
    disparate impact   : 1.00  <- looks PERFECT
    true positive rates: {'A': 0.9, 'B': 0.5}
    opportunity gap    : 0.40  <- badly unfair
```

**It passes the two most commonly cited metrics with a perfect score**, and it's far worse at finding qualified people in group B — reaching the same headline number by selecting 25 candidates it shouldn't have.

It is **mathematically impossible** to satisfy every fairness definition at once. So which one you optimise is a values decision, and it should be written down and defended.

### Experiment 2 — inference dominates

```
       queries/day      break-even    annual inference
            10,000    100,000 days           3,650 kWh
         1,000,000      1,000 days         365,000 kWh
       100,000,000         10 days      36,500,000 kWh
```

At a million queries a day, cumulative inference overtakes training in under three years. At a hundred million, ten days.

**That's the opposite of how this is usually stated** — and it's good news, because inference is the part you control. Every Module 13 optimisation is an environmental lever too.

### Experiment 3 — what needs more care

Four systems, from a personal note summariser to CV screening. The support-ticket router is the interesting one: it scores *worse* than it would with no human review, because it claims review that has no time.

### Experiment 4 — assess your own project

Edit `MY_PROJECT` in `starter.py` to describe something you're genuinely considering, and re-run.

**Answer honestly.** A checklist you flatter yourself through is worse than none, because it produces a defensible-looking document and changes nothing.

---

## 🚀 Stretch Challenges

### 1. Add false positive rate parity

`solution.py` has `false_positive_rates`. It's the mirror of equal opportunity, and it's the metric that matters when the harm is being **wrongly flagged** rather than wrongly excluded — fraud detection, content moderation, risk scoring.

Then construct a system that equalises TPR across groups but **not** FPR. You now have a third metric disagreeing with the first two.

### 2. Check intersectionality

`solution.py`'s second demo shows a system where **each attribute passes the four-fifths rule at 0.92** while one intersectional cell is selected at half the rate of another.

Build your own `intersectional_rates` and confirm it. **Auditing attributes one at a time misses this entirely, and one at a time is how it's usually done.**

### 3. Prove that removing the attribute doesn't help

`solution.py`'s first demo builds a population where a neutral-looking feature correlates with group membership, then trains a model that **never sees the group**. It still produces a disparate impact ratio of 0.32.

Build your own version. Then note the second-order problem: **had group membership not been recorded, you couldn't have produced these numbers at all** — and the system would look clean because nothing was being checked.

### 4. Add the grid to your carbon arithmetic

`carbon_from_energy` in `solution.py` shows a **26× spread** in emissions for identical energy, depending on grid intensity.

Compute the annual footprint of something you've actually built (Module 13's app), on your region's grid. Then compare against something familiar — a flight, a household, a commute — and note how hard it is to make that comparison honestly.

### 5. Write a real disclosure

Take §14.10's four elements — states it's AI, states the scope, admits fallibility, gives a route to a human — and write one for your Module 13 app.

Then show it to someone who hasn't used the app. **Did they understand what it can and can't do?**

### 6. Do a proper impact assessment

Take the highest-risk thing you're considering and write the assessment out properly: who it affects, what happens when it's wrong, who bears the cost, how someone contests it, what you'd measure and how often.

**Then have someone who isn't you read it.** The version of this that catches problems is the one reviewed by somebody without a stake in the answer.

### 7. Revisit an earlier lab with this lens

Pick one thing you built in Modules 8–13 and run `impact_assessment` on it honestly. Most will come out low or medium risk — **and the interesting part is noticing which assumptions made it so.**

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why the fairness metrics conflict as a matter of arithmetic rather than engineering.
3. Run `python solution.py` for three demos: why removing the protected attribute doesn't work, intersectional blind spots, and why a carbon figure without a grid figure means little.

---

## 🎓 That's the course

Fourteen modules. Fourteen labs. Two portfolio pieces.

**The habit that ran through all of it:** measure it, don't assume it. Retrieval recall, extraction accuracy, judge agreement, cache hit rate, cost per request, fairness across groups. Every one of those is a number you now know how to produce.

Head back to [Module 14's closing section](../../modules/14-ethics-limitations.md#-where-to-go-next) for what to do next.

**Go and build something.**
