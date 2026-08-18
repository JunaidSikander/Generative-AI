# Lab 13 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `TokenBucket`

```python
def _refill(self):
    now = self.clock()
    elapsed = now - self.last_refill
    if elapsed > 0:
        self.tokens = min(self.capacity,
                          self.tokens + elapsed * self.refill_per_second)
        self.last_refill = now
```

### Lazy refill

No background timer, no thread, no scheduled task. Tokens are computed from elapsed time **whenever someone asks**.

That matters practically: a rate limiter that needed a background thread would need one per process, wouldn't survive a fork, and would burn CPU on idle buckets. This version is a few floats and some arithmetic.

### The `min()` is doing real work

There's a test that advances the clock 1,000 seconds and confirms the burst is still just `capacity`:

```
[ OK ]  1. an idle bucket caps at capacity, not more
```

Without the cap, a user quiet for an hour accumulates 3,600 tokens and can then fire all of them at once. **The whole point of a rate limiter is bounding the worst case**, and an uncapped bucket has no worst case.

### Why it starts full

```python
self.tokens = float(capacity)
```

A brand-new user should not be rate-limited on their first request. Starting empty would mean every visitor waits before their first interaction — technically a rate limit, practically a broken product.

### Why a zero refill rate is rejected

```python
if refill_per_second <= 0:
    raise ValueError(...)
```

It would "work" — the bucket would allow `capacity` requests and then block forever. **That's a permanent block dressed up as a rate limit**, and a config typo shouldn't be able to produce it silently.

### `time_until_available` is what makes it usable

```python
return (cost - self.tokens) / self.refill_per_second
```

A limiter that only says "no" invites clients to retry immediately, which makes things worse. Returning the wait lets you send `429` with a `Retry-After` header, and a well-behaved client backs off correctly.

It's the same idea as Module 11 §11.6's backoff: **tell the caller when to come back**, don't just refuse.

### Cost weighting

```python
bucket.allow(cost=1.0 + 0.2 * top_k)
```

Counting requests treats a query retrieving 2 chunks the same as one retrieving 20. Weighting by expected cost **bounds spend rather than volume**, which is what you actually care about (§13.8).

---

## Task 2 — `make_cache_key`

```python
payload = {"model": model, "messages": messages,
           "temperature": temperature, **extra}
blob = json.dumps(payload, sort_keys=True, default=str)
return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

### `sort_keys=True` is the correctness bit

Without it, `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` serialise differently and hash differently — so two logically identical requests miss the cache. There's a test:

```
[ OK ]  2. keyword ORDER does not change the key
```

### `default=str` is the robustness bit

Something unserialisable in the payload — a datetime, a custom object — would otherwise raise `TypeError` deep inside your cache layer. `default=str` degrades to a string representation instead.

### The check that isn't about correctness

```
[ OK ]  2. THE SAFETY ONE: a different user gives a different key
```

`solution.py`'s second demo makes it concrete:

```
  Question: 'What is my account balance?'

  WITHOUT the user in the key:
    alice -> ecabf7e3c85cf10c4a071f19...
    bob   -> ecabf7e3c85cf10c4a071f19...
    same key: True   <-- BOB GETS ALICE'S BALANCE
```

**Nothing errors.** The cache is working exactly as designed — it just has the wrong idea of what makes two requests identical.

> **🚨 If the response depends on WHO is asking, the identity must be in the key.** This is not a performance bug. It is a data breach, and it's the kind that passes code review because the caching code is obviously correct.
>
> When in doubt, key per user. The extra hit rate is not worth the risk.

---

## Task 3 — `ResponseCache`

### Two independent eviction rules

| Rule | Fires when |
|---|---|
| **TTL** | The entry is older than `ttl_seconds`, however popular |
| **LRU** | The cache is full, and this is the least recently used entry |

They're independent: a heavily-used entry still expires, and a fresh entry still gets evicted if it's the least recent.

### LRU means *least recently used*, not oldest

```python
self.entries.move_to_end(key)     # in get(), on every hit
```

The test constructs the distinction deliberately:

```
[ OK ]  3. evicts the LEAST RECENTLY USED entry
```

Insert `a`, `b`, `c` → **read `a`** → insert `d`. The evicted key must be `b`, not `a`. `a` is the *oldest inserted* but the *most recently used*.

If you skipped `move_to_end` in `get`, you built a FIFO cache and the test catches it. FIFO caches evict exactly the entries that are proving useful.

### Deleting expired entries eagerly

```python
if self.clock() - stored_at >= self.ttl_seconds:
    del self.entries[key]
    self.expirations += 1
    self.misses += 1
    return None
```

You could leave it for LRU to evict. Deleting now is better: **a stale entry occupying a slot is worse than an empty slot**, because it can't be hit and it displaces something that could.

Note it counts as *both* an expiration and a miss. Expirations tell you whether your TTL is too short; misses tell you your hit rate. Different questions.

### `while`, not `if`

```python
while len(self.entries) > self.max_size:
```

With `if`, lowering `max_size` at runtime would only ever evict one entry per `set`. `while` shrinks properly.

### The separate counters

`hits`, `misses`, `evictions`, `expirations` answer four different questions:

| Signal | Tells you |
|---|---|
| Low hit rate | Caching isn't helping — is traffic actually repetitive? |
| High evictions | `max_size` is too small |
| High expirations | `ttl_seconds` is too short |
| Both high | You're thrashing; the cache is costing more than it saves |

A single `hit_rate` would hide all of that.

---

## Task 4 — `BudgetGuard`

### `time.time`, not `time.monotonic`

```python
self.clock = clock or time.time
```

The rest of this lab uses `monotonic` because it measures *elapsed* time reliably. The budget guard needs **wall-clock days**, and `monotonic` has an arbitrary origin — you cannot compute which calendar day it is from it.

The trade-off is real: `time.time` can jump when the system clock is adjusted. For a daily budget that's an acceptable risk; for the circuit breaker in Module 11 it wasn't.

### Integer day arithmetic

```python
def _day(self):
    return int(self.clock() // SECONDS_PER_DAY)
```

Days since the epoch. Simple, and it means "has the day changed?" is an integer comparison rather than date parsing.

**It resets at UTC midnight, not local midnight.** Fine for most purposes; if your billing period is local, you'd need timezone handling.

### Check before, record after

```python
if budget.can_spend(estimated_cost):
    result = call_model(...)
    budget.record(actual_cost)      # the REAL figure
```

You estimate to decide, then record what it actually cost. Module 10 §10.2 showed estimates and actuals differ, and recording the estimate would let error accumulate in whichever direction your estimator is biased.

### `max(0.0, ...)` on `remaining`

```python
return max(0.0, self.daily_limit - self.spent_today)
```

An overspend can slip through — you check with an estimate and record an actual, and actuals can be higher. Returning a negative "remaining" would be confusing in a UI and could break arithmetic downstream.

### The floating-point lesson

Experiment 3 shows something I didn't plan:

```
  after 50 requests: served 49, refused 1, remaining $0.02
```

49 requests served, not 50, with $0.02 apparently left. **Floating-point drift**: adding `0.02` to itself 49 times gives `0.9800000000000005`, not `0.98`, so the 50th request appears to exceed the limit by a fraction of a cent.

Harmless in a budget guard where being slightly conservative is fine. **Not harmless in anything that charges people.**

> **🔑 Never store money as a float.** Use integer minor units (cents) or `decimal.Decimal`. This is a classic bug and it's worth meeting in a lab rather than in production.

I left it in and explained it rather than switching to integers, because the drift is more instructive than its absence.

---

## Task 5 — `preflight_checks`

### Blockers versus warnings

```python
return {"ready": not blockers, ...}      # warnings do NOT block
```

Seven of each, and the split matters more than the specific items.

**Blockers bound your worst case:**

| Blocker | Worst case without it |
|---|---|
| Hardcoded key | The key is in your repo and image, forever |
| `.env` not git-ignored | Same |
| No provider spending cap | **Unbounded bill** |
| No rate limiting | One user or bug exhausts the budget |
| No `max_tokens` | One runaway generation costs more than a thousand normal ones |
| Stack traces returned | Leaks paths, versions, table names |
| Debug mode on | Verbose logs may include secrets |

**Warnings make the product better** but don't turn a bad afternoon into a story.

> **🔑 A checker that blocks on everything gets bypassed entirely, and then it protects nothing.** The split is what keeps it usable — and a tool people route around is worse than no tool, because it creates false confidence.

### Why the messages say *why*

```python
blockers.append(
    "no provider spending cap: an unbounded bill. A retry loop does "
    "as much damage as an attacker")
```

Not `"spending_cap missing"`. **The person reading this output is deciding whether to override it**, and a reason is what stops them.

### Experiment 4's first configuration isn't a straw man

```
  [NOT READY] Friday afternoon demo
              BLOCKER: api key source is 'hardcoded'...
              BLOCKER: no provider spending cap...
              BLOCKER: no rate limiting...
```

Every one of those settings is the **default**:

- You get a hardcoded key by pasting it into your code
- You get no spending cap by not visiting billing
- You get no rate limiting by not writing any
- You get stack traces by not catching anything

**Nothing in that list requires effort to get wrong.** That's why the checklist exists — not because these are exotic mistakes, but because they're what happens when you don't decide.

---

## The experiments — discussion

### Experiment 1: bursts survive, sustained volume doesn't

```
  requests attempted : 300
  allowed            : 69
  blocked            : 231
  saved              : $0.46 (77%)
```

Five minutes of one request per second. Scale that to overnight and it's the difference between a rounding error and a conversation with your finance team.

**The important detail is the first ten.** They all went through. A real user clicking around a demo is completely unaffected, because the burst allowance covers normal human interaction. Only a *sustained* rate gets bounded.

That's what makes token buckets the standard choice: a naive "N requests per minute" counter either blocks real users or permits a burst you can't afford.

### Experiment 2: traffic is skewed

92% hit rate. That looks implausible until you consider real usage:

- Demo visitors click the same example question
- Users ask the obvious things first
- Refreshes and retries repeat requests
- FAQ-style questions dominate any support corpus

**Uniform traffic is a modelling convenience, not a thing that happens.** The skew is exactly what a cache exploits, which is why §13.7 calls it the highest-return optimisation in the module.

### Experiment 3: what "graceful degradation" means

The budget guard refuses, and **the app stays up.** It can serve a cached answer, or a clear message, or a reduced-quality fallback.

Compare the alternative: hitting the *provider's* hard cap. There, every request fails with an API error, your app returns 500s, and nobody notices until a user complains.

**Both limits matter, and they do different jobs:**

| | Provider cap | App budget guard |
|---|---|---|
| Purpose | Bound worst-case loss absolutely | Keep the product usable as you approach it |
| Failure mode | Hard errors | Controlled degradation |
| Can you skip it? | **No** | You shouldn't, but it's less catastrophic |

---

## Part 2 — Discussion

### 2. The out-of-scope question

Your Module 8 bot should say "I don't know" (§8.9). If it invents an answer instead, the grounded prompt needs work — deploying a system that confidently answers questions your documents don't cover is worse than deploying nothing.

### 3. Watching someone else use it

**The most informative fifteen minutes in this lab.** Expect:

- They ask something you never anticipated
- They phrase things differently from you
- They don't read your description
- If there were no examples, they hesitated

That last one is why §13.2 insists on example questions. **A blank input box gets abandoned.**

### 5. Two browsers, same question

Depends entirely on your key.

With `session=session_id` in the key (as in the Part 2 code), they get **separate** cache entries — each browser is a different session. That's safe and it costs you a duplicate call.

Without it, they'd **share**. For a document Q&A bot where answers aren't personalised, sharing is correct and saves real money. For anything user-specific it's a breach.

**The right answer is a judgement about your product**, and the point of the question is that you have to make it deliberately.

`solution.py`'s third demo measures both:

```
  configuration                         API calls      cost    saved
  no protection                             2,000     $4.00        -
  cache keyed per user (safe)                 663     $1.33      67%
  cache shared (if impersonal)                 63     $0.13      97%
```

**The shared cache saves far more, and it's only correct when the answer genuinely doesn't depend on who asked.** That's the trade-off stated honestly: safety costs you 30 percentage points of saving here.

### 6. What breaks first at a thousand users a day

Usually, in this order:

1. **The budget.** 1,000 users × a few questions × $0.002 is $6–20/day
2. **The in-memory rate limiter** — resets on every restart, and doesn't work across instances (§13.8)
3. **The in-memory cache** — same problem
4. **Indexing at startup** — a slow boot, repeated on every deploy and restart

The first fix is usually Redis for both the buckets and the cache, and a persisted vector store rather than re-indexing on boot.

---

## 🚀 Stretch — Discussion

### 1. Logging

The interesting decision is what *not* to log. Question text may contain personal data, and logs get shipped to third parties, retained for years, and read by people the user never considered.

**Length and a hash are usually enough** to spot patterns, detect abuse and compute a `said_dont_know` rate. Log full text only if you've decided to, told users, and set a retention policy (§13.10).

### 4. Verifying the key isn't in the image

```powershell
docker history myapp
```

If `.env` made it in, **a later `RUN rm .env` does not help.** Docker layers are additive, and anyone who pulls the image can extract any layer. Fix `.dockerignore` and rebuild from scratch.

This is worth doing once by hand, because "the file isn't in the final filesystem" feels like it should be sufficient and isn't.

### 5. Streaming

Time to first token drops dramatically; total time barely changes. Module 6 §6.5's point, felt directly.

Then the harder question: **how do you validate output you haven't finished generating?** The usual compromises are to buffer the first sentence and check that, or to stream the answer but withhold sources and actions until the full response has passed validation.

There isn't a clean answer, which is why §13.9 lists it as a cost of streaming rather than a solved problem.

### 6. Breaking your own app

The prompt-injection attempt is the one to pay attention to. Module 11 §11.3: if your app has no tools and validates its output, an injection gets you a weird answer. If it has tools (Module 9), it gets you an action.

**Write down what happened for each.** That list is more valuable than any feature you could add next.

---

## Ready for Module 14?

- [ ] You can explain why a token bucket allows bursts but bounds sustained rate
- [ ] You know why an idle bucket must cap its accumulation
- [ ] You can explain the cache-key mistake that leaks user data
- [ ] You know why LRU means least *recently used*, not oldest
- [ ] You can say why the provider cap and the app budget guard are both needed
- [ ] You know why money should never be stored as a float
- [ ] You can explain why a checker that blocks on everything protects nothing
- [ ] You know why `.env` in `.dockerignore` cannot be fixed by a later `rm`
- [ ] **You have a public URL running your own pipeline**

That last box is the milestone. Two portfolio pieces down.

**Next: Module 14 — Ethics & Limitations.** The final module. You now know how to build these systems and how to ship them; Module 14 is about what you *should* build — bias, intellectual property, environmental cost, and the things GenAI genuinely cannot do.

---

<div align="center">

**[⬅ Back to Lab 13](README.md)** · **[📖 Module 13](../../modules/13-deployment.md)** · **[🏠 README](../../README.md)**

</div>
