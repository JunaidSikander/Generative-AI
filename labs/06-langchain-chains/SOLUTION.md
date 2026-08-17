# Lab 6 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `Runnable`, `__or__`, `__ror__` and `coerce`

### `__or__` — the whole of LCEL

```python
def __or__(self, other):
    return RunnableSequence([self, coerce(other)])
```

Three lines. Python calls this method for the expression `a | b`, and that is the entire mechanism behind LangChain Expression Language.

There is no parser, no DSL, no code generation. `|` is an operator you can overload, and LangChain overloads it. `solution.py`'s first demo makes this concrete by showing three spellings of the same thing:

```
add1 | double                    -> 4
RunnableSequence([add1, double]) -> 4
add1.__or__(double)              -> 4
```

**If LCEL felt like magic before, it shouldn't now.** That's the whole point of building it.

### `__ror__` — the subtle one

This is the task most people would never think to write, and it's required for the most important pattern in the course.

```python
def __ror__(self, other):
    return RunnableSequence([coerce(other), self])
```

Consider the RAG shape from Module 6 §6.8:

```python
{"context": retriever, "question": RunnablePassthrough()} | prompt
```

Python evaluates `a | b` by trying **the left operand first**: `dict.__or__(prompt)`. Since Python 3.9 `dict` *does* have `__or__` — it's dict merging — but it returns `NotImplemented` for a non-dict operand. Python then falls back to **the right operand's `__ror__`**.

Without `__ror__`:

```
TypeError: unsupported operand type(s) for |: 'dict' and 'PromptRunnable'
```

**Real LangChain defines `__ror__` for exactly this reason.** It's how a plain dict or function can sit on the left of a pipe.

Note the argument order matters: `other` was on the *left*, so it runs *first*. Get it backwards and your chain silently runs in reverse — which is why there's a test for it (`1 → +1 → 2 → add1 → 3 → double → 6`).

### `coerce` — why `chain | some_function` works

```python
def coerce(thing):
    if isinstance(thing, Runnable):
        return thing
    if isinstance(thing, dict):
        return RunnableParallel(**thing)
    if callable(thing):
        return RunnableLambda(thing)
    raise TypeError(...)
```

This is the ergonomics layer. Without it you'd write `chain | RunnableLambda(my_function)` everywhere.

**The dict branch is doing real work.** It's what turns `{"context": ..., "question": ...}` into a parallel fan-out, which is the shape almost every RAG pipeline uses.

**Order matters for readability**, though not correctness here: check `Runnable` first because it's the common case and the cheapest test.

### `batch`

```python
def batch(self, values):
    return [self.invoke(value) for value in values]
```

Ours is sequential; real LangChain runs these concurrently. **The interface is the point.** Callers write `.batch()` once and get whatever concurrency the implementation provides — and if the implementation improves, no caller changes. That's the argument for a shared interface in one line of code.

---

## Task 2 — `RunnableSequence`

```python
def invoke(self, value):
    for step in self.steps:
        value = step.invoke(value)
    return value
```

**Reassigning `value` in a loop is the whole of "chaining".** Everything else in this module is scaffolding around those three lines.

### Why flatten

```python
def __or__(self, other):
    return RunnableSequence(self.steps + [coerce(other)])
```

Without this override, the base `__or__` runs and you get:

```
a | b | c   ->   Sequence([Sequence([a, b]), c])
```

Correct, and awkward. It nests one level deeper per step, so a ten-step chain is ten levels deep. Tracing output becomes unreadable, and `len(chain.steps)` reports `2` for a three-step chain.

Flattening gives `Sequence([a, b, c])`. There's a test for it (`len((add1 | add1 | add1).steps) == 3`), and real implementations flatten for the same reason: **debuggability**.

---

## Task 3 — `RunnableLambda`

```python
def invoke(self, value):
    return self.function(value)
```

Trivial, and the most important class in the framework after `Runnable` itself.

**This is the escape hatch.** Anything the framework doesn't provide, you drop in as a plain Python function. Module 6 §6.7 makes the point: you are never stuck inside the abstraction.

In practice this is where a great deal of real chain code lives — adapters between steps, small transformations, logging taps, custom parsing. The framework handles the plumbing; `RunnableLambda` handles everything the framework didn't anticipate.

---

## Task 4 — `RunnableParallel`

```python
def invoke(self, value):
    return {key: step.invoke(value) for key, step in self.steps.items()}
```

### The critical detail

**Every step gets the SAME input** — not each other's output.

```
SEQUENTIAL (add1 | double), input 3:
  3 -> add1 -> 4 -> double -> 8        each step consumes the previous output

PARALLEL (RunnableParallel(a=add1, b=double)), input 3:
  3 -> both -> {'a': 4, 'b': 6}        each step consumes the ORIGINAL input
```

Note `b` is `6` (3 × 2), not `8` (4 × 2). If you got `8`, you accidentally threaded the output through.

This independence is what makes the steps safely concurrent in a real implementation — there are no data dependencies between them.

### `RunnablePassthrough`

```python
def invoke(self, value):
    return value
```

Two lines, and it enables the RAG shape. In:

```python
{"context": retriever, "question": RunnablePassthrough()}
```

the question goes to **both** the retriever (to find documents) *and* through to the prompt (to be answered). Without a passthrough you'd lose the original input the moment you computed something from it.

---

## Task 5 — `PromptRunnable`

```python
def invoke(self, value):
    return self.template.format(**value)
```

Same design decision as Lab 5, task 2: **let `KeyError` propagate.**

A prompt containing a literal `{topic}` does not crash the model. The model receives the characters `{topic}` and produces something fluent and meaningless. You'd find it days later in your data.

> **🔑 LLM applications fail softly by default.** The model always returns *something*. So the parts of your pipeline that *can* fail loudly should — they're your only source of hard errors in a system that otherwise degrades silently.

### Why `FakeModel` exists

```python
class FakeModel(Runnable):
    def invoke(self, value):
        self.call_count += 1
        return f"{self.prefix}: {value}"
```

Real models are stochastic (Module 3 §3.8), which makes them useless in unit tests — you can't assert on output that changes between runs.

Swapping a deterministic fake in at the model boundary lets you test the chain's **wiring** separately from the model's **behaviour**. Those are genuinely different concerns, and they need different kinds of testing: wiring gets unit tests, behaviour gets an evaluation set (Module 11).

The `call_count` attribute enables a test that would otherwise be impossible: proving the fallback *wasn't* called when the primary succeeded.

---

## Task 6 — `RunnableWithFallbacks`

```python
def invoke(self, value):
    last_error = None
    for candidate in [self.primary, *self.fallbacks]:
        try:
            return candidate.invoke(value)
        except Exception as exc:
            last_error = exc
    raise last_error
```

### Three design decisions worth noticing

**1. `[self.primary, *self.fallbacks]`** treats the primary as just the first candidate. One loop instead of duplicated logic.

**2. Return immediately on success.** The `return` inside the loop means later fallbacks are never touched. There's a test for this — `backup.call_count == 0` when the primary works — because a fallback that runs unnecessarily is a real bug: you'd pay twice and possibly get the worse answer.

**3. Raise the LAST error, not the first.** Deliberate. If your primary hit a rate limit and your backup has an invalid key, the useful message is the backup's — it describes the final state of the system after all recovery attempts. Someone reading the traceback needs to know what stopped you, not what started the cascade.

### Retry versus fallback

Module 6 §6.8 makes this distinction, and it's worth restating:

| Tool | For | Example |
|---|---|---|
| `.with_retry()` | **Transient** — waiting helps | Rate limit, timeout, network blip |
| `.with_fallbacks()` | **Persistent** — waiting won't help | Provider outage, model deprecated, bad key |

Retrying an invalid API key three times gets you three failures and three delays. Falling back on a transient blip gives you degraded output when a retry would have got the good answer. **Match the tool to the failure class.**

---

## Task 7 — Memory

### `BufferMemory`

```python
def save(self, human, ai):
    self.messages.append({"role": "human", "content": human})
    self.messages.append({"role": "ai", "content": ai})

def load(self):
    return list(self.messages)
```

**Why `list(self.messages)` and not `self.messages`?** There's a test that appends to the returned list and asserts the internal state is unchanged.

Returning the live list means any caller can mutate your history. In practice this happens when the returned list gets spliced into a prompt and something appends to it — you've now silently corrupted the conversation. **Returning a copy at a boundary is cheap; debugging aliasing is not.**

### `WindowMemory` and the zero trap

```python
def load(self):
    if self.k <= 0:
        return []
    return list(self.messages[-2 * self.k:])
```

The guard is not defensive padding. In Python:

```python
messages[-0:]     # is messages[0:]  ->  EVERYTHING
```

Negating zero gives zero, and `[0:]` is the whole list. So `WindowMemory(k=0)` without the guard returns the *entire history* — the exact opposite of what it says. There's a test for it, and it's a genuinely nasty off-by-one because it fails in the least intuitive direction.

Slicing *past* the start is safe, which is why `WindowMemory(k=5)` with one exchange just returns that exchange. No guard needed there.

### The cost demo — the real lesson

Run the memory demo and look at the accumulating cost:

```
 turn   buffer msgs   window msgs   buffer token cost
    1             2             2      18 chars sent so far
    2             4             4      54
    3             6             4     108
    4             8             4     180
    5            10             4     270
    6            12             4     378
```

The per-turn increments are 18, 36, 54, 72, 90, 108 — **growing linearly**. Which means the *total* grows **quadratically**.

That's the arithmetic behind Module 6 §6.9's warning. Because every turn re-sends the entire history:

- **Cost per turn** grows linearly with conversation length
- **Total cost** of a conversation grows with the *square* of its length

A 100-turn conversation with unbounded buffer memory doesn't cost 100× a one-turn conversation. It costs roughly 5,000×. This is one of the easiest ways to run up a bill you didn't expect, and it's why "bound your memory" is a day-one decision, not an optimisation.

Window memory stays flat at `2k` messages forever.

### `SummaryMemory` (bonus in `solution.py`)

The production compromise: keep recent turns verbatim, fold older ones into a summary.

```
strategy            messages   chars
buffer                    12     114
window(k=2)                4      38
summary(keep=2)            5      88
```

Bounded like a window, but it retains a trace of what came before. `solution.py`'s version concatenates topics so the demo stays free and deterministic; a real one calls a model to write the summary — which itself costs tokens, so there's a crossover point worth measuring (stretch challenge 4).

---

## Part 2 — Discussion

### The answers

**1. Removing `StrOutputParser()`.** You get an `AIMessage` object instead of a `str`:

```python
AIMessage(content='Vector databases store...', response_metadata={...})
```

Almost every chain ends with a parser because **a chat model returns a message object, not text.** This is the single most common beginner error in LangChain (Module 6's misconceptions list). `StrOutputParser` does `.content` for you and gives the chain a typed output.

**2. Streaming versus invoke.** Time to *complete* output is roughly the same — same tokens, same model. Time to *first* output is dramatically different: streaming shows text in a few hundred milliseconds, `invoke` shows nothing until everything is done.

**This is a user-experience difference, not a performance one**, and it matters more than it sounds. Users tolerate a slow complete answer; they don't tolerate a blank screen. Time-to-first-token dominates perceived speed.

**3. Breaking the prompt.** `KeyError: 'subject'`, raised by the **prompt stage** — before any API call.

That's the good outcome: it fails fast, locally, for free, with the variable name in the message. Compare it to a prompt that silently ships with `{subject}` in the text.

**4. Batch without `max_concurrency`.** With 20 inputs you'll likely hit a `RateLimitError`. `.batch()` fires requests concurrently *by design* — that's the feature — and it's exactly how you trip a provider's rate limit. Pass `config={"max_concurrency": 5}`.

**5. Which is easier to understand?** Your framework — it's 200 lines and you wrote all of them. Which would you rather debug at 2am is a genuinely harder question, and the honest answer is *it depends on the bug*:

- **A wiring bug** — yours, easily. You can read every line.
- **A streaming, async or concurrency bug** — LangChain's, because you'd have to *write* that machinery correctly first, and it's fiddly.

That trade-off is the real content of Module 6 §6.1. **Frameworks trade transparency for capability.** Knowing what you gave up is what lets you decide when the trade is worth it.

---

## 🚀 Stretch — Discussion

### 2. Should `with_retry` catch every exception?

**No**, and this is the interesting part of the exercise.

Retrying is only useful when **waiting changes the outcome**:

| Retry these | Don't retry these |
|---|---|
| `RateLimitError` | `AuthenticationError` — the key won't fix itself |
| Timeouts, connection errors | `KeyError` in your own template |
| Transient `500`s | `BadRequestError` — malformed request |
| Provider overload | Content-filter blocks |

A blanket `except Exception` turns a fast, clear failure into three slow, identical failures. Real implementations take a list of retryable exception types.

### 3. Tracing

Once you've added timing to each step, you've built the core of LangSmith. The insight most people get from doing this: **the slow step is almost never the one you'd guess.** Usually it's a retrieval call or an unnecessary extra model round-trip, not the "big" generation step.

### 5. Why `max_concurrency` needs a default

Unlimited concurrency on a 10,000-item batch means 10,000 simultaneous requests. You will hit rate limits, exhaust local file descriptors, and possibly get temporarily blocked. A sane default (5–10) makes the safe case the default case.

### 6. Where to draw the line

There's no single right answer, and having a considered position matters more than which position:

- **Provider SDK directly** — one or two calls, or when you need exact control over the payload
- **LangChain** — multi-step pipelines, retrieval, streaming with fallbacks, provider portability
- **Mixed** — entirely reasonable, and common in production. Framework for the pipeline, raw SDK for the one awkward step.

If you find yourself fighting the abstraction, that's information. Drop to `RunnableLambda`, or drop out of the framework for that step.

---

## Ready for Module 7?

- [ ] You can explain what `|` does in one sentence, without saying "magic"
- [ ] You can explain why `__ror__` is needed for the RAG shape
- [ ] You know the difference between sequential and parallel input flow
- [ ] You can say when to use `.with_retry()` versus `.with_fallbacks()`
- [ ] You can explain why memory costs grow quadratically over a conversation
- [ ] You know why almost every chain ends with an output parser
- [ ] You have a view on when a framework is worth its abstraction cost

**Next: Module 7 — Embeddings & Vector Databases.** The `fake_retriever` in `solution.py`'s last demo becomes a real one: FAISS and Chroma, approximate nearest-neighbour indexes, and metadata filtering. Then Module 8 assembles it all into the RAG pipeline whose *shape* you already built here.

---

<div align="center">

**[⬅ Back to Lab 6](README.md)** · **[📖 Module 6](../../modules/06-langchain-chains.md)** · **[🏠 README](../../README.md)**

</div>
