# 🧪 Lab 6: Build Your Own Chain Framework

**Module:** [6 — LangChain & Chains](../../modules/06-langchain-chains.md)

---

## Objective

Implement a working chain framework from scratch — the `|` operator, sequential and parallel composition, fallbacks, and three memory strategies — then rebuild the same pipeline in real LangChain and compare.

The point isn't to replace LangChain. It's that **LCEL stops being mysterious once you've written it.** `|` is Python's `__or__` and nothing more.

By the end you will have:

1. **Implemented `Runnable`** with `__or__` and `__ror__` — the whole of LCEL's composition
2. **Built** `RunnableSequence`, `RunnableLambda`, `RunnableParallel`, `RunnablePassthrough`
3. **Added fallbacks** with correct error semantics
4. **Implemented three memory strategies** and measured how their costs diverge
5. **Rebuilt the same pipeline in real LangChain** and compared line for line

## Expected outcome

`python starter.py` reports **29 of 29 checks passing**, then two demos — including a table showing buffer memory's cost growing quadratically while a window stays flat.

## Requirements

| | |
|---|---|
| **Packages** | **None for Part 1** — standard library only |
| **Part 2** | `langchain langchain-core` + `langchain-openai` *or* `langchain-ollama` |
| **API key** | Part 2 only. Free Ollama path provided. |
| **Cost** | Part 1 free · Part 2 ~$0.03 or free |
| **Time** | ~50 minutes |

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 7 tasks, 29-check self-test, 2 demos. |
| `solution.py` | Reference solution + 5 demos, including a `SummaryMemory` bonus. |
| `SOLUTION.md` | The reasoning, including the `__ror__` subtlety. |

---

## Part 1 — Build the framework (35 min)

```powershell
python labs/06-langchain-chains/starter.py
```

### Work in this order

**Task 3 first** — `RunnableLambda`. Everything else depends on it, and the self-test skips the composition checks until it works.

| Task | What you build | Module 6 § |
|---|---|---|
| 3 | `RunnableLambda` — wrap any function | 6.7 |
| 1 | `Runnable.__or__`, `__ror__`, `batch`, and `coerce()` | 6.4 |
| 2 | `RunnableSequence` — run steps in order | 6.4 |
| 4 | `RunnableParallel` — fan out to a dict | 6.8 |
| 5 | `PromptRunnable` — render a template | 6.6 |
| 6 | `RunnableWithFallbacks` | 6.8 |
| 7 | `BufferMemory` and `WindowMemory` | 6.9 |

### Four things worth slowing down for

**`__or__` is the entire trick.** Python calls `a.__or__(b)` for the expression `a | b`. That's all LCEL is. Three lines:

```python
def __or__(self, other):
    return RunnableSequence([self, coerce(other)])
```

**`__ror__` handles the dict-on-the-left case.** This one is genuinely subtle. For the RAG shape:

```python
{"context": retriever, "question": RunnablePassthrough()} | prompt
```

Python tries the **left** operand first — `dict.__or__(prompt)` — which fails, because `dict` has no idea what a `Runnable` is. Python then falls back to the **right** operand's `__ror__`. Without it, that line raises `TypeError`.

Real LangChain defines `__ror__` for exactly this reason. There are two checks for it.

**Flattening.** `RunnableSequence` overrides `__or__` to *extend* itself rather than nest:

```python
a | b | c    # -> Sequence([a, b, c])       with the override
             # -> Sequence([Sequence([a,b]), c])   without it
```

Both work. Flat is far easier to inspect when debugging, which is why real implementations do it.

**Parallel gets the *same* input.** In `RunnableParallel(a=step1, b=step2)`, both steps receive the original input — not each other's output. That's the difference between a fan-out and a sequence, and confusing them produces surprising results.

### Two traps the tests will catch

**`WindowMemory(k=0)`.** In Python, `messages[-0:]` is `messages[0:]` — which returns **everything**. Negating zero doesn't give you "nothing from the end". Handle `k <= 0` explicitly.

**`load()` must return a copy.** There's a check that appends to the returned list and confirms the internal state didn't change. Returning the live list means a caller splicing it into a prompt can silently corrupt your history.

**✅ Part 1 complete at `All 29 checks passed.`**

---

## Part 2 — The same thing in real LangChain (15 min)

Now see how little changes. Create `real_langchain.py` in the repo root:

```powershell
pip install langchain langchain-core langchain-openai
```

```python
"""real_langchain.py - the Part 1 pipeline, in the real framework."""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# --- Flip this one flag to switch the whole pipeline ---
USE_FREE = False

if USE_FREE:
    # Free, local, offline. One-time: install ollama.com, then `ollama pull llama3`
    from langchain_ollama import ChatOllama
    model = ChatOllama(model="llama3", temperature=0.3)
else:
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

print(f"Model: {type(model).__name__}\n")


# ============================================================
# 1. The basic chain - compare to your PromptRunnable | FakeModel
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer."),
    ("human", "Explain {topic} in exactly 2 sentences."),
])

chain = prompt | model | StrOutputParser()
print("--- invoke ---")
print(chain.invoke({"topic": "vector databases"}))


# ============================================================
# 2. What you get free from the shared interface
# ============================================================
print("\n--- batch (concurrent) ---")
for answer in chain.batch([{"topic": "RAG"}, {"topic": "embeddings"}],
                          config={"max_concurrency": 2}):
    print(f"  {answer[:70]}...")

print("\n--- stream (token by token) ---")
for chunk in chain.stream({"topic": "attention"}):
    print(chunk, end="", flush=True)
print()


# ============================================================
# 3. Parallel fan-out - compare to your RunnableParallel
# ============================================================
summary = ChatPromptTemplate.from_template(
    "Summarise in one sentence: {text}") | model | StrOutputParser()
keywords = ChatPromptTemplate.from_template(
    "List 3 keywords, comma-separated, for: {text}") | model | StrOutputParser()

analyse = RunnableParallel(summary=summary, keywords=keywords,
                           original=RunnablePassthrough())

print("\n--- parallel ---")
result = analyse.invoke({"text": "Retrieval-augmented generation grounds an "
                                 "LLM's answers in retrieved documents."})
for key in ("summary", "keywords"):
    print(f"  {key}: {result[key]}")


# ============================================================
# 4. Debugging: print the rendered prompt BEFORE blaming the model
# ============================================================
print("\n--- what the model actually receives ---")
for message in prompt.format_messages(topic="RAG"):
    print(f"  {message.type}: {message.content}")


# ============================================================
# 5. Fallbacks - compare to your RunnableWithFallbacks
# ============================================================
broken = ChatOpenAI(model="does-not-exist-xyz", max_retries=0)
robust = (prompt | broken | StrOutputParser()).with_fallbacks([chain])

print("\n--- fallback ---")
print(robust.invoke({"topic": "chunking"})[:100] + "...")
print("  (primary model does not exist; the fallback answered)")
```

```powershell
python real_langchain.py
```

### Then compare

| Your framework | Real LangChain |
|---|---|
| `PromptRunnable` | `ChatPromptTemplate` |
| `FakeModel` | `ChatOpenAI` / `ChatOllama` |
| `RunnableLambda` | `RunnableLambda` (same name!) |
| `RunnableParallel` | `RunnableParallel` (same) |
| `RunnablePassthrough` | `RunnablePassthrough` (same) |
| `.with_fallbacks()` | `.with_fallbacks()` (same) |
| `.batch()` — sequential | `.batch()` — genuinely concurrent |
| — | `.stream()`, `.ainvoke()`, `.with_retry()`, tracing |

**The names line up because you built the same abstraction.** What the real library adds is genuine concurrency, streaming, async, and the integration catalogue — not a different idea.

### Answer these

1. **Add `StrOutputParser()` to a chain, then remove it.** What type comes back each time? Why does almost every chain end with a parser?
2. **Run `chain.stream()` and `chain.invoke()` on the same input.** Time to *first output*, and time to *complete output* — which differs more?
3. **Break the prompt** — change `{topic}` to `{subject}` but keep invoking with `topic`. What error, and at which stage?
4. **Comment out `max_concurrency`** in the batch call and run a batch of 20. What happens?
5. **Which was easier to understand** — your 200-line framework or the real one? Which would you rather debug at 2am?

**✅ Part 2 complete when you can map every piece of your framework to its LangChain equivalent.**

---

## 🚀 Stretch Challenges

### 1. Add `RunnableBranch` to your framework

```python
class RunnableBranch(Runnable):
    def __init__(self, *branches, default):
        """branches: (condition_fn, runnable) pairs, checked in order."""
```

Then route "code questions" to one chain and everything else to another. **Deterministic routing is cheaper and more predictable than asking a model to route** — reserve model-driven routing for genuinely ambiguous cases.

### 2. Implement `with_retry` with exponential backoff

```python
def with_retry(self, stop_after_attempt=3):
    """Retry TRANSIENT failures with growing waits: 1s, 2s, 4s."""
```

Then answer the design question: **should `with_retry` catch every exception?** (No. Retrying a `KeyError` in your own prompt template three times gets you three identical failures. Retries are for network and rate-limit errors — the ones where waiting helps.)

### 3. Add tracing

Give `RunnableSequence.invoke` an optional `trace=True` that prints each step's input and output with timings. Then run a 4-step chain and find the slow one.

**You've just built the core of LangSmith.** Now you know what it's doing.

### 4. Implement real `SummaryMemory`

`solution.py` has a fake version that concatenates topics. Build the real thing: once history exceeds a threshold, call a model to summarise the oldest turns and replace them.

Then measure: at what conversation length does summary memory become cheaper than buffer memory? **Remember the summarisation call itself costs tokens** — there's a crossover point, and finding it is the actual engineering question.

### 5. Make `batch()` genuinely concurrent

```python
from concurrent.futures import ThreadPoolExecutor

def batch(self, values, max_concurrency=5):
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(self.invoke, values))
```

Time it against the sequential version on 10 fake steps that each `sleep(0.2)`. **Then explain why `max_concurrency` needs a default** rather than being unlimited.

### 6. Break the abstraction on purpose

Find something LangChain makes *harder* than the raw SDK. Candidates: inspecting the exact HTTP payload, using a provider parameter LangChain doesn't expose, or a custom retry policy.

**Then write down where you'd draw the line.** That judgement — when a framework is helping and when it's in the way — is worth more than knowing the API.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — especially the `__ror__` explanation and why fallbacks raise the *last* error.
3. Run `python solution.py` for five demos, including the RAG shape you'll build for real in Module 8.

**Keep your framework.** It's the clearest reference you'll have for what LCEL is doing, and reading real LangChain source becomes much easier after this.

**Next:** Module 7 — Embeddings & Vector Databases, where the `retriever` you faked in the RAG demo becomes real.
