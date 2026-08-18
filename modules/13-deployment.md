# Module 13: Deployment Basics

> **By the end of this module** you'll have taken something off your laptop and put it somewhere other people can use — with a UI, secrets handled properly, a spending cap that actually works, caching, rate limiting, and the observability to know what's happening.

| | |
|---|---|
| **Time** | ~2.5 hours (75 min reading, 75 min lab) |
| **Prerequisites** | [Modules 8](08-rag.md), [11](11-guardrails-evaluation.md). Any working app from 8, 9 or 10. |
| **Packages** | `gradio`, `fastapi`, `uvicorn` (Part 1 needs none) |
| **Cost** | Free hosting available; ~$0.10 of API usage |
| **🚀 Milestone** | **This is the course's second portfolio piece.** |

---

## Contents

- [13.0 Why This Matters](#130-why-this-matters)
- [13.1 What "Deployed" Actually Means](#131-what-deployed-actually-means)
- [13.2 Gradio: The Fastest Path](#132-gradio-the-fastest-path)
- [13.3 Streamlit: More Control](#133-streamlit-more-control)
- [13.4 FastAPI: A Real API](#134-fastapi-a-real-api)
- [13.5 Secrets in Production](#135-secrets-in-production)
- [13.6 Cost Control](#136-cost-control)
- [13.7 Caching](#137-caching)
- [13.8 Rate Limiting Your Users](#138-rate-limiting-your-users)
- [13.9 Streaming](#139-streaming)
- [13.10 Observability](#1310-observability)
- [13.11 Where to Host](#1311-where-to-host)
- [13.12 The Pre-Launch Checklist](#1312-the-pre-launch-checklist)
- [🧪 Hands-On Lab 13](#-hands-on-lab-13)
- [✅ Key Takeaways](#-key-takeaways)
- [⚠️ Common Mistakes & Misconceptions](#️-common-mistakes--misconceptions)
- [📚 Going Deeper](#-going-deeper)

---

## 13.0 Why This Matters

Everything you've built runs on your machine, with your API key, for one user: you.

Deployment changes three things at once, and each introduces a failure mode you haven't met:

| On your laptop | Deployed |
|---|---|
| One user (you) | **Many users, some hostile** |
| Your key, your bill | **Their usage, your bill** |
| A crash is an inconvenience | A crash is an outage |
| You see the traceback | **You see nothing unless you built for it** |

The middle row is the one that hurts. **A public app with an API key behind it is a bill with a public "increase" button.** §13.6 and §13.8 are about making that button safe to expose.

The good news: getting a working, shareable UI onto the internet takes about fifteen lines and no infrastructure knowledge. §13.2 does it. The rest of the module is what turns a demo into something you can leave running.

---

## 13.1 What "Deployed" Actually Means

Four distinct things people mean by "deployed", with very different requirements:

| Level | What it is | Effort |
|---|---|---|
| **1. Shareable link** | A URL you send someone. Runs while you're running it. | Minutes |
| **2. Hosted demo** | Always on, public, free tier | An hour |
| **3. Production app** | Auth, rate limits, monitoring, SLA | Days |
| **4. Scaled service** | Autoscaling, multi-region, on-call | Weeks |

**This module gets you solidly to level 2, with the level-3 components built so you understand them.** That's the right target: most learning projects and internal tools live at level 2, and level 3 is mostly about the pieces in §13.6–13.10 rather than about hosting.

### The architecture

```
   ┌──────────┐    ┌───────────────────────────────────┐    ┌──────────┐
   │  BROWSER │───▶│           YOUR APP                │───▶│ LLM API  │
   └──────────┘    │                                   │    └──────────┘
                   │  rate limit  ──▶  cache  ──▶  ... │
                   │      §13.8       §13.7            │
                   │  budget guard ──▶ guardrails      │
                   │      §13.6        Module 11       │
                   └───────────────────────────────────┘
```

Note the order. **Rate limiting and cache lookup come before anything expensive** — the cheapest way to survive a traffic spike is not to make the call.

---

## 13.2 Gradio: The Fastest Path

```powershell
pip install gradio
```

### A working app in fifteen lines

```python
"""app.py - a shareable UI for your RAG bot."""

import gradio as gr
from rag import DocumentQA, load_documents        # your Module 8 bot

qa = DocumentQA()
qa.index(load_documents("documents"))


def answer(question: str) -> tuple[str, str]:
    """Return the answer and its sources."""
    result = qa.answer(question)
    sources = "\n".join(
        f"[{i}] {chunk['metadata']['source']}"
        for i, chunk in enumerate(result["chunks"], start=1)
    )
    return result["answer"], sources


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Question", placeholder="Ask about the documents..."),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="Document Q&A",
    description="Answers come only from the indexed documents, with citations.",
)

if __name__ == "__main__":
    demo.launch()
```

```powershell
python app.py
```

Opens at `http://localhost:7860`. **That's a working app.**

### The share link

```python
demo.launch(share=True)
```

Prints a public `*.gradio.live` URL, valid for 72 hours, tunnelled to your machine.

> **⚠️ `share=True` exposes your laptop to the internet.** The link is unguessable but not authenticated, and it runs your code with your API key. Use it for a quick demo to a colleague; never leave it running unattended.

### Chat interfaces

For anything conversational, `ChatInterface` handles history for you:

```python
def respond(message: str, history: list) -> str:
    """history arrives as a list of prior turns - Module 6's memory, free."""
    result = qa.answer(message)
    return result["answer"]


gr.ChatInterface(
    fn=respond,
    title="Document Q&A",
    examples=["What is the refund policy?", "Who approves expenses?"],
).launch()
```

The `examples` list is worth including. **Most people who open your demo don't know what to ask**, and a blank box gets abandoned.

### Why Gradio first

| ✅ | ❌ |
|---|---|
| Fastest path from function to UI | Limited layout control |
| Free hosting on Hugging Face Spaces | Opinionated styling |
| Built-in components for audio, images, files | Awkward for complex multi-page apps |
| Streaming and queuing built in | — |

---

## 13.3 Streamlit: More Control

```powershell
pip install streamlit
```

```python
"""streamlit_app.py"""

import streamlit as st
from rag import DocumentQA, load_documents

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A")


# @st.cache_resource runs ONCE per session, not per interaction.
# Without it, indexing re-runs on every keystroke - see the warning below.
@st.cache_resource
def load_bot():
    bot = DocumentQA()
    bot.index(load_documents("documents"))
    return bot


qa = load_bot()

with st.sidebar:
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)
    show_sources = st.checkbox("Show sources", value=True)

question = st.text_input("Ask a question")

if question:
    with st.spinner("Thinking..."):
        result = qa.answer(question, top_k=top_k)

    st.markdown(result["answer"])

    if show_sources:
        with st.expander("Sources"):
            for i, chunk in enumerate(result["chunks"], start=1):
                st.caption(f"[{i}] {chunk['metadata']['source']}")
                st.text(chunk["text"][:300])
```

### The one thing to know about Streamlit

> **⚠️ Streamlit re-runs your entire script on every interaction.** Every keystroke, every slider drag.

Without `@st.cache_resource`, that means **re-indexing your documents on every keystroke** — slow, and if indexing calls an embedding API, expensive in a way that looks like nothing is wrong.

| Decorator | Use for |
|---|---|
| `@st.cache_resource` | Models, DB connections, indexes — things you create once |
| `@st.cache_data` | Computed results keyed by inputs |

### Gradio or Streamlit?

| | Gradio | Streamlit |
|---|---|---|
| Fastest to first working app | ✅ | |
| Layout control, dashboards | | ✅ |
| ML demo conventions | ✅ | |
| Multi-page apps | | ✅ |
| Free hosting | Spaces | Streamlit Community Cloud |

**Either is fine.** Pick one and ship; the difference matters far less than shipping.

---

## 13.4 FastAPI: A Real API

When something else needs to call your app — a frontend, a mobile client, another service — you want an API rather than a UI.

```powershell
pip install fastapi uvicorn
```

```python
"""api.py - your pipeline as an HTTP service."""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from rag import DocumentQA, load_documents

app = FastAPI(title="Document Q&A API", version="1.0")

qa = DocumentQA()
qa.index(load_documents("documents"))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool = False


@app.get("/health")
def health():
    """Liveness probe. Hosting platforms poll this."""
    return {"status": "ok", "chunks_indexed": len(qa.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question from the indexed documents."""
    try:
        result = qa.answer(request.question, top_k=request.top_k)
    except Exception as exc:
        # Never leak internals to a caller (Module 9, section 9.12).
        # Log the detail server-side; return something generic.
        raise HTTPException(status_code=500, detail="Internal error") from exc

    return AskResponse(
        answer=result["answer"],
        sources=[Source(source=c["metadata"]["source"],
                        page=c["metadata"].get("page"))
                 for c in result["chunks"]],
    )
```

```powershell
uvicorn api:app --reload
```

Interactive docs appear at `http://localhost:8000/docs`, generated from your Pydantic models.

### What the schema buys you

The `AskRequest` model is doing real work as a guardrail:

| Constraint | Blocks |
|---|---|
| `min_length=1` | Empty questions |
| `max_length=1000` | Context-flooding (Module 11 §11.4) |
| `ge=1, le=10` | `top_k=10000` blowing your context budget and bill |

**Input validation at the API boundary is the cheapest guardrail you have** — it runs in microseconds and rejects malformed requests before anything costs money.

### The `/health` endpoint isn't optional

Hosting platforms poll it to decide whether your container is alive. Without one, a hung process looks healthy and keeps receiving traffic.

---

## 13.5 Secrets in Production

Module 2 §2.10 covered `.env` for local development. **`.env` files do not go to production.**

| Environment | How secrets arrive |
|---|---|
| Local | `.env` file, git-ignored |
| Hugging Face Spaces | Repository secrets, injected as env vars |
| Render / Railway / Fly | Dashboard environment variables |
| Cloud Run / ECS | Secret Manager, injected at runtime |
| Docker (local) | `--env-file`, never `COPY .env` |

The code doesn't change — `os.getenv("OPENAI_API_KEY")` works everywhere. **Only the delivery mechanism differs.**

```python
import os
import sys

REQUIRED = ["OPENAI_API_KEY"]

missing = [name for name in REQUIRED if not os.getenv(name)]
if missing:
    # Fail at STARTUP, not on the first user request.
    sys.exit(f"Missing required environment variables: {missing}")
```

**Check at startup.** A container that boots fine and then fails every request is much harder to diagnose than one that refuses to start with a clear message.

### The rules

| Rule | Why |
|---|---|
| **Never `COPY .env` into a Docker image** | The key is now in a layer anyone who pulls the image can read |
| **Never log the key**, even truncated in error paths | Logs get shipped, indexed and shared |
| **Use a separate key per environment** | You can revoke production without breaking development |
| **Rotate after any exposure** | Deleting the commit does not help (Module 2 §2.10) |
| **Set a spending limit on every key** | The one control that bounds worst-case loss |

> **⚠️ A public app with an unlimited API key is an unbounded liability.** Not because of attackers particularly — a bug that retries in a loop does it just as well. §13.6 is about making that impossible.

---

## 13.6 Cost Control

The section that protects you.

### Four layers, cheapest first

```
   1. PROVIDER SPENDING CAP     hard limit at the account. Set this first.
   2. APPLICATION BUDGET GUARD  your code stops before the cap is hit
   3. RATE LIMITING             bounds requests per user (section 13.8)
   4. CACHING                   avoids paying twice (section 13.7)
```

**Layer 1 is non-negotiable and takes two minutes.** Every provider has a monthly spending limit in billing settings. Set it before you deploy anything public.

### The application budget guard

A hard cap at the provider protects your wallet by breaking your app. A guard in your code lets you degrade gracefully instead:

```python
import time


class BudgetGuard:
    """Track spend against a daily limit, resetting at day boundaries."""

    def __init__(self, daily_limit: float, clock=None):
        if daily_limit <= 0:
            raise ValueError("daily_limit must be positive")
        self.daily_limit = daily_limit
        self.clock = clock or time.time
        self.spent_today = 0.0
        self.current_day = self._day()

    def _day(self) -> int:
        """Which day are we in? Integer days since the epoch."""
        return int(self.clock() // 86_400)

    def _maybe_reset(self) -> None:
        day = self._day()
        if day != self.current_day:
            self.current_day = day
            self.spent_today = 0.0

    def can_spend(self, estimated_cost: float) -> bool:
        """Would this request keep us within budget?"""
        self._maybe_reset()
        return self.spent_today + estimated_cost <= self.daily_limit

    def record(self, actual_cost: float) -> None:
        """Record what a request actually cost."""
        self._maybe_reset()
        self.spent_today += actual_cost

    def remaining(self) -> float:
        self._maybe_reset()
        return max(0.0, self.daily_limit - self.spent_today)
```

**Check before, record after.** You estimate the cost to decide whether to proceed, then record what it actually cost — because Module 10 §10.2 showed estimates and actuals differ.

### Estimating before you spend

```python
def estimate_request_cost(prompt_tokens: int, expected_output_tokens: int,
                          input_price_per_million: float,
                          output_price_per_million: float) -> float:
    """Estimate a request's cost. Output is usually priced higher than input."""
    return ((prompt_tokens / 1_000_000) * input_price_per_million
            + (expected_output_tokens / 1_000_000) * output_price_per_million)
```

### The other levers

| Lever | Saving |
|---|---|
| **Use a smaller model** | Often 10–50× (Module 12 §12.1) |
| **Shorten prompts** | Directly proportional |
| **Retrieve fewer chunks** | Module 8 — re-rank to send less |
| **Cache** | Up to 100% on repeats (§13.7) |
| **`max_tokens`** | Bounds the worst case per request |
| **Downscale images** | Module 10 §10.2 |

**Set `max_tokens` on every call.** Without it a runaway generation can produce thousands of tokens, and output is the expensive half.

---

## 13.7 Caching

**The highest-return optimisation in this module**, because real traffic repeats far more than you'd expect. FAQ-style questions, retries, refreshed pages, the same demo question from twenty visitors.

```python
import hashlib
import json
from collections import OrderedDict


def make_cache_key(model: str, messages: list, temperature: float = 0.0,
                   **extra) -> str:
    """A stable key for a request.

    sort_keys makes the serialisation canonical, so two logically identical
    requests hash the same regardless of dict ordering.
    """
    payload = {"model": model, "messages": messages,
               "temperature": temperature, **extra}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """An LRU cache with TTL expiry."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0,
                 clock=None):
        import time
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.monotonic
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.expirations = 0

    def get(self, key: str):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None

        value, stored_at = entry
        if self.clock() - stored_at >= self.ttl_seconds:
            del self.entries[key]
            self.expirations += 1
            self.misses += 1
            return None

        self.entries.move_to_end(key)     # mark as recently used
        self.hits += 1
        return value

    def set(self, key: str, value) -> None:
        if key in self.entries:
            self.entries.move_to_end(key)
        self.entries[key] = (value, self.clock())

        while len(self.entries) > self.max_size:
            self.entries.popitem(last=False)    # evict least recently used
            self.evictions += 1
```

### When caching is safe

| Safe | Unsafe |
|---|---|
| `temperature=0` — deterministic by intent | High temperature, where variety is the point |
| Public, non-personalised content | **Anything user-specific** |
| Stable knowledge | Rapidly changing data |

> **🚨 The dangerous mistake: a cache key that omits the user.** Cache a personalised answer under a key that doesn't include the user identity and **you will serve one user's data to another.** That is a data breach, not a bug.
>
> If responses depend on who is asking, the user identity **must** be part of the key — or don't cache at all.

### The TTL decision

TTL is how long you're willing to serve a stale answer. Short TTLs mean fewer hits; long ones mean answers outliving the documents behind them.

**For a RAG system, tie cache invalidation to re-indexing.** When documents change, clear the cache — otherwise you're confidently citing a superseded policy.

---

## 13.8 Rate Limiting Your Users

Without this, one user (or one bug) can exhaust your budget in minutes.

### The token bucket

The standard algorithm, and it has a property you want: **it permits short bursts while bounding the sustained rate.**

```
   capacity 5, refill 1/second

   tokens  5 ████████ ─── user makes 5 rapid requests ───▶ 0 ░░░░░░░░
                                                            │
                          refills at 1 token per second      ▼
                                                    after 3s: 3 ████░░░░
```

```python
class TokenBucket:
    """Allow bursts up to `capacity`, sustained at `refill_per_second`."""

    def __init__(self, capacity: float, refill_per_second: float, clock=None):
        import time
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")

        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self.clock = clock or time.monotonic

        self.tokens = float(capacity)      # start full
        self.last_refill = self.clock()

    def _refill(self) -> None:
        """Add tokens for the time elapsed since the last check.

        Lazy refill: no background timer, no thread. Tokens are computed
        from elapsed time whenever someone asks.
        """
        now = self.clock()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity,
                              self.tokens + elapsed * self.refill_per_second)
            self.last_refill = now

    def allow(self, cost: float = 1.0) -> bool:
        """Consume `cost` tokens if available."""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def time_until_available(self, cost: float = 1.0) -> float:
        """Seconds until `cost` tokens will be available. 0.0 if now."""
        self._refill()
        if self.tokens >= cost:
            return 0.0
        return (cost - self.tokens) / self.refill_per_second
```

**`time_until_available` is what makes this usable.** Returning `429 Too Many Requests` with a `Retry-After` header lets a well-behaved client back off correctly instead of hammering you.

### Per-user buckets

```python
from collections import defaultdict

buckets = defaultdict(lambda: TokenBucket(capacity=10, refill_per_second=0.2))


def check_rate_limit(user_id: str) -> tuple[bool, float]:
    bucket = buckets[user_id]
    if bucket.allow():
        return (True, 0.0)
    return (False, bucket.time_until_available())
```

10 requests of burst, then one every 5 seconds sustained.

> **⚠️ An in-memory bucket dict resets on restart and doesn't work across instances.** Fine for a single-process demo; use Redis once you have more than one worker.
>
> And **choose your key deliberately.** IP address is easy and wrong for users behind a shared NAT; an API key or session identifier is better where you have one.

### Weight by cost, not just count

```python
# A request retrieving 10 chunks costs far more than one retrieving 2.
bucket.allow(cost=1.0 + 0.2 * top_k)
```

Counting requests treats a cheap and an expensive one identically. **Weighting by expected cost bounds spend rather than volume**, which is what you actually care about.

---

## 13.9 Streaming

Module 6 §6.5 made the case: **time to first token dominates perceived speed.** A user will wait ten seconds watching text appear; they won't wait four staring at a blank box.

```python
# --- Gradio ---
def stream_answer(message, history):
    partial = ""
    for chunk in qa.stream(message):
        partial += chunk
        yield partial          # yield the accumulated text, not the delta


gr.ChatInterface(fn=stream_answer).launch()
```

```python
# --- FastAPI ---
from fastapi.responses import StreamingResponse


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    def generate():
        for chunk in qa.stream(request.question):
            yield f"data: {chunk}\n\n"      # server-sent events
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### What streaming costs you

| | |
|---|---|
| **Guardrails get harder** | You can't validate output you haven't finished generating |
| **Caching gets harder** | Accumulate the full response before storing it |
| **Error handling gets harder** | A failure mid-stream leaves a partial response on screen |

**The usual compromise:** stream the answer, but validate before showing sources or taking any action. Or buffer the first sentence, run your fast checks on it, then stream the rest.

---

## 13.10 Observability

Module 11 §11.12 listed what to track. Here's how to emit it.

```python
import json
import logging
import time
import uuid

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("app")


def log_request(**fields) -> None:
    """Emit one structured JSON line per request."""
    logger.info(json.dumps(fields, default=str))


def handle(question: str, user_id: str) -> dict:
    request_id = str(uuid.uuid4())
    started = time.monotonic()

    result = qa.answer(question)

    log_request(
        request_id=request_id,
        user_id=user_id,
        # NEVER log the question itself without considering privacy.
        # Length and a hash are usually enough to spot patterns.
        question_length=len(question),
        question_hash=hash(question),
        latency_ms=round((time.monotonic() - started) * 1000),
        cached=result.get("cached", False),
        prompt_tokens=result.get("prompt_tokens"),
        estimated_cost=result.get("cost"),
        said_dont_know="i don't know" in result["answer"].lower(),
        guardrail_triggers=result.get("warnings", []),
    )
    return result
```

### Why structured logs

One JSON object per line means you can query them:

```powershell
# What is the "I don't know" rate?
cat app.log | jq -s 'map(select(.said_dont_know)) | length'
```

Module 11 §11.12: **the `said_dont_know` rate is your cheapest early warning.** A rise means retrieval is degrading or questions have shifted — before any user complains.

### Log privacy

> **⚠️ Logging user questions is a privacy decision, not a technical one.** They may contain personal data, and logs get shipped to third parties, retained for years, and read by people who never asked the user.
>
> Log **length and a hash** by default. Log full text only if you've decided to, told users, and set a retention policy.

### The four signals to alert on

| Signal | Threshold |
|---|---|
| **Error rate** | Above a few percent |
| **p95 latency** | Above your SLA |
| **Daily spend** | Approaching the budget |
| **Rate-limit rejections** | A spike means an attack or a bug |

---

## 13.11 Where to Host

| Platform | Best for | Free tier | Notes |
|---|---|---|---|
| **Hugging Face Spaces** | Gradio/Streamlit demos | ✅ Generous | Simplest path. Git push to deploy. |
| **Streamlit Community Cloud** | Streamlit apps | ✅ | Deploys from GitHub |
| **Render** | FastAPI, containers | ✅ Limited | Free tier sleeps when idle |
| **Railway** | Anything containerised | Trial credit | Very smooth developer experience |
| **Fly.io** | Global, low latency | Small free tier | More configuration |
| **Google Cloud Run** | Serverless containers | ✅ Generous | Scales to zero; production-grade |
| **A VPS** | Full control | ❌ | You manage everything |

**Start with Hugging Face Spaces** for a Gradio app. Push three files and you have a public URL.

### Deploying to Spaces

```
your-space/
├── app.py             your Gradio app
├── requirements.txt   pinned dependencies
└── README.md          with the Spaces YAML header
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
```

Then add `OPENAI_API_KEY` under **Settings → Repository secrets**. It arrives as an environment variable; your `os.getenv` call is unchanged.

### Containerising

When you need portability:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements FIRST so this layer caches when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add a `.dockerignore`:

```
.env
.venv/
__pycache__/
*.pyc
.git/
documents/
chroma_db/
```

> **🚨 `.env` in `.dockerignore` is not optional.** Without it, `COPY . .` bakes your API key into an image layer that anyone who pulls the image can extract. Deleting the file in a later layer does **not** remove it — layers are additive and every one is readable.

---

## 13.12 The Pre-Launch Checklist

Before you share the link.

### Secrets and cost

- [ ] No key hardcoded anywhere (`git grep -i "sk-"` returns nothing)
- [ ] `.env` in `.gitignore` **and** `.dockerignore`
- [ ] Secrets set via the platform, not a file
- [ ] **Provider spending cap set**
- [ ] Application budget guard in place
- [ ] `max_tokens` set on every call

### Abuse and safety

- [ ] Rate limiting per user
- [ ] Input length capped
- [ ] Output validated (Module 11 §11.4)
- [ ] Errors return generic messages, not tracebacks
- [ ] No PII in logs by default

### Operations

- [ ] `/health` endpoint
- [ ] Structured request logging
- [ ] Startup fails loudly on missing configuration
- [ ] Dependencies pinned in `requirements.txt`
- [ ] A fallback response for when the model call fails

### Product

- [ ] Example questions on the landing screen
- [ ] Limitations stated plainly ("answers only from these documents")
- [ ] Loading state — users need to see something happening
- [ ] Sources shown, if the answer is grounded

> **🔑 The four that matter most, if you do nothing else:** provider spending cap, rate limiting, `max_tokens`, and `.env` excluded from your image. Those bound your worst case. Everything else improves the product; those four stop it becoming a story.

---

## 🧪 Hands-On Lab 13

**→ [Go to Lab 13: Ship It Safely](../labs/13-deployment/README.md)**

**🚀 The course's second portfolio milestone.** Implement a token-bucket rate limiter, an LRU cache with TTL, a budget guard and a pre-launch checker — all with injected clocks so they're testable — then deploy a real Gradio app to a public URL.

Part 1 is pure standard library. Budget 75 minutes.

---

## ✅ Key Takeaways

1. **A public app with an API key is a bill with a public "increase" button.** Rate limiting and a spending cap are not optional extras.

2. **Set the provider spending cap first.** Two minutes, and it bounds your worst case absolutely.

3. **Gradio gets you from function to shareable UI in fifteen lines.** Ship, then improve.

4. **Streamlit re-runs your whole script on every interaction.** Without `@st.cache_resource` you re-index on every keystroke.

5. **Pydantic models at an API boundary are guardrails** — length caps and range checks reject bad requests in microseconds.

6. **`.env` files never go to production.** Platform secrets, and `.env` in `.dockerignore`.

7. **Check required configuration at startup**, so a misconfigured container refuses to boot instead of failing every request.

8. **Order the pipeline cheapest-first:** rate limit, then cache, then the expensive call.

9. **Caching is the highest-return optimisation** — real traffic repeats far more than you'd expect.

10. **A cache key that omits the user identity will serve one user's data to another.** That's a breach, not a bug.

11. **Token buckets allow bursts while bounding the sustained rate**, and `time_until_available` lets clients back off properly.

12. **Weight rate limits by cost, not request count**, so an expensive request counts for more.

13. **Set `max_tokens` on every call.** Output is the expensive half.

14. **Log length and a hash, not the question**, unless you've made a deliberate privacy decision.

15. **`.env` in `.dockerignore` is not optional** — Docker layers are additive and every one is readable.

---

## ⚠️ Common Mistakes & Misconceptions

<br>

> ### ❌ Deploying without a provider spending cap
> **Reality:** the single highest-consequence omission in this module. A bug that retries in a loop will do as much damage as an attacker. Two minutes in billing settings bounds your worst case.

<br>

> ### ❌ `COPY . .` in a Dockerfile with no `.dockerignore`
> **Reality:** your API key is now baked into an image layer. Deleting the file in a later layer does not remove it — layers are additive and readable by anyone who pulls the image.

<br>

> ### ❌ Leaving `share=True` running unattended
> **Reality:** an unauthenticated public tunnel to your laptop, running your code with your key. Fine for a five-minute demo; not something to leave overnight.

<br>

> ### ❌ Caching without the user in the key
> **Reality:** you will serve one user's personalised answer to another. This is a data breach. If responses depend on who is asking, the identity must be part of the key.

<br>

> ### ❌ Streamlit without `@st.cache_resource`
> **Reality:** the script re-runs on every keystroke, so you re-index your documents every time someone types a letter. If indexing calls an embedding API, that's a bill with no visible symptom.

<br>

> ### ❌ No `max_tokens`
> **Reality:** a runaway generation can produce thousands of tokens, and output is priced higher than input. One unbounded call can cost more than a thousand normal ones.

<br>

> ### ❌ Rate limiting by request count only
> **Reality:** a request retrieving 20 chunks costs many times one retrieving 2. Weight by expected cost, or your limit bounds volume rather than spend.

<br>

> ### ❌ Rate limiting on IP address
> **Reality:** users behind a shared NAT get one bucket between them; an attacker rotates addresses freely. Use an API key or session identifier where you have one.

<br>

> ### ❌ An in-memory rate limiter behind multiple workers
> **Reality:** each process gets its own bucket, so your effective limit is multiplied by the worker count — and resets on every deploy. Use Redis once you scale past one process.

<br>

> ### ❌ Returning exception details to callers
> **Reality:** stack traces leak file paths, library versions and table names. Log the detail server-side; return something generic (Module 9 §9.12).

<br>

> ### ❌ Logging user questions by default
> **Reality:** a privacy decision, not a technical one. Questions may contain personal data, and logs get shipped, retained and read. Log length and a hash unless you've decided otherwise and told users.

<br>

> ### ❌ No `/health` endpoint
> **Reality:** platforms poll it to decide whether your container is alive. Without one, a hung process looks healthy and keeps receiving traffic.

<br>

> ### ❌ Unpinned dependencies
> **Reality:** your app worked at deploy time and breaks on the next rebuild when a dependency ships a change. Pin versions in `requirements.txt`.

<br>

> ### ❌ Configuration checked on first request rather than at startup
> **Reality:** a container that boots healthy and fails every request is far harder to diagnose than one that refuses to start with a clear message.

<br>

> ### ❌ Shipping a demo with an empty input box
> **Reality:** most visitors don't know what to ask and leave. Example questions cost nothing and are the difference between a demo that lands and one that doesn't.

---

## 📚 Going Deeper

**Frameworks**
- [Gradio docs](https://www.gradio.app/docs) — including `ChatInterface` and streaming
- [Streamlit docs](https://docs.streamlit.io/) — read the caching page first
- [FastAPI docs](https://fastapi.tiangolo.com/) — genuinely excellent
- [LangServe](https://python.langchain.com/docs/langserve/) — deploy LangChain runnables as REST endpoints directly

**Hosting**
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) — the fastest free route
- [Google Cloud Run quickstart](https://cloud.google.com/run/docs/quickstarts) — serverless containers that scale to zero

**Operations**
- [The Twelve-Factor App](https://12factor.net/) — pre-dates all of this and still the clearest statement of config-in-environment
- [AWS: rate limiting strategies](https://aws.amazon.com/builders-library/) — token buckets and beyond

**Local and self-hosted**
- `appendix/A-local-stack.md` — Ollama, LM Studio and Flowise, for deployments that never leave your infrastructure

---

<div align="center">

**[⬅ Module 12](12-fine-tuning.md)** · **[🧪 Do Lab 13](../labs/13-deployment/README.md)** · **[🏠 README](../README.md)** · **➡️ Module 14: Ethics & Limitations** *(coming next)*

</div>
