# Appendix A: Running Models Locally

> **The free, private route.** Every lab in this course has a local path — no API key, no per-token cost, no data leaving your machine. This appendix sets it up.

**Referenced from:** [Module 2 §2.11](../modules/02-python-and-environment.md#211-your-first-api-call) · [Module 6 §6.2](../modules/06-langchain-chains.md#62-what-langchain-gives-you) · [Module 13 §13.11](../modules/13-deployment.md#1311-where-to-host)

---

## Contents

- [Why run locally](#why-run-locally)
- [Ollama — the one to start with](#ollama--the-one-to-start-with)
- [Choosing a model](#choosing-a-model)
- [Using Ollama in this course](#using-ollama-in-this-course)
- [LM Studio — a GUI alternative](#lm-studio--a-gui-alternative)
- [Local embeddings](#local-embeddings)
- [Hugging Face — where the models come from](#hugging-face--where-the-models-come-from)
- [Flowise and LangServe](#flowise-and-langserve)
- [The full local stack](#the-full-local-stack)
- [What you give up](#what-you-give-up)
- [Troubleshooting](#troubleshooting)

---

## Why run locally

| Reason | Detail |
|---|---|
| **Privacy** | Prompts and documents never leave your machine. Relevant for health data, legal work, IP, and anything regulated. |
| **Cost** | Free after the download. Iterate and load-test without metering. |
| **Offline** | Works on a plane, in an air-gapped environment, during a provider outage. |
| **Control** | Choose the model, set the system prompt, pin the version. Nothing changes under you. |
| **Learning** | Experiment without watching a bill. |

That last one matters for this course specifically. **You can complete every lab for free**, and the labs are designed so the core work needs no model at all.

---

## Ollama — the one to start with

A CLI-first runtime: pull a model once, and Ollama runs it as a background service exposing an HTTP API.

```
   your code  ──▶  localhost:11434  ──▶  Ollama daemon  ──▶  model weights
                   (OpenAI-compatible)                       (~/.ollama)
```

### Install

Download from [ollama.com](https://ollama.com) — macOS, Windows and Linux. It installs a background service that starts automatically.

```powershell
ollama --version
```

### Pull a model

```powershell
ollama pull llama3
```

A few gigabytes, once. Cached in `~/.ollama` and reused.

### Chat from the terminal

```powershell
ollama run llama3
>>> Explain RAG in one sentence.
```

`/bye` to exit.

### Call it from code

The important part. **Ollama exposes an OpenAI-compatible endpoint**, so course code works with two lines changed:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",   # local, not OpenAI
    api_key="ollama",                        # required by the library, ignored by Ollama
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "What is a token?"}],
)
print(response.choices[0].message.content)
```

That compatibility is why every lab in this course has a free path — the same code runs against either.

---

## Choosing a model

The main constraint is memory. A rough guide:

| RAM / VRAM | Model size | Expect |
|---|---|---|
| **8 GB** | 3B | Fast; noticeably weaker reasoning |
| **16 GB** | 7–8B | **The sweet spot.** Good enough for most labs. |
| **32 GB** | 13–14B | Better reasoning, slower |
| **64 GB+** | 30B+ | Approaching hosted quality, much slower |

```powershell
ollama pull llama3.2:3b        # small and quick
ollama pull llama3             # 8B - the default recommendation
ollama pull qwen2.5:7b         # strong at structured output
ollama pull mistral            # solid general-purpose
ollama pull nomic-embed-text   # embeddings, not chat
```

```powershell
ollama list      # what you have
ollama rm <name> # reclaim disk space
```

> **💡 Start with an 8B model.** If it's too slow, drop to 3B before concluding local models don't work. Speed differences between sizes are large.

### Quantization

Model names sometimes carry a suffix like `q4_0` or `q8_0` — how aggressively the weights were compressed (the same idea as Module 12 §12.5's QLoRA).

| Level | Size | Quality |
|---|---|---|
| `q4` | Smallest | Slight degradation, usually acceptable |
| `q5`–`q6` | Middle | A good balance |
| `q8` | Largest | Near-original |

Ollama's defaults are sensible. Only reach for these when you're fitting a specific model into specific memory.

---

## Using Ollama in this course

### The one-flag switch

Adopt this pattern early — it's what Module 6 §6.2 recommends:

```python
import os
from dotenv import load_dotenv

load_dotenv()

USE_FREE = True          # flip this one line

if USE_FREE:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = "llama3"
else:
    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-4o-mini"
```

### With LangChain

```powershell
pip install langchain-ollama
```

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3", temperature=0.3)
chain = prompt | model | StrOutputParser()      # Module 6, unchanged
```

### Which labs work well locally

| Lab | Local? | Notes |
|---|---|---|
| 1–4, 7, 11, 12, 14 | ✅ **No model needed** | The core work is pure Python or NumPy |
| 3 (embeddings), 8 (RAG) | ✅ Fully local | `sentence-transformers` runs on CPU |
| 5 (prompting) | ✅ Works | Weaker format-following makes the experiment *more* interesting |
| 6 (chains) | ✅ Works | |
| 9 (agents) | ⚠️ Variable | Tool calling is where small models struggle most |
| 10 (multimodal) | ⚠️ Needs a vision model | `ollama pull llava` |
| 13 (deployment) | ✅ Works | Cost controls become moot, which is itself instructive |

> **📌 Lab 9 is the honest exception.** Smaller models produce malformed tool calls and pick the wrong tool more often. That makes your validation layer *visibly* useful — but expect a rougher ride than the lab text describes.

---

## LM Studio — a GUI alternative

A desktop app over the same engines. No terminal required.

**Good for:** browsing and comparing models before committing to a download, seeing estimated VRAM up front, chatting with parameter sliders, and side-by-side quality comparisons.

It also runs an OpenAI-compatible server on port **1234**:

```python
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
```

### Ollama or LM Studio?

| | Ollama | LM Studio |
|---|---|---|
| Interface | CLI + API | GUI |
| Scripting and CI | ✅ | Awkward |
| Discovering models | Curated list | **Browse everything** |
| Comparing quantizations | Manual | **Built in** |
| Course labs | ✅ Recommended | Works, change the port |

**They complement each other:** LM Studio to explore and evaluate, Ollama to script and serve.

---

## Local embeddings

Modules 3, 7 and 8 need embeddings. Two free local options.

### sentence-transformers (what the labs use)

```powershell
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # ~90 MB, CPU, offline
embeddings = model.encode(documents, normalize_embeddings=True)
```

Downloads once, then works offline. **384 dimensions**, which Module 7 §7.11 notes is a genuine sweet spot rather than a compromise.

### Ollama embeddings

```powershell
ollama pull nomic-embed-text
```

```python
response = client.embeddings.create(model="nomic-embed-text", input=["some text"])
```

> **⚠️ Module 3 §3.7 applies with full force:** query and documents **must** be embedded by the same model. Switching between `all-MiniLM-L6-v2` and `nomic-embed-text` means re-indexing everything. Nothing errors — results just become nonsense.

---

## Hugging Face — where the models come from

The registry the rest of the ecosystem builds on. Ollama and LM Studio both pull from it.

| What | Use for |
|---|---|
| **The Hub** | Versioned repos for models and datasets, each with a model card giving licence, training data and benchmarks |
| **Libraries** | `transformers`, `datasets`, `peft` (Module 12), `sentence-transformers` |
| **Spaces** | Hosted demo apps — Module 13's deployment target |
| **Leaderboards** | [MTEB](https://huggingface.co/spaces/mteb/leaderboard) for embeddings; various for chat models |

> **💡 Read the model card before you commit to a model.** Licence terms vary enormously — some open-weight models restrict commercial use or impose conditions above a user threshold. "Open weights" is not the same as "open source", and the difference can matter commercially.

---

## Flowise and LangServe

Two ways to go from a working chain to something other people can use.

### Flowise — visual chain building

```powershell
npx flowise start
```

Opens at `localhost:3000`. Drag-and-drop nodes for chains, prompts, memory, retrieval and tools, built on LangChain. Point a `ChatOllama` node at your local model and you have a chatbot with no code.

| ✅ | ❌ |
|---|---|
| Fast to prototype; good for showing non-developers | Hard to version-control or test |
| Exports as an embeddable widget or API | You'll outgrow it for anything complex |

**Genuinely useful for exploration**, and Module 6's argument applies: when you're fighting the abstraction, drop to code.

### LangServe — chains as REST APIs

```powershell
pip install "langserve[all]"
```

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()
add_routes(app, my_chain, path="/chat")     # auto schema, docs, playground
```

Built on FastAPI, with streaming and batching. **If your app is already a LangChain runnable, this is the shortest path to serving it** — an alternative to hand-writing the FastAPI layer in Module 13 §13.4.

---

## The full local stack

```
   ┌────────────────────────────────────────────────────────┐
   │  INTERFACE      Gradio · Streamlit · HuggingChat        │
   │                 what your users touch                  │
   ├────────────────────────────────────────────────────────┤
   │  ORCHESTRATION  LangChain · Flowise · LangServe         │
   │                 chains, prompts, memory, retrieval      │
   ├────────────────────────────────────────────────────────┤
   │  RETRIEVAL      FAISS · Chroma · sentence-transformers  │
   │                 embeddings and vector search            │
   ├────────────────────────────────────────────────────────┤
   │  RUNTIME        Ollama · LM Studio                      │
   │                 the model itself                        │
   └────────────────────────────────────────────────────────┘
```

**Every layer has a free, local, open option.** A complete RAG system where no data leaves your machine is entirely achievable with what's in this appendix — which is the right answer for some problems, not just the cheap one.

---

## What you give up

Honest accounting.

| | Hosted API | Local |
|---|---|---|
| **Quality** | Frontier | Noticeably weaker on complex reasoning |
| **Speed** | Fast | Depends on your hardware; often much slower |
| **Setup** | An API key | A download and some configuration |
| **Cost** | Per token | Free after hardware |
| **Privacy** | Data leaves | **Data stays** |
| **Availability** | Their uptime | Yours |
| **Tool calling** | Reliable | **Variable** — the biggest practical gap |
| **Context window** | Large | Smaller, and slower when full |

### Where the gap is widest

1. **Tool calling and structured output** — small models produce malformed JSON and pick wrong tools more often. This is why Lab 9 is flagged above.
2. **Multi-step reasoning** — the gap widens with problem complexity.
3. **Long context** — smaller windows, and quality degrades faster as you fill them.

### Where it's narrowest

1. **Summarisation and rewriting**
2. **Classification with clear categories**
3. **Simple extraction**
4. **RAG answering** — retrieval does the hard part, and the model mostly rephrases what it was given

> **🔑 The narrow-gap list is exactly Module 12 §12.1's argument for small models.** For a great many production tasks, a local 8B model is genuinely sufficient — and Module 11's evaluation harness is how you find out rather than guess.

---

## Troubleshooting

**`Connection refused` on port 11434**

The service isn't running.

```powershell
ollama list      # if this fails, the daemon is down
ollama serve     # start it in the foreground
```

On Windows, check whether Ollama is in the system tray.

**Very slow generation**

- Try a smaller model (`llama3.2:3b`)
- Check whether the GPU is being used — `ollama ps` shows CPU/GPU split
- Close memory-hungry applications; a model swapping to disk is drastically slower

**`model not found`**

```powershell
ollama list                 # exact names, including any :tag
ollama pull llama3          # pull it first
```

Model names are exact: `llama3` and `llama3.2` are different models.

**Out of memory**

Drop a size tier, or a quantization level. A 7B model needs roughly 8 GB of free memory at typical quantization.

**Ollama works in the terminal but not from Python**

Almost always the `base_url`. It needs the `/v1` suffix:

```python
base_url="http://localhost:11434/v1"     # ✅
base_url="http://localhost:11434"        # ❌
```

**Different answers than the lab text describes**

Expected. The labs were written against hosted models, and local models differ — particularly on format-following. **The self-tests never depend on model output**, so Part 1 of every lab passes identically either way.

---

<div align="center">

**[🏠 Course README](../README.md)** · **[📖 Glossary](B-glossary.md)** · **[🔧 Troubleshooting](D-troubleshooting.md)**

</div>
