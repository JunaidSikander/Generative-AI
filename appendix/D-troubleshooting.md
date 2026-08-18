# Appendix D: Troubleshooting & FAQ

> **Something broken? Start here.** Every error below is one learners actually hit working through this course, with the cause and the fix.

**Referenced from:** [Module 2 §2.3](../modules/02-python-and-environment.md#23-virtual-environments-why-before-how) · [Module 3 §3.2](../modules/03-tokens-embeddings-similarity.md#32-tokenization-text-into-pieces) · [README](../README.md)

---

> ### 🔧 Before reading anything else
>
> ```bash
> python labs/02-python-environment/check_setup.py
> ```
>
> It checks your Python version, virtual environment, installed packages and API-key setup, and **prints the exact command to fix each problem it finds**. It has no dependencies and cannot crash. Run it from the repo root.
>
> Roughly three-quarters of the problems on this page are diagnosed by that one command.

---

## Contents

- [Python and environment](#python-and-environment)
- [Installing packages](#installing-packages)
- [API keys](#api-keys)
- [API errors](#api-errors)
- [Encoding and display](#encoding-and-display)
- [Lab self-tests](#lab-self-tests)
- [Vector stores and embeddings](#vector-stores-and-embeddings)
- [LangChain](#langchain)
- [Ollama and local models](#ollama-and-local-models)
- [Notebooks and Colab](#notebooks-and-colab)
- [FAQ](#faq)
- [How to ask a good question](#how-to-ask-a-good-question)

---

## Python and environment

### `python` is not recognised / command not found

Python isn't installed, or isn't on your PATH.

| Platform | Try |
|---|---|
| **Windows** | `py --version` — the Python launcher is often installed when `python` isn't. If it works, use `py` everywhere this course says `python`. |
| **macOS** | `python3 --version`. On macOS, bare `python` frequently doesn't exist. |
| **Linux** | `python3 --version`; install with your package manager if missing. |

If none work, install from [python.org](https://www.python.org/downloads/). **On Windows, tick "Add Python to PATH" in the installer** — that box is the single most common cause of this error.

Then **close and reopen your terminal**. PATH changes don't apply to already-open terminals.

### Wrong Python version

```bash
python --version        # need 3.10 or newer
```

Below 3.10, the type-hint syntax used throughout the labs (`list[str]`, `str | None`) is a `SyntaxError`. Upgrade rather than rewriting the labs.

Several Pythons installed? `where python` (Windows) or `which -a python3` (macOS/Linux) lists them all in PATH order. The first one wins.

### `venv\Scripts\activate` won't run on Windows

```
cannot be loaded because running scripts is disabled on this system
```

PowerShell's execution policy. Fix it for your user only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again. Alternatively use `venv\Scripts\activate.bat` from `cmd.exe`, which isn't subject to the policy.

### Is my virtual environment actually active?

Your prompt should start with `(venv)`. The reliable check:

```bash
python -c "import sys; print(sys.prefix)"
```

If that path doesn't contain your project folder, **the venv is not active** — whatever your prompt says.

> **⚠️ Activation is per-terminal and per-session.** New terminal, new tab, restarted editor, reopened VS Code — activate again. This is the number-one cause of "but I installed it!" (see below).

### I installed the package but Python says it's missing

Nearly always: **you installed into one Python and are running another.**

```bash
python -m pip install package_name
```

`python -m pip` guarantees pip and the interpreter are the same installation. Bare `pip` does not — it's whichever `pip` PATH finds first, which may belong to a different Python entirely.

Confirm where a package landed:

```bash
python -c "import numpy; print(numpy.__file__)"
```

---

## Installing packages

### `pip` is not recognised

```bash
python -m pip --version
```

Use `python -m pip` in place of `pip` everywhere. If that also fails:

```bash
python -m ensurepip --upgrade
```

### Install fails with a compiler error

```
error: Microsoft Visual C++ 14.0 or greater is required
```

A package needs to build from source because no prebuilt wheel matches your Python version — most often because your Python is *newer* than the package supports.

| Fix | Notes |
|---|---|
| **Upgrade pip first** | `python -m pip install --upgrade pip` — old pip misses newer wheels |
| **Use Python 3.11 or 3.12** | Best wheel coverage. Bleeding-edge versions often have none. |
| **Use Colab** | Everything is prebuilt there |

Installing a C++ toolchain works but is a large detour for a course about prompts.

### `faiss-cpu` won't install

Wheel availability lags new Python releases. Options, in order:

1. Use Python 3.11 or 3.12
2. Use Colab for Labs 7 and 8
3. Skip it — **Lab 7 Part 1 is pure NumPy** and teaches the same index concepts. FAISS is only needed for the scale demonstration.

### Install is extremely slow, or hangs

Usually `torch` — it's over 2 GB with CUDA bundled. For this course you don't need GPU support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Roughly a tenth the size. Module 12's lab runs fine on CPU.

### Dependency conflicts

```
ERROR: pip's dependency resolver does not currently take into account...
```

Usually harmless, occasionally not. The clean fix is a fresh venv:

```bash
python -m venv venv-fresh
# activate it, then:
python -m pip install -r requirements.txt
```

> **💡 A virtual environment is disposable.** Delete it and rebuild rather than debugging a tangled one — it takes two minutes and always works. Nothing of yours lives inside it.

---

## API keys

### `AuthenticationError` / `Incorrect API key provided`

Work down this list:

| Check | How |
|---|---|
| **Key exists** | `python labs/02-python-environment/check_setup.py` shows a masked prefix |
| **File is named `.env` exactly** | Not `.env.txt`. Windows hides known extensions — turn that off in File Explorer's View menu |
| **File is in the repo root** | Not in `labs/`, not in `modules/` |
| **`load_dotenv()` is called** | Before you construct the client, not after |
| **No quotes or spaces** | `OPENAI_API_KEY=sk-...` — not `OPENAI_API_KEY = "sk-..."` |
| **Whole key was pasted** | Keys are long; a partial paste is common |
| **Key is still active** | Check the provider dashboard — leaked keys get auto-revoked |

Confirm what Python actually sees:

```python
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
print(f"loaded: {bool(key)}  length: {len(key) if key else 0}")
```

> **⚠️ Never `print(key)` in full**, and never paste one into a chat, issue or screenshot. Length and a short prefix are all you need to diagnose. If you have leaked one, revoke it in the dashboard immediately — revoking is free, and a leaked key gets used within minutes.

### `.env` works in a script but not in a notebook

The notebook's working directory differs. Point at the file explicitly:

```python
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")     # adjust depth to suit
```

### Is my `.env` safe?

```bash
git check-ignore -v .env
```

Output means it's ignored. **No output means it is not** — and if it's already tracked, `.gitignore` won't help:

```bash
git ls-files | grep -E "(^|/)\.env$"
```

Anything listed is tracked. Untrack it, keeping the local file:

```bash
git rm --cached .env
```

> This exact trap existed in the original session material — see [`archive/README.md`](../archive/README.md#changes-made). `.gitignore` only prevents *new* files being added; it does nothing about files git already knows.

---

## API errors

| Error | Meaning | Fix |
|---|---|---|
| `RateLimitError` | Too many requests, too fast | Exponential backoff — [Module 11 §11.10](../modules/11-guardrails-evaluation.md) |
| `insufficient_quota` | No credit, despite a valid key | Add billing. **A free trial key is not the same as credit.** |
| `context_length_exceeded` | Prompt + answer exceed the window | Retrieve less, chunk smaller ([Module 8 §8.4](../modules/08-rag.md)) |
| `model_not_found` | Wrong name, or no access to that model | Check the provider's model list — [Appendix C](C-model-landscape.md) |
| `APIConnectionError` | Network, proxy or firewall | Check connectivity; corporate networks often block API hosts |
| `Timeout` | Model too slow for the deadline | Raise `timeout=`; consider streaming ([Module 13 §13.9](../modules/13-deployment.md)) |

### `insufficient_quota` with credit showing

Two causes, both common:

1. **Credit was added to the wrong organisation.** Multi-org accounts route keys per org — check which org your key belongs to.
2. **Credits expired.** Prepaid API credit typically expires after a year.

### Costs higher than expected

Instrument before you guess — [Module 13 §13.6](../modules/13-deployment.md) has the `BudgetGuard`:

| Cause | Check |
|---|---|
| **Output tokens** | Priced 3–5× input. Long answers dominate. |
| **Retrieved context** | `k=20` costs four times `k=5`, and often answers no better |
| **Full chat history resent** | Every turn re-sends every prior turn ([Module 6 §6.5](../modules/06-langchain-chains.md)) |
| **High-resolution images** | [Module 10 §10.2](../modules/10-multimodal.md) — above a threshold you pay more for nothing |
| **A retry loop** | An agent without an iteration cap can loop expensively ([Module 9 §9.7](../modules/09-agents.md)) |

**Set a hard spending limit in the provider dashboard.** Not a warning threshold — a hard cap.

---

## Encoding and display

### Output shows `�` or `?` instead of proper characters

Windows consoles default to a legacy codepage that can't represent characters like `—` or `→`.

```python
print("cost -> $5")     # ASCII: always works
```

**Every `.py` file in this course is pure ASCII for exactly this reason.** If you add characters of your own and see mojibake:

```bash
chcp 65001              # switch the console to UTF-8
```

Or set `PYTHONUTF8=1` in your environment permanently.

### `UnicodeDecodeError` reading a file

```python
open("data.txt", encoding="utf-8")            # always specify
open("data.txt", encoding="utf-8", errors="replace")   # if the file is genuinely mixed
```

Python on Windows defaults to the system codepage, not UTF-8. **Always pass `encoding="utf-8"` explicitly** — this is a portability bug, not a Windows bug.

---

## Lab self-tests

### A check fails and I can't see why

Every `starter.py` prints the expected and actual value:

```
[FAIL] cosine_similarity: expected 0.7746, got 0.6000
```

Order of attack:

1. **Read the assertion in `starter.py`.** It states the property being tested.
2. **Read that section of the module.** Checks map to sections.
3. **Read `SOLUTION.md`** — it explains reasoning, not just code.
4. **Read `solution.py`** last.

> **💡 Getting a check to fail differently is progress.** A different wrong answer means you moved. Change one thing at a time.

### `ModuleNotFoundError` running a lab

Run from the **repo root**, not from inside the lab folder:

```bash
python labs/03-tokens-embeddings/starter.py       # correct
```

The labs are standalone scripts, but your venv must be active — see [above](#is-my-virtual-environment-actually-active).

### My answer looks right but the check fails

Three near-universal causes:

| Cause | Example |
|---|---|
| **Floating point** | `0.1 + 0.2 != 0.3`. Compare with a tolerance; the checks do. |
| **Ordering** | Search results rank by *score*, not index. Lab 3's `[3, 2]` is not `[2, 3]`. |
| **Off-by-one** | `range(n)` vs `range(1, n+1)` — RRF ranks start at 1. |

### Lab 7's k-means is very slow

You're likely computing distances in a Python loop. Lab 7's `SOLUTION.md` covers this: the matmul form runs about forty times faster and uses a fraction of the memory. **This is the lesson of that lab, not an obstacle to it.**

---

## Vector stores and embeddings

### Search returns nonsense

**The most likely cause is mismatched embedding models.** Query and documents must be embedded by the same model — otherwise the vectors are in different spaces and *nothing errors*, results are just meaningless.

```python
EMBEDDING_MODEL = "text-embedding-3-small"    # one constant, used for both
```

Store the model name in your index metadata. [Module 3 §3.7](../modules/03-tokens-embeddings-similarity.md).

### Shape errors from FAISS

| Error | Cause |
|---|---|
| `AssertionError: d == self.d` | Vector dimension doesn't match the index |
| `ValueError: array must be 2D` | FAISS wants `(n, d)`; reshape a single vector with `.reshape(1, -1)` |
| Silent wrong results | Wrong dtype — FAISS needs `float32`, NumPy defaults to `float64` |

```python
vectors = np.array(vectors, dtype="float32")
```

### Cosine similarity gives strange values

Cosine needs **normalised** vectors. FAISS's `IndexFlatIP` computes a raw dot product — equal to cosine *only* when inputs are unit length:

```python
faiss.normalize_L2(vectors)    # in place, before adding and before searching
```

Normalise the query too. Normalising one side only is a quiet, hard-to-spot bug. [Module 7 §7.4](../modules/07-vector-databases.md).

### Chroma: "collection already exists" or stale results

Chroma persists to disk. Re-running a script appends rather than replacing:

```python
client.delete_collection("my_collection")     # then recreate
```

Duplicate documents are the usual explanation for "my results got worse after re-running the ingest".

### Retrieval finds the wrong chunks

| Symptom | Likely fix |
|---|---|
| Answer is cut off mid-fact | Chunks too small, or no overlap ([Module 8 §8.4](../modules/08-rag.md)) |
| Retrieves vaguely related text | Chunks too large — the signal is diluted |
| Exact IDs/codes never found | Semantic search genuinely fails on these. **Add BM25** ([Module 8 §8.7](../modules/08-rag.md)) |
| Right chunk retrieved, ranked low | Add re-ranking ([Module 8 §8.8](../modules/08-rag.md)) |

> **🔑 Diagnose retrieval before blaming the model.** Print the retrieved chunks. If the answer isn't in them, no model can produce it — and no prompt change will help.

---

## LangChain

### `TypeError: unsupported operand type(s) for |: 'dict' and '...'`

Python tries the **left** operand's `__or__` first. For `{"a": x} | prompt`, that's `dict.__or__`, which fails — so Python falls back to the right operand's `__ror__`. A custom Runnable must implement `__ror__` to sit on the right of a `|`. Lab 6 builds this deliberately; [Lab 6 `SOLUTION.md`](../labs/06-langchain-chains/SOLUTION.md) has the full explanation.

With real LangChain, wrap explicitly:

```python
from langchain_core.runnables import RunnablePassthrough
chain = {"a": RunnablePassthrough()} | prompt | model
```

### Deprecation warnings everywhere

LangChain moved to LCEL. The old API still runs but is deprecated:

| Old | New |
|---|---|
| `LLMChain(llm=..., prompt=...)` | `prompt \| model` |
| `chain.run(x)` | `chain.invoke(x)` |
| `ConversationBufferMemory` | Explicit message-list management |
| `initialize_agent` | LangGraph, or tool calling directly |

Full table in [Module 6 §6.10](../modules/06-langchain-chains.md). **Most tutorials online still show the old API** — that's the main reason for the confusion, not anything you did.

### `ImportError` after upgrading LangChain

The package was split: `langchain-core`, `langchain-community`, `langchain-openai`. Integrations moved out of `langchain` itself.

```python
from langchain_openai import ChatOpenAI              # not langchain.llms
from langchain_core.prompts import ChatPromptTemplate
```

**Pin your versions** — `requirements.txt` does. LangChain moves fast enough that unpinned installs break between sessions.

---

## Ollama and local models

### `Connection refused` on `localhost:11434`

Ollama isn't running.

```bash
ollama serve      # starts it in the foreground
ollama list       # succeeds only if the server is up
```

On macOS and Windows the desktop app usually starts it automatically — check for the menu-bar/tray icon.

### The model is very slow

| Cause | Fix |
|---|---|
| **Model exceeds your RAM** | Use a smaller one. [Appendix A](A-local-stack.md) has the sizing table. |
| **First run** | Weights load into memory once; the second run is much faster |
| **CPU-only** | Expected. A 7B model on CPU runs a few tokens per second. |

**Rule of thumb: model size in GB should be under half your RAM.** Above that, the OS swaps and throughput collapses.

### Local model output is worse than the API

Expected, and honest. Smaller models are weaker at multi-step reasoning and structured output. [Appendix A](A-local-stack.md) has a per-lab viability table — **Lab 9 (agents) is the one that genuinely struggles**, because malformed tool calls break the loop.

If output quality matters more than cost for one lab, spend the few cents on that lab.

### JSON output from a local model won't parse

Common with smaller models. In order of effectiveness:

1. Use Ollama's `format="json"` option
2. Give a complete worked example in the prompt ([Module 5 §5.8](../modules/05-prompt-engineering.md))
3. Extract the first `{...}` block with a regex before parsing
4. Retry once with the parse error fed back in

---

## Notebooks and Colab

### `pip install` in a notebook doesn't take effect

Install into the **running kernel**, then restart it:

```python
import sys
!{sys.executable} -m pip install package_name
```

Bare `!pip install` may target a different Python. After installing, restart the kernel — already-imported modules aren't reloaded.

### Colab loses my files

Colab sessions are ephemeral; disconnection deletes everything not in Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

For keys, use Colab's **Secrets** panel (🔑 in the sidebar) rather than a `.env`:

```python
from google.colab import userdata
key = userdata.get('OPENAI_API_KEY')
```

> **⚠️ Never hardcode a key in a notebook cell.** Notebooks get shared, committed and screenshotted with output intact. Colab Secrets exist precisely for this.

### Variables are undefined even though I defined them

You ran cells out of order, or restarted the kernel. **Run → Restart and run all** is the reliable reset, and how you should verify a notebook before sharing it.

---

## FAQ

**Do I need a GPU?**
No. Everything in the course runs on CPU. Module 12's LoRA lab uses a model small enough for CPU, and Colab provides a free GPU if you want to go bigger.

**Can I do this entirely free?**
Yes. Local models via Ollama ([Appendix A](A-local-stack.md)) cover almost everything, and Part 1 of every lab needs no API at all. Lab 9 is the one place a hosted model is noticeably better.

**How much will the API route cost?**
Roughly $5–15 for the whole course if you use the small/workhorse tier. Set a hard spending cap on day one regardless.

**Can I skip modules?**
Not comfortably — the dependencies are real. Module 8 needs 7, which needs 3. The shortest honest path to a working RAG app is 1 → 2 → 3 → 5 → 7 → 8.

**Which provider should I use?**
Any of the main ones works; the labs use a single `MODEL` constant so switching is one line. [Appendix C](C-model-landscape.md) covers how to choose.

**The model names in the labs look outdated.**
Likely true — they age fast. Every lab keeps the name in one constant at the top. Check the provider's current list ([Appendix C](C-model-landscape.md)) and change that line.

**Can I use this material to teach?**
Yes — content is CC BY 4.0, code is MIT. Each module is roughly one 90-minute session.

**Why does my output differ from the module's?**
Models are non-deterministic. Same prompt, different words. `temperature=0` reduces variation but does not eliminate it ([Module 3 §3.8](../modules/03-tokens-embeddings-similarity.md)). Judge by whether the output is *correct*, not whether it matches character for character.

**Where do I go after Module 14?**
Module 14's wrap-up lists next steps. The short version: rebuild one of the portfolio projects on a problem you actually have. That teaches more than another course.

---

## How to ask a good question

If nothing here helps, open an issue. Include:

| Include | Why |
|---|---|
| **The full error**, last traceback line included | The last line names the actual failure |
| **`check_setup.py` output** | Answers the environment questions in advance |
| **OS and Python version** | Many issues are platform-specific |
| **The command you ran** | Often the problem is the working directory |
| **What you expected vs got** | Distinguishes a bug from a misunderstanding |

**Never include your API key** — not even partially, not in a screenshot, not in a traceback you pasted without reading. If you already have, revoke it now. It costs nothing and takes ten seconds.

> Corrections to the course are welcome and valuable. If an explanation confused you, it will confuse the next person — that's a bug in the material, not in you.

---

<div align="center">

**[🏠 Course README](../README.md)** · **[💻 Local stack](A-local-stack.md)** · **[📖 Glossary](B-glossary.md)** · **[🗺️ Model landscape](C-model-landscape.md)**

</div>
