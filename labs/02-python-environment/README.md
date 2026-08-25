# 🧪 Lab 2: Build Your Workbench

**Module:** [2 — Python & Your Environment](../../modules/02-python-and-environment.md)

---

## Objective

Set up a working Python environment you'll use for the remaining twelve modules, then write a small prompt toolkit that exercises every Python concept from Module 2.

By the end you will have:

1. **A verified environment** — Python, a virtual environment, packages, and a safely-stored API key
2. **A diagnostic script** you can re-run any time something breaks later
3. **A working prompt toolkit** — five functions, all passing an automated self-test
4. **One real API call** made from your own code
5. *(Stretch)* The same code running against a free local model

## Expected outcome

`python check_setup.py` reports zero failures. `python starter.py` reports **11 of 11 checks passing**. And you've seen a language model answer a question that your own script sent it.

## Requirements

| | |
|---|---|
| **Python** | 3.10 or newer |
| **Packages** | `python-dotenv`, `openai` (Part B needs none at all) |
| **API key** | Optional — only for Part C. A free Ollama path is provided. |
| **Cost** | Free, or ~$0.001 if you use a paid API in Part C |
| **Time** | ~60 minutes |

**Files in this folder:**

| File | What it's for |
|---|---|
| `check_setup.py` | Diagnostic. Run it whenever something breaks. Don't edit it. |
| `starter.py` | **Your work goes here.** Five `TODO`s plus a self-test. |
| `solution.py` | Reference solution. Open after attempting. |
| `SOLUTION.md` | Explanation of each solution and the reasoning behind it. |

> **💡 No local Python?** Everything except Part A works in [Google Colab](https://colab.research.google.com/). Upload `starter.py`, or paste each function into a cell. Part A's venv work is specific to a local machine — read it, skip it, come back if you set one up later.

---

## Part A — Set up and verify (20 min)

### A1. Create and activate a virtual environment

From the **repo root** (not this lab folder):

```powershell
# Windows PowerShell
cd "F:\Programming\AI\prompt-to-production"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
cd /path/to/prompt-to-production
python3 -m venv .venv
source .venv/bin/activate
```

**Check:** your prompt now starts with `(.venv)`.

> **Windows: got "running scripts is disabled on this system"?** Run this once, answer `Y`, then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### A2. Install what you need

```powershell
pip install python-dotenv openai
```

Or install everything for the whole course now (takes a few minutes):

```powershell
pip install -r requirements.txt
```

### A3. Run the diagnostic

```powershell
python labs/02-python-environment/check_setup.py
```

You'll get a report across five areas. **Read it.** Warnings about packages needed by later modules are expected and fine — you're only looking for `[FAIL]`.

Expected at this point:

```
[ OK ]  Python version: 3.12.x
[ OK ]  Virtual environment active: .venv
[ OK ]  python-dotenv: installed
[ OK ]  openai: 1.x.x
[WARN]  .env file: not found          <- fixed in A4
```

### A4. Set up your API key

Even if you plan to use free local models, do this — it establishes the habit.

```powershell
# Windows
Copy-Item .env.example .env
```
```bash
# macOS / Linux
cp .env.example .env
```

Open `.env` and add a key (see [Module 2 §2.10](../../modules/02-python-and-environment.md#210-api-keys-how-not-to-leak-money) for where to get one). No quotes, no spaces:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

> **⚠️ Set a spending limit first.** In your provider's billing settings, cap monthly spend at $5. Do this *before* generating the key. It's the cheapest insurance in the course.

### A5. Prove your key is safe

Two checks. Both matter.

```powershell
git status
```

**`.env` must not appear in the output.** If it does, stop — fix `.gitignore` before you commit anything.

```powershell
python labs/02-python-environment/check_setup.py
```

Look for:

```
[ OK ]  .env file: found
[ OK ]  .env is git-ignored: your key will not be committed
[ OK ]  OPENAI_API_KEY: loaded (sk-proj..., 164 chars)
```

Notice it shows only a 7-character prefix and a length — never the whole key. **That's deliberate, and it's how you should always check a secret.** Printing a full key puts it in your terminal history, your notebook output, and any screenshot you take.

**✅ Part A complete when `check_setup.py` reports 0 failures.**

---

## Part B — Build the prompt toolkit (25 min)

Open `starter.py`. It has five functions, each with a `TODO`. Fill them in.

```powershell
python labs/02-python-environment/starter.py
```

Run it now, before writing anything — you'll see 10 of 11 checks failing. That's your starting point. **Re-run after each function**; the failures tell you exactly what was expected versus what you returned.

### The five tasks

| Task | Function | Practises | Module 2 § |
|---|---|---|---|
| 1 | `build_prompt` | f-strings, default parameter values | 2.5, 2.7 |
| 2 | `estimate_tokens` | `len()`, conditionals, `round()`, `max()` | 2.5, 2.7 |
| 3 | `estimate_cost` | float arithmetic | 2.5 |
| 4 | `build_messages` | **lists of dictionaries** — the API shape | 2.6 |
| 5 | `summarise_batch` | loops, dicts, edge cases, reuse | 2.6, 2.7 |

Task 4 is the one that matters most beyond this lab. That list-of-dicts structure is what you'll pass to every model API for the rest of the course.

Task 5 has a deliberate trap: **what should happen with an empty list?** The self-test checks it. If you hit `ValueError: max() iterable argument is empty`, you've found it — that's the lesson, not a mistake.

### Suggested order

Do them in order — task 5 reuses task 2.

**✅ Part B complete when you see `All 11 checks passed.`**

---

## Part C — Your first API call (10 min)

Create `first_call.py` in the repo root:

```python
"""first_call.py - my first LLM call."""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Import the toolkit you just built. Reuse beats rewriting.
import sys
sys.path.append("labs/02-python-environment")
from starter import build_messages, estimate_tokens, estimate_cost

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("No OPENAI_API_KEY found. Copy .env.example to .env and add your key.")

client = OpenAI()
MODEL = "gpt-4o-mini"    # check platform.openai.com/docs/models for current options

# Build the request using YOUR functions.
messages = build_messages(
    system_prompt="You are a concise tutor. Answer in exactly 2 sentences.",
    user_prompt="What is a token in a language model?",
)

# Estimate the cost BEFORE spending anything - a good habit.
estimated = sum(estimate_tokens(m["content"]) for m in messages)
print(f"Estimated input tokens: {estimated} (~${estimate_cost(estimated):.6f})\n")

# Send it.
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
    max_tokens=150,
)

print(response.choices[0].message.content)
print(f"\n--- actual tokens used: {response.usage.total_tokens} ---")
```

Run it:

```powershell
python first_call.py
```

**Then answer these in your notes:**

1. How close was your estimate to the actual token count? Why the difference?
2. Run it twice. Are the answers identical? Connect this to Lab 1, Experiment 1.
3. Change the system prompt to `"You are a pirate. Answer in exactly 2 sentences."` What changed, and what didn't?

### Free alternative — no key, no cost

Ollama exposes an OpenAI-compatible API, so **two lines change**:

```python
client = OpenAI(
    base_url="http://localhost:11434/v1",   # local Ollama instead of OpenAI
    api_key="ollama",                        # required by the library, ignored by Ollama
)
MODEL = "llama3"                             # a model you've pulled
```

Setup: install from [ollama.com](https://ollama.com/), then `ollama pull llama3`. See Appendix A.

**✅ Part C complete when a model answers a question your script sent.**

---

## 🚀 Stretch Challenges

**Optional.** Nothing later depends on these.

### 1. Compare two models on the same prompt

Send identical `messages` to two different models — a paid one and a local one, or a small model and a larger one. Print both answers side by side, plus token counts.

*What you'll notice:* smaller models are often perfectly adequate for simple tasks at a fraction of the cost. Knowing when you can drop down is a real production skill.

### 2. Add a retry with backoff

API calls fail. Wrap yours in a loop that retries up to 3 times, waiting 1s, then 2s, then 4s.

```python
import time

for attempt in range(3):
    try:
        response = client.chat.completions.create(...)
        break
    except Exception as exc:
        wait = 2 ** attempt          # 1, 2, 4 seconds
        print(f"Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s...")
        time.sleep(wait)
else:
    raise SystemExit("All 3 attempts failed.")
```

*Note the `else` on the `for` loop* — it runs only if the loop finished without `break`. An unusual and genuinely useful Python feature. This is a preview of Module 11.

### 3. Make the toolkit a real CLI

Use the standard-library `argparse` so you can run:

```powershell
python ask.py --topic "attention" --level expert --max-words 50
```

### 4. Count tokens properly

Your `estimate_tokens` uses a 4-characters-per-token approximation. Replace it with the real thing:

```powershell
pip install tiktoken
```

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
exact = len(encoder.encode("Explain embeddings in under 100 words."))
print(exact)
```

Compare exact against your estimate on ten different strings. **Where is the approximation worst?** Try code, non-English text, long numbers, and unusual punctuation. This is Module 3's opening question, and arriving with an answer already in hand will make it click faster.

---

## When you're done

1. Attempt everything before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — it explains the reasoning, not just the code, including why the empty-list case in task 5 is a genuine bug class rather than a puzzle.
3. **Keep `check_setup.py`.** When something breaks in Module 7, run it first.

**Stuck?** Check `appendix/D-troubleshooting.md`, then open an issue. If the environment setup beat you, switch to Colab and carry on — don't let tooling cost you the course.
