# 🧪 Lab 8: Build a Document Q&A Bot

**Module:** [8 — Retrieval-Augmented Generation](../../modules/08-rag.md)

> ### 🏗️ This is the course's first portfolio milestone.
> By the end you'll have a working system that answers questions about **your own documents**, cites its sources, and says "I don't know" rather than inventing an answer. That's something you can show people.

---

## Objective

Implement every piece of a RAG pipeline from scratch, then assemble them over real PDFs and **measure whether retrieval is actually working.**

By the end you will have:

1. **Implemented two chunking strategies** and seen where each one cuts
2. **Implemented BM25** keyword scoring from the formula
3. **Implemented reciprocal rank fusion** — the algorithm that makes hybrid search possible
4. **Built a grounded prompt** with citations and an "I don't know" escape hatch
5. **Written citation validation** that catches the model answering from memory
6. **Assembled a working RAG bot** over your own documents
7. **Measured retrieval quality** with an evaluation set

## Expected outcome

`python starter.py` reports **40 of 40 checks passing**, then three experiments. Part 2 gives you a `rag.py` that answers questions about your PDFs with verifiable citations.

## Requirements

| | |
|---|---|
| **Part 1** | **Standard library only.** No packages, no API key. |
| **Part 2** | `sentence-transformers`, `pypdf`, `numpy`, `openai` |
| **API key** | Part 2 only — free Ollama path provided |
| **Cost** | Part 1 free · Part 2 ~$0.10 |
| **Time** | ~90 minutes |

```powershell
pip install sentence-transformers pypdf numpy openai python-dotenv
```

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 7 tasks, 40-check self-test, 3 experiments. |
| `solution.py` | Reference solution + 4 demos. |
| `SOLUTION.md` | The reasoning, and what the experiments are really showing. |

---

## Part 1 — The RAG toolkit (45 min)

```powershell
python labs/08-rag/starter.py
```

| Task | Function | Key idea | Module 8 § |
|---|---|---|---|
| 1 | `chunk_fixed` | Fixed-size splitting, with an infinite-loop guard | 8.4 |
| 2 | `chunk_recursive` | Respect natural boundaries — **the default strategy** | 8.4 |
| 3 | `tokenize_words` | Crude tokenisation for BM25 | 8.6 |
| 4 | `bm25_scores` | Keyword relevance from the formula | 8.6 |
| 5 | `reciprocal_rank_fusion` | **How hybrid search actually works** | 8.6 |
| 6 | `build_grounded_prompt` | "Only from context", with citation markers | 8.9 |
| 7 | `extract_cited_indices` + `validate_citations` | Catch uncited claims | 8.10 |

### Four things to watch for

**Task 1 — guard `overlap >= chunk_size`.** The step becomes zero or negative. Depending on how you write the loop, that either yields nothing or **hangs forever**. There's a test that expects `ValueError`. A config typo shouldn't be able to freeze a pipeline.

**Task 2 — handle `""` before calling `.split()`.** In Python, `text.split("")` raises `ValueError: empty separator`. The empty separator means "hard split", so check for it *before* you split.

**Task 4 — use `set(tokens)` for document frequency.** A term appearing five times in one document must count as **one** document, not five. Get this wrong and IDF is silently wrong for every common term.

**Task 5 — `enumerate(ranking, start=1)`.** Ranks are 1-based. With `start=0`, the top result gets `1/(k+0)` — which divides by zero when `k=0`, and over-weights rank 1 generally.

### Task 7 is the interesting one

`validate_citations` catches two distinct failures:

- **An answer with no citations that isn't "I don't know."** This usually means the model ignored your context and answered from training knowledge — the exact failure RAG exists to prevent.
- **A marker outside the supplied range.** The model cited `[7]` when you gave it 5 chunks.

Both cost one regex to detect, and both are invisible otherwise.

**✅ Part 1 complete at `All 40 checks passed.`**

---

## The three experiments

### Experiment 1 — where chunking cuts

Compare fixed and recursive splitting on the same document. Fixed splits mid-word; recursive respects paragraph and word boundaries. **Look at the actual chunk contents**, not just the counts.

### Experiment 2 — what keyword search can and cannot do

```
  query type          correct chunk on top?      score
  exact identifier    yes                         3.67     <- SKU-4471
  exact identifier    yes                         1.90     <- E1042
  shares key terms    yes                         3.94
  partial overlap     yes                         2.34
  pure paraphrase     NO - wrong chunk            1.00     <- the failure
```

**This experiment is deliberately half the argument.** BM25 nails exact identifiers — precisely where semantic search failed you in Lab 3 — and fails on pure paraphrase, precisely where semantic search excels. Part 2 supplies the other half with real embeddings.

*(An earlier draft faked the semantic half with word-overlap scoring. It "worked" — and it was rigged, because word overlap is just keyword matching under another name. Half an honest demonstration beats a whole dishonest one.)*

### Experiment 3 — grounded prompts and citation validation

```
  answer type                             valid   problems
  well-grounded                            True   -
  no citations (answering from memory?)   False   answer makes claims but cites no sources
  invented a source                       False   cited non-existent sources: [7]
  correct refusal                          True   -
```

Four plausible model responses, and your validator's verdict on each. The second row is the one worth internalising.

---

## Part 2 — The real bot (45 min)

### Step 1: get some documents

Put 2–5 PDFs or text files in a `documents/` folder at the repo root. **Use documents you actually know**, so you can tell when the answers are wrong — a company handbook, a paper you've read, your own notes.

> **💡 No PDFs handy?** Save a few Wikipedia pages as `.txt`, or use this repo's own module files. Testing against content you know well is the whole point.

### Step 2: build it

Create `rag.py` in the repo root:

```python
"""rag.py - a document Q&A bot with citations."""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

sys.path.append("labs/08-rag")
from starter import (chunk_recursive, bm25_scores, reciprocal_rank_fusion,
                     build_grounded_prompt, validate_citations)

load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

USE_FREE = False
if USE_FREE:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = "llama3"
else:
    client = OpenAI()
    MODEL = "gpt-4o-mini"

RAG_SYSTEM_PROMPT = """You are a documentation assistant. You answer questions
strictly from the provided context.

RULES:
- Use ONLY information present in the context. Never add outside knowledge.
- Cite every claim with the bracketed number of its source, e.g. [2].
- If the context does not answer the question, reply exactly: I don't know.
- If sources conflict, say so and cite both.
- Quote exact figures, names and dates rather than paraphrasing them.
"""


# ---------- LOADING ----------

def load_documents(directory: str) -> list:
    """Load PDFs and text files into {text, metadata} records."""
    records = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    records.append({"text": text,
                                    "metadata": {"source": path.name,
                                                 "page": page_number}})
        elif path.suffix.lower() in {".txt", ".md"}:
            records.append({
                "text": path.read_text(encoding="utf-8", errors="replace"),
                "metadata": {"source": path.name, "page": None},
            })
    return records


# ---------- THE PIPELINE ----------

class DocumentQA:
    def __init__(self, chunk_size: int = 600):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None

    def index(self, records: list) -> None:
        for record in records:
            for piece in chunk_recursive(record["text"], self.chunk_size):
                self.chunks.append({"text": piece, "metadata": record["metadata"]})

        texts = [c["text"] for c in self.chunks]
        self.embeddings = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True)
        print(f"Indexed {len(self.chunks)} chunks from {len(records)} records")

    def retrieve(self, question: str, top_k: int = 4, fetch_k: int = 20) -> list:
        """Hybrid retrieval: dense + BM25, fused by rank."""
        texts = [c["text"] for c in self.chunks]

        query_vector = embed_model.encode(question, normalize_embeddings=True)
        dense = list(np.argsort(self.embeddings @ query_vector)[::-1][:fetch_k])
        sparse = list(np.argsort(bm25_scores(question, texts))[::-1][:fetch_k])

        fused = reciprocal_rank_fusion([dense, sparse])[:top_k]
        return [self.chunks[i] for i in fused]

    def answer(self, question: str, top_k: int = 4) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {"answer": "I don't know.", "sources": [], "warnings": []}

        prompt = build_grounded_prompt(question, [c["text"] for c in retrieved])

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0,          # grounded extraction, not creativity
        )
        answer = response.choices[0].message.content

        _, warnings = validate_citations(answer, len(retrieved))
        return {"answer": answer, "chunks": retrieved, "warnings": warnings}


if __name__ == "__main__":
    qa = DocumentQA()
    records = load_documents("documents")

    # ALWAYS check what the loader extracted before indexing (section 8.3).
    print(f"\nLoaded {len(records)} records. First 300 characters of the first:")
    print(repr(records[0]["text"][:300]) if records else "  (nothing loaded!)")
    print()

    qa.index(records)

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break

        result = qa.answer(question)
        print(f"\n{result['answer']}\n")

        for i, chunk in enumerate(result["chunks"], start=1):
            meta = chunk["metadata"]
            page = f" p.{meta['page']}" if meta.get("page") else ""
            print(f"  [{i}] {meta['source']}{page}")
            print(f"      \"{chunk['text'][:120].strip()}...\"")

        if result["warnings"]:
            print(f"\n  WARNING: {result['warnings']}")
```

```powershell
python rag.py
```

### Step 3: make it fail

This is the actual work. Try:

1. **A question your documents clearly answer.** Does it get it right? Are the citations correct — actually check them.
2. **A question your documents don't cover.** Does it say "I don't know", or invent something? If it invents, you get a warning from `validate_citations` — good.
3. **A question needing information from two documents.** Does retrieval find both?
4. **An exact identifier** — a specific number, code or name from your documents. Did hybrid search find it?
5. **A paraphrase** using none of the document's vocabulary. Did the dense half find it?

### Step 4: measure it

Now build the evaluation set from Module 8 §8.11. Create `evaluate.py`:

```python
"""evaluate.py - is retrieval actually working?"""

import sys
sys.path.append("labs/08-rag")
from rag import DocumentQA, load_documents

# 10-20 questions, each with text you KNOW appears in the correct chunk.
EVAL_SET = [
    {"question": "What is the refund window?",     "must_contain": "14 days"},
    {"question": "Who approves expenses over £500?", "must_contain": "director"},
    # ... add your own, from your own documents
]

qa = DocumentQA()
qa.index(load_documents("documents"))

hits = 0
for case in EVAL_SET:
    retrieved = qa.retrieve(case["question"], top_k=4)
    found = any(case["must_contain"].lower() in c["text"].lower() for c in retrieved)
    hits += found
    print(f"  {'HIT ' if found else 'MISS'}  {case['question']}")

print(f"\n  Retrieval recall@4: {hits}/{len(EVAL_SET)} ({hits/len(EVAL_SET):.0%})")
```

**This measures retrieval alone, separately from generation** — which is the whole point of Module 8 §8.11. A miss here means no prompt can save the answer.

Then tune **one thing at a time** and re-run:

| Change | Try |
|---|---|
| `chunk_size` | 300, 600, 1200 |
| `top_k` | 2, 4, 8 |
| `fetch_k` | 10, 20, 50 |
| Dense only | Comment out the `sparse` ranking |
| Keyword only | Comment out the `dense` ranking |

**Record the numbers.** That table is the difference between engineering and guessing.

**✅ Part 2 complete when you have a working bot and a retrieval recall number.**

---

## 🚀 Stretch Challenges

### 1. Add a cross-encoder re-ranker

```powershell
pip install sentence-transformers   # already installed
```

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Fetch 20, re-rank, keep 4
pairs = [(question, self.chunks[i]["text"]) for i in fused[:20]]
scores = reranker.predict(pairs)
best = [i for i, _ in sorted(zip(fused, scores), key=lambda x: -x[1])][:top_k]
```

Re-run `evaluate.py`. **Module 8 §8.7 claims this is often the highest-return single improvement — verify it on your data.**

### 2. Implement parent-document retrieval

Index 200-character chunks for precise matching, but return their 1000-character parents for context. Measure whether recall *and* answer quality both improve.

Often the single best structural change to a RAG pipeline (§8.8).

### 3. Prepend heading context to chunks

For Markdown documents, prefix each chunk with its heading path:

```
"Billing > Refunds > Partial refunds\n\n<chunk text>"
```

Cheap, and it markedly improves retrieval for questions phrased using heading vocabulary (§8.4, strategy 3).

### 4. Add multi-query retrieval

Have the model rewrite the question three ways, retrieve for each, and fuse with your `reciprocal_rank_fusion`. Measure the recall gain against the extra latency and cost (§8.8).

### 5. Detect and surface conflicts

Add test documents that contradict each other — an old policy and its replacement. Does your bot notice? Does the "if sources conflict, say so" rule fire?

**This is a genuinely hard problem** and most RAG systems fail it silently.

### 6. Build a Gradio UI

```python
import gradio as gr

def ask(question):
    result = qa.answer(question)
    sources = "\n".join(f"[{i}] {c['metadata']['source']}"
                        for i, c in enumerate(result["chunks"], 1))
    return result["answer"], sources

gr.Interface(fn=ask, inputs="text", outputs=["text", "text"]).launch()
```

A preview of Module 13 — and it makes the portfolio piece shareable.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why RRF beats naive score addition, with numbers.
3. Run `python solution.py` for four demos: overlap cost, chunk-size trade-off, RRF versus score addition, and a full offline retrieval pass.

**Keep `rag.py` and `evaluate.py`.** Module 11 turns the evaluation set into a proper harness, and Module 13 deploys the bot.

**Next:** Module 9 — AI Agents & Tool Use. Your bot can answer from documents; an agent can decide *what to do* — search, calculate, call an API — and chain those decisions together.
