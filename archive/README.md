# Archive — Original Session Material

This folder holds the material the course was built **from**. It's kept for reference and provenance; you don't need it to follow the course.

**Start at the [README](../README.md) and work through [`modules/`](../modules/).**

---

## What's here

```
archive/
├── notes-original/     the 10 original Markdown notes files
└── coding-examples/
    └── original/       the original session notebooks, demo scripts and PDF handouts
```

Everything is preserved as it was, with two exceptions noted under [Changes made](#changes-made) below.

---

## Where each piece went

The original ten notes were reorganised into fourteen sequential modules. Several were split, and a few topics were merged from more than one source.

| Original note | Became |
|---|---|
| `0.GenAI Basics.md` | [Module 1](../modules/01-foundations.md) (AI/ML/DL/GenAI, AGI, agentic AI), plus tokens and context window into [Module 3](../modules/03-tokens-embeddings-similarity.md), prompting into [Module 5](../modules/05-prompt-engineering.md) |
| `1.Python.md` | [Module 2](../modules/02-python-and-environment.md) — expanded with environment setup, virtual environments, notebooks and API-key safety |
| `2.Transformers.md` | Split: language models, tokenization, embeddings and similarity into [Module 3](../modules/03-tokens-embeddings-similarity.md); attention, multi-head, positional encoding and MoE into [Module 4](../modules/04-transformers.md) |
| `3.Art of Prompt Engineering.md` | [Module 5](../modules/05-prompt-engineering.md); instruction tuning moved to [Module 4 §4.9](../modules/04-transformers.md) |
| `4.Concepts RAG Systems.md` | Split: vector databases, ANN indexes and similarity metrics into [Module 7](../modules/07-vector-databases.md); chunking, re-ranking and the RAG pipeline into [Module 8](../modules/08-rag.md); pretraining-vs-fine-tuning into [Module 4 §4.9](../modules/04-transformers.md) |
| `5.LangChain_Toolchains.md` | Split: chains, LCEL, memory and parsers into [Module 6](../modules/06-langchain-chains.md); tools, ReAct, routing and composition into [Module 9](../modules/09-agents.md) |
| `6.Evaluation_Guardrails.md` | [Module 11](../modules/11-guardrails-evaluation.md) |
| `7.Multi-Modal LLM.md` | [Module 10](../modules/10-multimodal.md) |
| `8.Tool Augmented Agents.md` | [Module 9](../modules/09-agents.md) |
| `9.Open Ecosystem tools.md` | [Appendix A](../appendix/A-local-stack.md), with LangServe also referenced in [Module 13](../modules/13-deployment.md) |

### Topics that were added

Four of the fourteen modules had no source in the original notes:

| Module | Why it was added |
|---|---|
| [Module 2](../modules/02-python-and-environment.md) | The original `1.Python.md` covered syntax only — no environment setup, no API-key handling |
| [Module 12](../modules/12-fine-tuning.md) | Fine-tuning and LoRA were mentioned in passing but never taught |
| [Module 13](../modules/13-deployment.md) | Deployment, cost control and rate limiting were absent |
| [Module 14](../modules/14-ethics-limitations.md) | Ethics appeared as a single bullet list |

---

## The original notebooks

| Original folder | Corresponding lab |
|---|---|
| `1.Python Basics/` | [Lab 2](../labs/02-python-environment/) |
| `2.LLM Architecture/` | [Lab 3](../labs/03-tokens-embeddings/) |
| `3.Prompt Engineering/` | [Lab 5](../labs/05-prompt-engineering/) |
| `4.RAG/` | [Labs 7](../labs/07-vector-databases/) and [8](../labs/08-rag/) |
| `5.Langchain/` | [Lab 6](../labs/06-langchain-chains/) |
| `6.Evaluation Guardrails/` | [Lab 11](../labs/11-guardrails-evaluation/) |
| `7.Multi-Modal LLM/` | [Lab 10](../labs/10-multimodal/) |
| `8.Tool Augmented Agents/` | [Lab 9](../labs/09-agents/) |
| `9.Open Ecosystem Tool/` | [Appendix A](../appendix/A-local-stack.md) |

### How they differ from the course labs

The originals are **demonstration notebooks** — runnable examples showing what a technique looks like. The course labs are **exercises**: a `starter.py` with tasks to complete, a self-test that tells you when you're right, and a `SOLUTION.md` explaining the reasoning.

Both are useful. If you'd rather read working code than fill in blanks, the originals are a good complement — particularly `2.LLM Architecture` and `5.Langchain`, which are the most complete.

> **⚠️ The notebooks use provider APIs directly**, including Azure OpenAI in `3.Prompt Engineering`. The course labs use plain OpenAI with a free Ollama fallback. If you run the originals, check which client they expect and what environment variables they need.

---

## Changes made

Two, both for safety. Everything else is byte-for-byte as it was.

**1. Image paths fixed.** Thirty-five image links were repointed at [`images/`](../images/). Nineteen of them used a leading slash (`](/images/...)`), which renders in a local editor and **404s on GitHub**; all of them then needed re-depthing (`](../../images/...)`) once the notes moved down two folders into this archive. The images themselves are untouched and still shared with the modules.

**2. Two `.env` files renamed to `.env.example`.** They contained placeholders (`I-am-the-key`, `sk-your-api-key-here`), not real keys — but they were **tracked by git under the name `.env`**, which is a trap:

> `.gitignore` does not untrack files git already knows about. Anyone replacing that placeholder with a real key and committing would have leaked it, despite `.env` being in `.gitignore`.

Renaming to `.env.example` removes the trap and matches what [Module 2 §2.10](../modules/02-python-and-environment.md#210-api-keys-how-not-to-leak-money) teaches: commit the template, never the file.

---

<div align="center">

**[🏠 Back to the course](../README.md)**

</div>
