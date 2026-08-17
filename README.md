# GenAI From Scratch

**A 14-module, hands-on course that takes you from "I've never written Python" to "I deployed an AI app that answers questions about my own documents."**

No prior AI knowledge. No prior Python. No maths beyond arithmetic. Every concept gets a plain-English explanation, an everyday analogy, and a small piece of commented code you can actually run.

---

## Table of Contents

**Start here**

- [Why This Course Exists](#why-this-course-exists)
- [Who It's For](#who-its-for)
- [What You'll Build](#what-youll-build)
- [Prerequisites](#prerequisites)
- [How to Use This Repo](#how-to-use-this-repo)
- [A Note on Cost](#a-note-on-cost)
- [Suggested Timeline](#suggested-timeline)

**Part I — Foundations**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 1 | [Foundations: AI, ML, DL & Generative AI](modules/01-foundations.md) | Place any AI product on a map and explain how it works |
| 2 | [Python & Your Environment](modules/02-python-and-environment.md) | Run Python, manage packages, keep API keys safe |

**Part II — How Models Work**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 3 | [Tokens, Embeddings & Similarity](modules/03-tokens-embeddings-similarity.md) | Split text into tokens, turn it into vectors, measure meaning |
| 4 | [Transformers & Model Architecture](modules/04-transformers.md) | Explain attention, and pick between BERT-style and GPT-style models |

**Part III — Talking to Models**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 5 | [Prompt Engineering](modules/05-prompt-engineering.md) | Write prompts that produce reliable, parseable output |
| 6 | [LangChain & Chains](modules/06-langchain-chains.md) | Compose multi-step LLM pipelines with memory and output parsers |

**Part IV — Grounding Models in Your Data**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 7 | [Embeddings & Vector Databases](modules/07-vector-databases.md) | Store and search millions of vectors with FAISS and Chroma |
| 8 | [Retrieval-Augmented Generation (RAG)](modules/08-rag.md) 🏗️ | Build a chatbot that answers from *your* documents, with citations |

**Part V — Models That Act**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 9 | [AI Agents & Tool Use](modules/09-agents.md) | Give a model tools and let it plan multi-step work |
| 10 | [Multimodal AI](modules/10-multimodal.md) | Extract structured data from images, PDFs and audio |

**Part VI — Shipping Responsibly**

| # | Module | What you'll be able to do |
|---|--------|---------------------------|
| 11 | [Guardrails, Evaluation & Responsible AI](modules/11-guardrails-evaluation.md) | Measure quality and block unsafe output |
| 12 | Fine-Tuning & Model Customization | Decide between prompting, RAG and LoRA — then train an adapter |
| 13 | Deployment Basics | Ship a shareable demo and control your costs |
| 14 | Ethics & Limitations | Reason clearly about bias, IP, energy and what GenAI cannot do |

**Reference**

- Appendix A: Running Models Locally (Ollama, LM Studio, Flowise)
- Appendix B: Glossary
- Appendix C: Model Landscape
- Appendix D: Troubleshooting & FAQ

> **📌 Build status:** Modules 1–11 and Labs 1–11 are complete — including the first portfolio milestone. Modules 12–14 and the appendices are being written and published in order — links above go live as each lands. The original session notes are still in the repo root (`0.GenAI Basics.md` … `9.Open Ecosystem tools.md`) and remain readable until the module that supersedes them is published.
>
> **🔧 Something broken?** Run `python labs/02-python-environment/check_setup.py` — it diagnoses Python version, virtual environment, packages and API-key setup, and tells you the command to fix each problem.

---

## Why This Course Exists

Most GenAI material fails beginners in one of two ways.

Some of it is **marketing** — "AI will transform your business!" — and leaves you unable to build anything. The rest is **research papers and API docs**, which assume you already know what a tensor is.

This course sits in the gap. It assumes you are smart and motivated but have never trained a model, never called an API, and possibly never opened a terminal. It moves in a strict sequence where nothing is used before it is explained, and it ends with working software you can show people.

## Who It's For

**This course is for you if:**

- You're a developer, tester, analyst or student who keeps hearing "LLM", "RAG", "embeddings" and wants to genuinely understand them
- You want to *build* with GenAI, not just discuss it
- You've tried tutorials before and got lost the moment they said "just spin up a vector store"

**This course is not for you if:**

- You want the maths of backpropagation and gradient descent derived from first principles (start with [fast.ai](https://course.fast.ai/) or Andrew Ng's Deep Learning Specialization instead)
- You're already shipping LLM applications in production — skim Modules 11–13 and skip the rest

**Required background: none.** Genuinely. Module 2 teaches you Python from `print("hello")`.

## What You'll Build

Two portfolio pieces, plus a lab in every module:

| Milestone | Module | What it is |
|-----------|--------|------------|
| 🏗️ **Document Q&A bot** | 8 | Ask questions about your own PDFs and get answers with source citations |
| 🚀 **Deployed AI app** | 13 | A public, shareable web app running your own pipeline |

Along the way: a tokeniser explorer, a prompt template library, a semantic search engine, a tool-using agent, a receipt-to-JSON extractor, an evaluation harness, and a fine-tuned model adapter.

## Prerequisites

### What you need before Module 1

**Nothing.** Module 1 is concept-only — no installs, no accounts, no code. Read it and do the lab in your browser.

### What you need before Module 2

Set these up when you get there. Module 2 walks you through every step.

| Requirement | Version | Notes |
|-------------|---------|-------|
| **A computer** | Windows, macOS or Linux | 8 GB RAM is comfortable. Any machine from the last 8 years is fine. |
| **Python** | 3.10 or newer | Module 2 shows you how to install and verify it. |
| **A code editor** | — | [VS Code](https://code.visualstudio.com/) is free and what we assume. |
| **Git** | any recent | Only needed to clone this repo. You can also download it as a ZIP. |
| **A terminal** | — | PowerShell on Windows; Terminal on macOS/Linux. Module 2 covers the five commands you need. |

**Zero-install alternative:** every lab from Module 3 onward runs in [Google Colab](https://colab.research.google.com/) — free, browser-based, nothing to install. If your machine is locked down or underpowered, use Colab throughout.

### What you need before Module 3

An **API key** from at least one model provider, *or* [Ollama](https://ollama.com/) installed for the free local route. Module 2 sets this up and explains the trade-off. See [A Note on Cost](#a-note-on-cost) below.

## How to Use This Repo

### 1. Get the files

```bash
# Clone the repository
git clone <your-repo-url>
cd GenAI
```

No Git? Click the green **Code** button on GitHub → **Download ZIP** → unzip it.

### 2. Understand the layout

```
GenAI/
├── README.md              ← you are here
├── requirements.txt       ← all Python packages, with comments explaining each
├── .env.example           ← template for your API keys (copy to .env)
│
├── modules/               ← THE COURSE. Read these in order.
│   └── 01-foundations.md
│
├── labs/                  ← hands-on work, one folder per module
│   ├── 01-foundations/
│   │   ├── README.md      ← the lab brief: objective, steps, stretch challenge
│   │   ├── worksheet.md   ← what you fill in
│   │   └── SOLUTION.md    ← answer key — open only after attempting
│   └── 02-python-environment/
│       ├── check_setup.py ← run this whenever something breaks
│       ├── starter.py     ← your work goes here (has a built-in self-test)
│       ├── solution.py
│       └── SOLUTION.md
│
├── images/                ← diagrams referenced by the modules
├── appendix/              ← glossary, troubleshooting, reference tables
└── Coding Examples/       ← original session notebooks (being migrated into labs/)
```

### 3. Work through it

For each module, in this order:

1. **Read the module.** Don't skip the "Why This Matters" section — it's the hook that makes the rest make sense.
2. **Run the code as you read.** Every snippet is self-contained and commented. Type it out rather than copy-pasting; it's slower and you'll learn more.
3. **Do the lab.** This is where the learning actually happens. Attempt it before opening `SOLUTION.md`.
4. **Read "Common Mistakes & Misconceptions."** These are the specific wrong turns that cost people hours.
5. **Try the stretch challenge** if you have time and energy. Skip it guilt-free if not — it's never a prerequisite for the next module.

### 4. Rules of engagement

- **Go in order.** Each module genuinely depends on the previous one. Module 8 (RAG) will be meaningless without Module 7 (vector databases), which needs Module 3 (embeddings).
- **Getting stuck is normal.** Check `appendix/D-troubleshooting.md` first, then open an issue.
- **Type the code.** Copy-paste teaches your clipboard, not you.

## A Note on Cost

You can complete this entire course for **free**, or spend roughly **$5–15** for a smoother experience. Your choice, and you can mix the two.

| Route | Cost | Trade-off |
|-------|------|-----------|
| **Paid API** (OpenAI, Anthropic) | ~$5–15 total for all 14 modules | Fast, high quality, one line of setup. This is the default path in the labs. |
| **Free tier** (Google Gemini) | $0 | Generous free quota. Rate-limited, occasionally requires waiting. |
| **Local models** (Ollama) | $0 | Fully private, unlimited, works offline. Slower, and small local models are noticeably weaker at complex reasoning. |

Every lab that costs money says so at the top and gives an Ollama-based alternative. Module 13 covers cost control, rate limits and how to avoid a surprise bill in detail.

> **⚠️ Before you spend anything:** set a hard spending limit in your provider's billing dashboard. Module 2 shows you where. Do this on day one — it is the single cheapest insurance in this course.

## Suggested Timeline

Designed for **~5 hours per week** alongside a job. Adjust freely — this is a map, not a schedule.

| Week | Modules | Milestone |
|------|---------|-----------|
| **1** | 1–2 | Environment working; your first API call succeeds |
| **2** | 3–4 | You can explain tokenization and attention in your own words |
| **3** | 5 | A reusable prompt template library |
| **4** | 6 | A working multi-step chain with memory |
| **5** | 7–8 | 🏗️ **A RAG chatbot over your own documents** |
| **6** | 9–10 | An agent that uses tools; a vision extraction script |
| **7** | 11–12 | An evaluation harness; one fine-tuned adapter |
| **8** | 13–14 | 🚀 **A deployed, shareable demo app** |

**In a hurry?** The shortest path to a working RAG app is Modules 1 → 2 → 3 → 5 → 7 → 8. You can come back for the theory.

**Teaching this?** Each module is roughly one 90-minute session: 50 minutes of content, 40 minutes of lab.

---

## Contributing

Found a typo, a broken link, or an explanation that didn't land? Please open an issue or a pull request. Corrections from beginners are the most valuable kind — if something confused you, it will confuse the next person, and that's a bug in the material.

## License

Course content is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use it, remix it, teach with it, just credit the source. Code samples are MIT licensed.

---

<div align="center">

**Ready?** → **[Start with Module 1: Foundations](modules/01-foundations.md)**

</div>
