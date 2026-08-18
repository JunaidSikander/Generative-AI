# Appendix B: Glossary

> Every term this course uses, in plain English, with a link to where it's taught properly.

**Jump to:** [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [Z](#z)

---

## A

**Adapter** — A small set of trained parameters that specialises a frozen base model. A LoRA adapter for a 7B model is about 8 MB. → [Module 12 §12.4](../modules/12-fine-tuning.md#124-lora-how-it-works)

**Agent** — An LLM in a loop with tools, choosing what to do next at runtime. Distinguished from a chain, where *you* fix the sequence. → [Module 9](../modules/09-agents.md)

**AGI (Artificial General Intelligence)** — Hypothetical AI matching human flexibility across any intellectual task. Does not exist; whether current approaches lead there is disputed. → [Module 1 §1.6](../modules/01-foundations.md#16-a-second-way-to-sort-ai-narrow-general-super)

**ANN (Approximate Nearest Neighbour)** — Search that finds *probably* the closest vectors, quickly, rather than certainly the closest, slowly. The trade at the heart of vector databases. → [Module 7 §7.4](../modules/07-vector-databases.md#74-ann-indexes)

**ANI (Artificial Narrow Intelligence)** — AI competent at one class of task. Everything that exists today, including frontier chat models. → [Module 1 §1.6](../modules/01-foundations.md#16-a-second-way-to-sort-ai-narrow-general-super)

**ASI (Artificial Superintelligence)** — Hypothetical AI exceeding the best human performance across all domains. Speculative. → [Module 1 §1.6](../modules/01-foundations.md#16-a-second-way-to-sort-ai-narrow-general-super)

**Attention** — The mechanism letting each token rebuild itself as a weighted blend of the tokens relevant to it. Why one word gets different vectors in different sentences. → [Module 4 §4.2](../modules/04-transformers.md#42-self-attention-query-key-value)

**Autoregressive** — Generating one token at a time, feeding each output back as input. How every LLM produces text. → [Module 1 §1.7](../modules/01-foundations.md#17-how-a-large-language-model-actually-works)

---

## B

**Base model** — A model after pretraining but before instruction tuning. Fluent at continuing text, poor at following instructions. → [Module 4 §4.9](../modules/04-transformers.md#49-pretraining-vs-fine-tuning)

**BERT** — An encoder-only model family. Reads bidirectionally, excellent for embeddings and classification, cannot generate text left to right. → [Module 4 §4.8](../modules/04-transformers.md#48-encoder-decoder-or-both)

**Bi-encoder** — Embeds query and document *independently*, so documents can be indexed in advance. Fast and scalable; no cross-attention. Contrast **cross-encoder**. → [Module 8 §8.7](../modules/08-rag.md#87-re-ranking)

**BM25** — The standard keyword-relevance function. Term frequency with saturation and length normalisation. What semantic search can't do. → [Module 8 §8.6](../modules/08-rag.md#86-hybrid-search)

**BPE (Byte-Pair Encoding)** — The dominant tokenisation algorithm: repeatedly merge the most frequent adjacent pair into a single token. → [Module 3 §3.2](../modules/03-tokens-embeddings-similarity.md#32-tokenization-text-into-pieces)

---

## C

**Catastrophic forgetting** — When fine-tuning on narrow data makes a model worse at everything else. PEFT largely avoids it by leaving base weights untouched. → [Module 12 §12.2](../modules/12-fine-tuning.md#122-what-fine-tuning-actually-changes)

**Chain** — A fixed sequence of steps, each taking structured input and passing its result on. You write the path. → [Module 6 §6.3](../modules/06-langchain-chains.md#63-what-a-chain-actually-is)

**Chunking** — Splitting documents into pieces small enough to embed and retrieve individually. **The biggest quality lever in a RAG pipeline.** → [Module 8 §8.4](../modules/08-rag.md#84-chunking-the-biggest-quality-lever)

**Circuit breaker** — Stops calling a service that's clearly down, failing in microseconds instead of waiting for timeouts. → [Module 11 §11.6](../modules/11-guardrails-evaluation.md#116-resilience-patterns)

**CLIP** — A model trained so images and their captions land at the same point in one shared embedding space. Enables cross-modal search. → [Module 10 §10.9](../modules/10-multimodal.md#109-shared-embedding-spaces)

**Context window** — How many tokens a model can see at once. One shared budget: prompt, history, retrieved documents and the answer all compete. → [Module 3 §3.9](../modules/03-tokens-embeddings-similarity.md#39-the-context-window)

**Contextual embedding** — A vector that depends on surrounding words, so the same token gets different vectors in different sentences. What attention produces. Contrast **static embedding**. → [Module 3 §3.4](../modules/03-tokens-embeddings-similarity.md#34-embeddings-meaning-as-coordinates)

**Cosine similarity** — Angle between two vectors, ignoring length. The default metric for text. Higher is more similar. → [Module 3 §3.6](../modules/03-tokens-embeddings-similarity.md#36-measuring-similarity)

**CoT (Chain-of-Thought)** — Prompting a model to show its reasoning before the answer, which puts intermediate results into the visible text. → [Module 5 §5.6](../modules/05-prompt-engineering.md#56-chain-of-thought)

**Cross-encoder** — Reads query and document *together*, so attention spans both. Far more accurate than a bi-encoder, far slower. Used for re-ranking. → [Module 8 §8.7](../modules/08-rag.md#87-re-ranking)

---

## D

**Deep learning** — Machine learning using neural networks with many layers, where each layer discovers its own useful features. → [Module 1 §1.4](../modules/01-foundations.md#14-deep-learning-layers-that-find-their-own-features)

**Demographic parity** — A fairness definition: each group receives positive outcomes at the same rate. Can look perfect while the system is badly unfair on another definition. → [Module 14 §14.2](../modules/14-ethics-limitations.md#142-measuring-fairness)

**Disparate impact ratio** — Lowest selection rate divided by highest. Below 0.8 is the "four-fifths rule", a screening threshold from US employment practice. → [Module 14 §14.2](../modules/14-ethics-limitations.md#142-measuring-fairness)

**Decoder-only** — The GPT-style architecture. Causal masking means each token sees only itself and the past — which is what makes generation possible. → [Module 4 §4.8](../modules/04-transformers.md#48-encoder-decoder-or-both)

---

## E

**Embedding** — A list of numbers representing meaning, positioned so related things land near each other. → [Module 3 §3.4](../modules/03-tokens-embeddings-similarity.md#34-embeddings-meaning-as-coordinates)

**Embedding dimension** — How many numbers per vector — 384, 768, 1536. A cost decision as much as a quality one. → [Module 3 §3.5](../modules/03-tokens-embeddings-similarity.md#35-making-sense-of-vector-space)

**Encoder-only** — The BERT-style architecture. Bidirectional attention, excellent representations, cannot generate. Where embedding models come from. → [Module 4 §4.8](../modules/04-transformers.md#48-encoder-decoder-or-both)

**Equal opportunity** — A fairness definition: among people who *should* receive a positive outcome, each group does so equally often. Requires ground truth. → [Module 14 §14.2](../modules/14-ethics-limitations.md#142-measuring-fairness)

**Exponential backoff** — Growing delays between retries — 1s, 2s, 4s — so a struggling service gets room to recover. → [Module 11 §11.6](../modules/11-guardrails-evaluation.md#116-resilience-patterns)

---

## F

**Faithfulness** — Whether an answer is actually supported by the context it was given. One of the RAG evaluation triad. → [Module 11 §11.8](../modules/11-guardrails-evaluation.md#118-the-rag-triad)

**FAISS** — An in-process vector search library. Fast, no server — and it stores vectors only, not your text. → [Module 7 §7.7](../modules/07-vector-databases.md#77-faiss-in-practice)

**Few-shot prompting** — Putting 2–5 examples in the prompt. Mostly teaches *format and label space*, not concepts. → [Module 5 §5.5](../modules/05-prompt-engineering.md#55-zero--one--and-few-shot-prompting)

**Fine-tuning** — Continuing training on your own data to change behaviour, style or format. **Teaches skills; does not reliably supply facts.** → [Module 12](../modules/12-fine-tuning.md)

**F1 score** — Harmonic mean of precision and recall. Weights them equally, which is almost never what you want. → [Module 11 §11.10](../modules/11-guardrails-evaluation.md#1110-metrics-which-ones-matter)

**Four-fifths rule** — See **disparate impact ratio**.

**Function calling** — See **tool calling**.

---

## G

**Generative AI** — Deep learning used to create new content rather than classify or score existing content. → [Module 1 §1.5](../modules/01-foundations.md#15-generative-ai-models-that-produce-not-just-judge)

**GPT** — A decoder-only model family trained on next-token prediction. → [Module 4 §4.8](../modules/04-transformers.md#48-encoder-decoder-or-both)

**Gradio** — A Python library that turns a function into a web UI in a few lines. The fastest path to a shareable demo. → [Module 13 §13.2](../modules/13-deployment.md#132-gradio-the-fastest-path)

**Grounding** — Making a model answer from supplied evidence rather than its training. What RAG does. → [Module 8 §8.9](../modules/08-rag.md#89-grounding-the-generation)

**Guardrail** — A fast, deterministic runtime check that blocks bad input or output. Distinguished from evaluation, which measures offline. → [Module 11 §11.1](../modules/11-guardrails-evaluation.md#111-guardrails-vs-evaluation)

---

## H

**Hallucination** — Confident, fluent, false output. A structural consequence of optimising for plausibility rather than truth — not a bug awaiting a fix. → [Module 1 §1.7](../modules/01-foundations.md#17-how-a-large-language-model-actually-works)

**HNSW** — A graph-based ANN index. The production default: very fast, very accurate, memory-hungry. → [Module 7 §7.4](../modules/07-vector-databases.md#74-ann-indexes)

**Hybrid search** — Combining semantic and keyword retrieval, usually fused with RRF. Fixes semantic search's failure on exact identifiers. → [Module 8 §8.6](../modules/08-rag.md#86-hybrid-search)

---

## I

**Indirect prompt injection** — Hidden instructions in content the model *reads* — a document, a web page, an image. More serious than direct injection: the user may be innocent, and it persists in your index. → [Module 11 §11.3](../modules/11-guardrails-evaluation.md#113-the-threat-landscape)

**Inference** — Using a trained model. Fast and cheap per call, and it dominates a deployed model's lifetime cost and footprint. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**Instruction hierarchy** — The convention that system instructions outrank user instructions, which outrank tool data. **Trained-in preference, not architecture.** → [Module 5 §5.3](../modules/05-prompt-engineering.md#53-the-message-hierarchy)

**Instruction tuning** — Fine-tuning on instruction/response pairs so a base model reliably follows directions. What turns a text predictor into an assistant. → [Module 4 §4.9](../modules/04-transformers.md#49-pretraining-vs-fine-tuning)

**IVF (Inverted File Index)** — An ANN index that clusters vectors, then searches only the clusters nearest the query. → [Module 7 §7.4](../modules/07-vector-databases.md#74-ann-indexes)

---

## J

**Jailbreak** — Direct prompt injection: a user crafting input to override a system prompt. → [Module 11 §11.3](../modules/11-guardrails-evaluation.md#113-the-threat-landscape)

**JSON mode** — A provider feature guaranteeing syntactically valid JSON. **Guarantees syntax, not your schema.** → [Module 5 §5.8](../modules/05-prompt-engineering.md#58-structured-output)

---

## K

**Knowledge cutoff** — The date training data ends. The model knows nothing after it and has no live connection to anything. → [Module 1 §1.7](../modules/01-foundations.md#17-how-a-large-language-model-actually-works)

---

## L

**LangChain** — A framework for composing LLM pipelines. Useful for genuine composition needs; heavier than the provider SDK for simple calls. → [Module 6](../modules/06-langchain-chains.md)

**LCEL (LangChain Expression Language)** — The `|` operator plus one shared `Runnable` interface. Python's `__or__`, and nothing more. → [Module 6 §6.4](../modules/06-langchain-chains.md#64-lcel-and-the-runnable-interface)

**LLM (Large Language Model)** — A model that predicts the next token, scaled up in parameters, data and compute until new capabilities emerge. → [Module 3 §3.1](../modules/03-tokens-embeddings-similarity.md#31-what-a-language-model-is)

**LoRA (Low-Rank Adaptation)** — Fine-tuning that trains two small matrices instead of updating the full weights. About 0.06% of parameters for a 7B model. → [Module 12 §12.4](../modules/12-fine-tuning.md#124-lora-how-it-works)

**Lost in the middle** — Models recall the start and end of a long context more reliably than the middle. Shapes how you order retrieved chunks. → [Module 3 §3.9](../modules/03-tokens-embeddings-similarity.md#39-the-context-window)

**LRU (Least Recently Used)** — A cache eviction policy: drop whatever was used longest ago. Note *recently used*, not *oldest inserted*. → [Module 13 §13.7](../modules/13-deployment.md#137-caching)

---

## M

**Machine learning** — Supplying examples to get rules, rather than supplying rules to get answers. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**MCP (Model Context Protocol)** — A standard for publishing tools so any compatible client can use them. Standardises *how* tools are exposed, not what tool calling is. → [Module 9 §9.3](../modules/09-agents.md#93-function-calling-the-mechanics)

**Memory** — Re-sending conversation history as part of the prompt. There is no state inside the model. → [Module 6 §6.9](../modules/06-langchain-chains.md#69-memory-conversational-state)

**MoE (Mixture of Experts)** — An architecture that routes each token to a few specialised sub-networks. Big-model capacity at small-model compute — but full memory. → [Module 4 §4.10](../modules/04-transformers.md#410-mixture-of-experts)

**MRR (Mean Reciprocal Rank)** — Average of 1/(rank of the first relevant result). Asks *how high*, where recall asks *whether*. → [Module 11 §11.10](../modules/11-guardrails-evaluation.md#1110-metrics-which-ones-matter)

**Multi-head attention** — Running several attention patterns in parallel, each learning a different relationship. Costs about the same as one, by splitting the dimensions. → [Module 4 §4.4](../modules/04-transformers.md#44-multi-head-attention)

---

## N

**Normalisation (of vectors)** — Scaling to length 1. Once normalised, cosine similarity and dot product are identical and all three metrics rank the same. → [Module 3 §3.6](../modules/03-tokens-embeddings-similarity.md#36-measuring-similarity)

---

## O

**Ollama** — A local model runtime with an OpenAI-compatible API. The free path through this course. → [Appendix A](A-local-stack.md)

**Overfitting** — When a model memorises training examples instead of learning to generalise. Visible as training loss falling while validation loss rises. → [Module 12 §12.8](../modules/12-fine-tuning.md#128-training-the-knobs-that-matter)

---

## P

**Parameters** — The adjustable numbers inside a model. Training tunes them. "70 billion parameters" means 70 billion tuned values. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**PEFT (Parameter-Efficient Fine-Tuning)** — Freezing the base model and training a small number of new parameters. LoRA is the dominant method. → [Module 12 §12.3](../modules/12-fine-tuning.md#123-full-fine-tuning-vs-peft)

**Positional encoding** — Adding position information to token embeddings, because attention alone is order-blind. → [Module 4 §4.5](../modules/04-transformers.md#45-positional-encoding)

**Precision** — Of what you flagged, how much was right. Contrast **recall**. → [Module 11 §11.10](../modules/11-guardrails-evaluation.md#1110-metrics-which-ones-matter)

**Pretraining** — Training from scratch on enormous raw text via next-token prediction. Produces a base model. Costs millions. → [Module 4 §4.9](../modules/04-transformers.md#49-pretraining-vs-fine-tuning)

**Prompt** — The text given to a model as input. The only interface to a model you cannot retrain. → [Module 5 §5.1](../modules/05-prompt-engineering.md#51-what-a-prompt-actually-is)

**Prompt injection** — Text in your input being interpreted as instructions. **Unsolved**, because there is no code/data boundary in a prompt. → [Module 11 §11.2](../modules/11-guardrails-evaluation.md#112-the-semantic-gap)

---

## Q

**QLoRA** — LoRA with the frozen base quantized to 4 bits. What makes fine-tuning possible on a free Colab GPU. → [Module 12 §12.5](../modules/12-fine-tuning.md#125-qlora-and-quantization)

**Quantization** — Storing weights at lower precision to save memory, at some quality cost. → [Module 12 §12.5](../modules/12-fine-tuning.md#125-qlora-and-quantization)

**Query, Key, Value** — The three vectors each token produces for attention: what I'm looking for, what I offer, what I contribute. → [Module 4 §4.2](../modules/04-transformers.md#42-self-attention-query-key-value)

---

## R

**RAG (Retrieval-Augmented Generation)** — Retrieving relevant text at query time and putting it in the prompt. **Supplies facts; fine-tuning does not.** → [Module 8](../modules/08-rag.md)

**Rate limiting** — Bounding how many requests a user can make. Without it, one user or one bug exhausts your budget. → [Module 13 §13.8](../modules/13-deployment.md#138-rate-limiting-your-users)

**ReAct** — Reasoning + Acting: a model interleaving thought, tool call and observation until it can answer. → [Module 9 §9.6](../modules/09-agents.md#96-react-reason-and-act)

**Recall** — Of what you *should* have found, how much you did. Contrast **precision**. → [Module 11 §11.10](../modules/11-guardrails-evaluation.md#1110-metrics-which-ones-matter)

**Recall@k** — In retrieval, whether the correct item appeared in the top k results. → [Module 7 §7.5](../modules/07-vector-databases.md#75-the-recalllatency-trade-off)

**Re-ranking** — A second, more accurate pass over retrieved candidates using a cross-encoder. Often the highest-return single addition to a RAG system. → [Module 8 §8.7](../modules/08-rag.md#87-re-ranking)

**RLHF (Reinforcement Learning from Human Feedback)** — Training on human preference rankings. What makes a model helpful rather than merely fluent — and a source of sycophancy. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**RoPE (Rotary Position Embedding)** — The modern positional encoding: rotates query and key vectors by an angle set by position. → [Module 4 §4.5](../modules/04-transformers.md#45-positional-encoding)

**RRF (Reciprocal Rank Fusion)** — Merging ranked lists using **only rank position**, never the scores — which is why it can fuse cosine similarity and BM25 without normalisation. → [Module 8 §8.6](../modules/08-rag.md#86-hybrid-search)

**Runnable** — LangChain's shared interface: `invoke`, `batch`, `stream`, `ainvoke`. A chain of Runnables is itself a Runnable. → [Module 6 §6.4](../modules/06-langchain-chains.md#64-lcel-and-the-runnable-interface)

---

## S

**Self-attention** — Attention where queries, keys and values all come from the same sequence. → [Module 4 §4.2](../modules/04-transformers.md#42-self-attention-query-key-value)

**Self-consistency** — Sampling several reasoning chains and taking the majority answer. → [Module 5 §5.7](../modules/05-prompt-engineering.md#57-beyond-a-single-chain)

**Self-supervised learning** — Labels derived from the data itself, like predicting a hidden next word. **How LLMs are trained**, and why they scale to trillions of examples. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**Semantic search** — Finding documents by meaning rather than shared words. → [Module 3 §3.7](../modules/03-tokens-embeddings-similarity.md#37-semantic-search-putting-it-together)

**Softmax** — Converting raw scores into probabilities summing to 1. Subtract the max first for numerical stability. → [Module 3 §3.8](../modules/03-tokens-embeddings-similarity.md#38-from-numbers-back-to-a-token)

**Spurious correlation** — A model learning an accidental pattern in your data instead of the thing you meant. Appears at every level of the stack. → [Module 12 §12.6](../modules/12-fine-tuning.md#126-the-data-is-the-work)

**Static embedding** — One fixed vector per word, regardless of context (word2vec, GloVe). Cannot distinguish the two senses of "bit". → [Module 3 §3.4](../modules/03-tokens-embeddings-similarity.md#34-embeddings-meaning-as-coordinates)

**Streaming** — Returning output token by token. Barely changes total time; transforms *perceived* speed. → [Module 13 §13.9](../modules/13-deployment.md#139-streaming)

**Structured output** — Constraining a model to return data matching a schema. The right default for anything a program consumes. → [Module 5 §5.8](../modules/05-prompt-engineering.md#58-structured-output)

**Supervised learning** — Learning from labelled examples. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**Sycophancy** — A model agreeing with whatever you assert. Partly a consequence of RLHF. → [Module 5 §5.9](../modules/05-prompt-engineering.md#59-rubrics-making-the-model-a-useful-critic)

**System prompt** — Standing instructions setting persona, rules and format. Influential, **not a security boundary**. → [Module 5 §5.3](../modules/05-prompt-engineering.md#53-the-message-hierarchy)

---

## T

**Temperature** — How adventurous sampling is. 0 is near-deterministic; use it for anything you'll parse. → [Module 3 §3.8](../modules/03-tokens-embeddings-similarity.md#38-from-numbers-back-to-a-token)

**Token** — A word or word-fragment; the model's unit of text. It sees integer IDs, never letters. → [Module 3 §3.2](../modules/03-tokens-embeddings-similarity.md#32-tokenization-text-into-pieces)

**Token bucket** — A rate-limiting algorithm permitting short bursts while bounding the sustained rate. → [Module 13 §13.8](../modules/13-deployment.md#138-rate-limiting-your-users)

**Tool calling** — A model emitting a structured request — a name and arguments — that **your** runtime validates and executes. The model never runs code. → [Module 9 §9.3](../modules/09-agents.md#93-function-calling-the-mechanics)

**Top-k / Top-p** — Sampling strategies that restrict which tokens can be chosen. Top-p adapts its cutoff per step; top-k doesn't. → [Module 3 §3.8](../modules/03-tokens-embeddings-similarity.md#38-from-numbers-back-to-a-token)

**Training** — Teaching a model from data. Slow and expensive; happens once. Contrast **inference**. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

**Transformer** — The architecture behind every model in this course: attention, multi-head, positional encoding, feed-forward, residuals. → [Module 4](../modules/04-transformers.md)

**TTL (Time To Live)** — How long a cache entry stays valid. → [Module 13 §13.7](../modules/13-deployment.md#137-caching)

---

## U

**Unsupervised learning** — Finding structure in unlabelled data, such as clustering. → [Module 1 §1.3](../modules/01-foundations.md#13-machine-learning-learning-from-examples)

---

## V

**Vector database** — A store built for high-dimensional vectors and fast nearest-neighbour search. Returns *probably* the nearest, quickly. → [Module 7 §7.2](../modules/07-vector-databases.md#72-what-a-vector-database-actually-is)

**Virtual environment** — An isolated Python installation per project, so one project's dependencies can't break another's. → [Module 2 §2.3](../modules/02-python-and-environment.md#23-virtual-environments-why-before-how)

---

## W

**Workflow** — LLM calls on a predefined path. **Most systems described as agents are actually workflows**, and that's usually correct. → [Module 9 §9.9](../modules/09-agents.md#99-workflows-vs-agents)

---

## Z

**Zero-shot** — Prompting with an instruction and no examples. → [Module 5 §5.5](../modules/05-prompt-engineering.md#55-zero--one--and-few-shot-prompting)

---

<div align="center">

**[🏠 Course README](../README.md)** · **[💻 Local stack](A-local-stack.md)** · **[📊 Model landscape](C-model-landscape.md)** · **[🔧 Troubleshooting](D-troubleshooting.md)**

</div>
