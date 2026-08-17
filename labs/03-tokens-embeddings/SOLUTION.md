# Lab 3 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Part 1 — The maths

### Task 1 — `cosine_similarity`

```python
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

Dividing by both magnitudes cancels length, leaving only direction.

**Why ignoring length is the right choice for text:** a 500-word document produces a longer vector than a 10-word one purely because there's more text. You don't want "similar" to mean "similar length". Direction carries meaning; magnitude mostly carries verbosity.

The check `cosine_similarity([1,1], [50,50]) == 1.0` makes this concrete — those vectors differ 50× in length and score a perfect 1.0.

> **⚠️ Edge case worth knowing:** a zero vector `[0,0,0]` has magnitude 0, so this divides by zero and returns `nan`. Production code guards it. It shows up in practice when you accidentally embed an empty string.

---

### Task 2 — `dot_product`

```python
def dot_product(a, b):
    return float(np.dot(a, b))
```

Trivial to write, and included to make the comparison with cosine explicit. `[1,2,3]·[4,5,6]` = `4 + 10 + 18` = `32`.

**When magnitude-sensitivity is what you want:** recommender systems, where a popular item legitimately *should* score higher. In text retrieval it's usually a nuisance — unless your vectors are normalised, in which case it's identical to cosine and cheaper.

---

### Task 3 — `euclidean_distance`

```python
def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))
```

`norm(a - b)` is exactly `sqrt(sum((a-b)²))`. The 3-4-5 triangle check gives `5.0`.

**The direction trap.** This is a **distance**:

| Metric | Better means |
|---|---|
| Cosine similarity | **Higher** |
| Dot product | **Higher** |
| Euclidean distance | **Lower** |

Sort Euclidean descending and you return your *worst* matches. Nothing raises an exception; you just get a RAG system that confidently retrieves irrelevant documents. It's one of the most annoying bugs in this space precisely because it's silent — and `solution.py`'s `demo_metrics_agree()` shows the ranking flip directly.

---

### Task 4 — `normalise`

```python
def normalise(v):
    return v / np.linalg.norm(v)
```

`[3,4]` has length 5, so it becomes `[0.6, 0.8]` — same direction, length 1.

**Why this matters more than it looks.** After normalisation:

- Cosine similarity **=** dot product (the denominator is 1×1)
- Euclidean distance becomes a monotonic function of cosine similarity

**All three metrics then rank identically.** That's the bonus check, and it's why real vector databases normalise on ingest: you get cosine's meaning with the dot product's speed, and switching metrics later requires no re-ranking.

`all-MiniLM-L6-v2` will do it for you via `normalize_embeddings=True`.

---

### Task 5 — `top_k_indices`

```python
def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]
```

Three things happening:

1. `np.argsort(scores)` — indices that would sort **ascending**
2. `[::-1]` — reverse, giving descending
3. `[:k]` — take the first k

**Why `argsort` and not `sort`?** You need to know *which document* scored highly, not just the score. `argsort` returns positions; `sort` returns values and loses the mapping.

> **📌 At scale, use `np.argpartition` instead.** `argsort` sorts everything — O(n log n). To get the top 10 of a million, you don't need the other 999,990 in order. `np.argpartition(scores, -k)[-k:]` is O(n). It matters once your corpus is large, and it's roughly what vector databases do internally.

---

## Part 2 — Real text

### Task 6 — `count_tokens`

```python
def count_tokens(text, encoding_name="cl100k_base"):
    import tiktoken
    encoder = tiktoken.get_encoding(encoding_name)
    return len(encoder.encode(text))
```

### The answers to the questions

**1. Worst chars-per-token ratio?** Python code or Spanish, depending on your samples.

- **Code** fragments because `{`, `}`, `:`, `->` and identifiers like `items` often tokenise individually. A vocabulary learned mostly from prose has few merges covering code punctuation.
- **Spanish** costs more than English despite the same alphabet — accented characters and different morphology produce fewer useful merges.

**2. The cost implication.** The same *meaning* costs more tokens in non-English languages. Concretely:

- Non-English users cost you more per request
- Their effective context window is **smaller** — fewer words fit in the same token budget
- Quality is often lower too, since the model saw less of that language

This is a real fairness issue baked into the plumbing, not a pricing quirk. If you build for a multilingual audience, measure it rather than assuming parity.

**3. `"strawberry"`.** Typically splits into around three tokens — something like `["str", "aw", "berry"]`.

Now the reasoning: the model **never sees the letters**. It sees three integer IDs. Asking it to count `r` characters is asking it to report on information that isn't in its input. It can sometimes get there by having read *about* the spelling in training text, which is why it's inconsistent rather than uniformly wrong.

**This is the correct mental model for a whole family of "why is it so dumb at this?" complaints** — counting letters, reversing strings, rhyming, and character-level arithmetic. It's not a reasoning failure; it's a representation limit.

---

### Task 7 — `search`

```python
def search(query, documents, doc_embeddings, model, top_k=3):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = doc_embeddings @ query_embedding
    best = top_k_indices(scores, top_k)
    return [(documents[i], float(scores[i])) for i in best]
```

**Three things worth pointing at:**

**1. Same model for query and documents.** This is not a style preference. Different embedding models produce unrelated coordinate systems — the numbers aren't comparable at all. Use a different model for the query and you get plausible-looking scores that are meaningless, **with no error raised**. This also means changing your embedding model requires re-indexing your entire corpus.

**2. One matrix multiply scores everything.** `doc_embeddings @ query_embedding` is `(n_docs, 384) @ (384,)` → `(n_docs,)`. One operation, every document scored. Vastly faster than a Python loop, and this vectorised thinking is why NumPy exists.

**3. Normalised, so dot product = cosine.** Task 4's point, cashed in.

### The answers to the questions

**1. Score gap.** You'll see something like:

```
Query: 'cat on a mat'
  0.782  The cat sat on the mat.
  0.601  A kitten rested on the rug.     <- no shared words
  0.094  LLMs can generate human-like text.
```

There's a clear gap here — but **that's a property of this tiny, well-separated corpus, not a general guarantee.** On real data with many near-duplicates the distribution is much flatter, and picking a cutoff score becomes genuinely hard.

> **⚠️ This is why you should never hardcode a similarity threshold from a tutorial.** "Above 0.7 is relevant" is meaningless without knowing the model and the corpus. For some models unrelated text sits near 0.1; for others near 0.6. **Calibrate on your own data:** score pairs you know are related and pairs you know aren't, then pick a threshold from that distribution. And a cosine of 0.8 is not "80% similar" — the scale isn't a percentage and isn't calibrated.

**2. `"SKU-4471"`.** Returns essentially random documents with middling confidence. The embedding model has no idea what a product code is; it encodes something like "short alphanumeric string" and matches on that vague signal.

**This is the single most common production failure of pure semantic search.** Users search for exact things — order numbers, error codes, model numbers, names — and semantic search is bad at all of them.

**The fix is hybrid search:** run keyword search (BM25) and semantic search, then combine the scores. Keyword search nails exact matches; semantic search handles paraphrase. Module 8 builds this.

**3. A query that should match but doesn't.** Common culprits:

- **Negation.** *"coffee that isn't bitter"* usually matches *"bitter coffee"* strongly. Embeddings are notoriously weak at negation because "not" barely shifts the vector.
- **Multi-hop.** *"the animal that says meow"* may miss *"cat"* — that needs inference, not similarity.
- **Very short queries.** One or two words carry little signal.
- **Jargon the model never saw.** Domain-specific terms embed poorly.

If you found one, you've derived Module 8's motivation yourself.

---

## 🚀 Stretch — Discussion

### 1. Where semantic search fails

The negation result is the most important. **Embeddings measure topical similarity, not logical content.** "Bitter" and "not bitter" are about the same *topic*, so they land near each other. No amount of tuning fixes this at the embedding layer — you need reranking (a cross-encoder that reads query and document together) or an LLM filtering step. Module 8 covers both.

### 2. Comparing embedding models

`all-mpnet-base-v2` is generally better on nuanced retrieval — but **often not enough to justify 2× storage, 2× memory and slower search.** Bigger is not automatically better; the right answer depends on your corpus and your latency budget.

The habit worth forming: build a small evaluation set of query/expected-document pairs from your *own* data, then measure. Public leaderboards (MTEB) are a starting point, not an answer.

### 3. PCA caveat

You should have found pairs that look adjacent in 2-D but aren't especially close in 384 dimensions. **Projections discard most of the information.** Use them to build intuition; never as evidence.

### 4. Top-p sampling

```python
def top_p_filter(probabilities, p=0.9):
    """Keep the smallest set of tokens whose probabilities sum to >= p."""
    order = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[order]

    cumulative = np.cumsum(sorted_probs)
    # Keep everything up to and including the token that crosses p.
    keep_count = int(np.searchsorted(cumulative, p) + 1)

    filtered = np.zeros_like(probabilities)
    filtered[order[:keep_count]] = sorted_probs[:keep_count]
    return filtered / filtered.sum()          # renormalise
```

The point of top-p over top-k: **the candidate set adapts.** When the model is confident, one token may hold 95% of the mass and top-p keeps just that one. When it's uncertain, the set widens automatically. Top-k applies the same fixed cutoff either way.

### 5. Cost modelling

The shape of the answer, using illustrative rates:

- **Indexing:** 10,000 × 800 = 8M tokens. At embedding prices (a few cents per million) this is **well under a dollar, once.**
- **Queries:** 1,000 users × 5 questions × 30 days = 150,000 calls/month. Each sends ~4 chunks (~3,200 tokens) plus generates a response. That's roughly **500M+ tokens/month** through a generation model.

**The query cost dwarfs the indexing cost — by three or four orders of magnitude.** Most people guess the opposite, because embedding the whole corpus *sounds* like the expensive part.

The practical consequences, all of which Module 13 formalises: cache aggressively, retrieve fewer and better chunks rather than more, and use the smallest model that passes your evaluation. **Retrieval quality is a cost lever, not just a quality lever** — better retrieval means fewer chunks means a smaller bill.

---

## Ready for Module 4?

- [ ] You can explain why cosine ignores magnitude, and why that's desirable for text
- [ ] You know which metric sorts the other way
- [ ] You can explain why normalising makes all three agree
- [ ] You can state why the query and documents must share an embedding model
- [ ] You can explain why counting letters in "strawberry" is hard
- [ ] You know why pure semantic search fails on `"SKU-4471"`

**Next: [Module 4 — Transformers & Model Architecture](../../modules/04-transformers.md).** It answers the question this module left open — how the same word gets different vectors in different sentences — and in Lab 4 you'll implement attention yourself in about 20 lines of NumPy.

---

<div align="center">

**[⬅ Back to Lab 3](README.md)** · **[📖 Module 3](../../modules/03-tokens-embeddings-similarity.md)** · **[🏠 README](../../README.md)**

</div>
