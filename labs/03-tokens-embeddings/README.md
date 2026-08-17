# 🧪 Lab 3: From Text to Meaning

**Module:** [3 — Tokens, Embeddings & Similarity](../../modules/03-tokens-embeddings-similarity.md)

---

## Objective

Turn text into numbers, measure meaning, and build a search engine that finds documents by what they mean rather than which words they contain.

By the end you will have:

1. **Counted tokens exactly** and found where the 4-chars-per-token rule breaks
2. **Implemented three similarity metrics** from scratch in NumPy
3. **Proved** that normalising vectors makes all three agree
4. **Built a working semantic search engine** in about 10 lines
5. **Broken it deliberately** to find its limits — which is what Module 8 fixes

## Expected outcome

`python starter.py` reports **13 of 13 checks passing**, then runs two demos: a token-count table, and a search that finds *"A kitten rested on the rug"* when you query *"cat on a mat"* — despite zero shared words.

## Requirements

| | |
|---|---|
| **Packages** | `numpy` (Part 1) · `tiktoken`, `sentence-transformers` (Part 2) |
| **API key** | **None.** Everything runs locally. |
| **Cost** | Free |
| **Download** | ~90 MB, once, for the embedding model |
| **Time** | ~45 minutes |

```powershell
pip install numpy tiktoken sentence-transformers
```

> **💡 Part 1 needs only NumPy**, so the self-test always runs. The Part 2 demos skip gracefully with a message if the other packages aren't installed — you can do the whole maths section on a plane.

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 7 `TODO`s + a self-test + two demos. |
| `solution.py` | Reference solution with three extra demos. |
| `SOLUTION.md` | The reasoning behind each answer. |

---

## Part 1 — The maths (20 min)

Open `starter.py` and run it before writing anything:

```powershell
python labs/03-tokens-embeddings/starter.py
```

You'll see 9 of 13 failing. Fill in tasks 1–5, re-running as you go.

| Task | Function | Key idea | Module 3 § |
|---|---|---|---|
| 1 | `cosine_similarity` | Angle only — magnitude cancels out | 3.6 |
| 2 | `dot_product` | Magnitude-sensitive | 3.6 |
| 3 | `euclidean_distance` | **Lower is better** — opposite direction | 3.6 |
| 4 | `normalise` | Length 1, direction preserved | 3.6 |
| 5 | `top_k_indices` | `argsort` is *ascending* | 3.7 |

**Watch for two traps:**

- **Task 3's direction.** Euclidean distance is a *distance*. Lower means more similar — the reverse of cosine. Getting this backwards is a silent bug that returns your worst matches with total confidence.
- **Task 5's sort order.** `np.argsort` sorts lowest-to-highest. You want highest first, so you must reverse it.

**The last check is a bonus** that proves something from §3.6: once vectors are normalised, `dot_product` and `cosine_similarity` return the *same number*. If you implemented tasks 1, 2 and 4 correctly it passes automatically. That equivalence is why production vector databases normalise on ingest — you get cosine's semantics at the dot product's speed.

**✅ Part 1 complete at `All 13 checks passed.`**

---

## Part 2 — Real text (25 min)

### Task 6 — Count tokens exactly

Implement `count_tokens` with `tiktoken`. The demo then prints a table across five kinds of text.

**Study that table.** Plain English lands near 4 characters per token. Code, long numbers and Spanish all fall well below it — meaning *more* tokens and a *higher bill* than the rule of thumb suggests.

Then answer these:

1. Which sample had the **worst** chars-per-token ratio? Why?
2. What does that imply for cost if your users write in Hindi or Arabic rather than English?
3. Run `encoder.encode("strawberry")` and decode each ID separately. **How many tokens, and where are the splits?** Now explain why "how many r's in strawberry?" is genuinely hard for an LLM.

### Task 7 — Semantic search

This is the important one. Implement `search`, then read the results carefully.

For the query `"cat on a mat"`, look at the **second** result. It shares no words at all with the query. It was found because "kitten"/"cat" and "rug"/"mat" sit near each other in the embedding space.

**That is the engine inside every RAG system.** Modules 7 and 8 scale exactly this up.

Then answer:

1. What score did the unrelated documents get? Is there a clean gap between relevant and irrelevant, or is it blurry?
2. Try `search("SKU-4471", ...)`. What comes back, and with what confidence? What does that tell you about semantic search and exact identifiers?
3. Write a query that *should* match a document but doesn't. What broke?

**✅ Part 2 complete when the search returns sensible rankings for all three queries.**

---

## 🚀 Stretch Challenges

**Optional.** Nothing later depends on these.

### 1. Find where semantic search fails

Build a corpus of 15–20 documents on a topic you know well. Then hunt for failures:

- **Negation:** does *"coffee that isn't bitter"* match *"bitter coffee"*? (It usually does — embeddings handle negation poorly, which is a genuine production problem.)
- **Exact codes:** product IDs, version numbers, error codes
- **Very short queries:** one or two words
- **Very long documents:** does a 500-word document retrieve well as a single vector?

Write down each failure. **You've just derived the motivation for Module 8's chunking and hybrid search from first principles**, which is a better way to learn it than being told.

### 2. Compare embedding models

```python
for name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    model = SentenceTransformer(name)
    # same corpus, same queries - compare rankings and dimensions
```

`all-mpnet-base-v2` is larger (768 dimensions vs 384). **Is it better on your corpus? Enough to justify twice the storage and slower search?** Answering that with data rather than assumption is a real production skill.

### 3. Visualise the space

Use the PCA snippet from §3.5 on 15–20 words across three or four categories. Do the clusters appear?

Then check the caveat: pick two points that look adjacent in the plot and compute their *actual* cosine similarity in the full 384 dimensions. **Are they really as close as they look?**

### 4. Implement top-p sampling

Using the decoding functions from §3.8, implement nucleus sampling:

```python
def top_p_filter(probabilities: np.ndarray, p: float = 0.9) -> np.ndarray:
    """Keep the smallest set of tokens whose probabilities sum to >= p."""
    # Sort descending, take the cumulative sum, cut off past p, renormalise.
    ...
```

Then compare, on the same logits: greedy, temperature 0.7, and top-p 0.9. **How does the set of candidate tokens differ?**

### 5. Cost-model a real workload

You're building a RAG system over 10,000 documents averaging 800 tokens each.

- What does it cost to embed the corpus once?
- If 1,000 users each ask 5 questions a day, with 4 retrieved chunks per question, what's the monthly LLM bill?
- Which is bigger: the one-off indexing cost or the monthly query cost?

Use real published prices. **The answer surprises most people**, and it's the reasoning Module 13 formalises.

---

## When you're done

1. Attempt everything before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why "cosine similarity of 0.8" doesn't mean "80% similar".
3. Run `python solution.py` for three extra demos, including a direct demonstration that unnormalised metrics disagree on ranking while normalised ones agree.

**Next:** [Module 4 — Transformers](../../modules/04-transformers.md), which answers the question Module 3 left hanging: *how does the same word get different vectors in different sentences?*
