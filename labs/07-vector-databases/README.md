# 🧪 Lab 7: Build an ANN Index and Measure It

**Module:** [7 — Embeddings & Vector Databases](../../modules/07-vector-databases.md)

---

## Objective

Implement a real approximate nearest-neighbour index from scratch, then **measure what the approximation costs you.** That measurement habit is the whole point — a mistuned index returns plausible-but-worse results silently, forever.

By the end you will have:

1. **Implemented brute-force search** — the exact baseline you measure against
2. **Built a working IVF index** — cluster, then probe
3. **Measured the recall–latency curve** and found where it saturates
4. **Proved** that ANN only works because real embeddings have structure
5. **Demonstrated** how post-filtering silently returns too few results
6. **Done the same with FAISS and Chroma**, with metadata filtering

## Expected outcome

`python starter.py` reports **26 of 26 checks passing**, then three experiments — including a table showing 85% recall while scanning under 7% of the data, and a demonstration that post-filtering on a 1% category needs to fetch the *entire corpus* to return 5 results.

## Requirements

| | |
|---|---|
| **Packages** | **`numpy` only** for Part 1 |
| **Part 2** | `faiss-cpu`, `chromadb`, `sentence-transformers` |
| **API key** | **None.** Everything runs locally. |
| **Cost** | Free |
| **Time** | ~50 minutes |

```powershell
pip install numpy                                   # Part 1
pip install faiss-cpu chromadb sentence-transformers  # Part 2
```

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 6 tasks, 26-check self-test, 3 experiments. |
| `solution.py` | Reference solution + 3 demos with real timings. |
| `SOLUTION.md` | The reasoning, including the index-mapping trap. |

---

## Part 1 — Build and measure (30 min)

```powershell
python labs/07-vector-databases/starter.py
```

Two helpers are **provided** so you can focus on the index rather than the clustering: `kmeans()` and `make_clustered_corpus()`.

| Task | Function | Key idea | Module 7 § |
|---|---|---|---|
| 1 | `brute_force_search` | The exact baseline | 7.4 |
| 2 | `build_ivf_index` | Assign vectors to clusters | 7.4 |
| 3 | `ivf_search` | Probe only the nearest clusters | 7.4, 7.5 |
| 4 | `recall_at_k` | **The measurement that matters** | 7.5 |
| 5 | `filter_by_metadata` | Exact match, `$gte`, `$lte`, `$in` | 7.6 |
| 6 | `search_with_filter` | Pre-filtered search | 7.6 |

### The trap that catches everyone

Tasks 3 and 6 both narrow the corpus and then search the subset. When you `argsort` the subset's scores, you get **positions within the subset**, not positions in the full corpus:

```python
candidates = np.array([17, 42, 88, 103])         # corpus indices
scores = vectors[candidates] @ query              # 4 scores
best = np.argsort(scores)[::-1][:2]               # e.g. [2, 0]  <- positions in `candidates`!

return best                # ❌ returns [2, 0]   — wrong documents
return candidates[best]    # ✅ returns [88, 17] — correct
```

The bug doesn't crash. It returns valid-looking indices pointing at the wrong documents. There's a dedicated check for it in each task.

### Two smaller gotchas

**Empty clusters must still be keys.** In task 2, a cluster can legitimately attract no vectors. If you only create keys for clusters that got members, `ivf_search` raises `KeyError` when it probes an empty one. Create a list for every cluster id up front.

**`recall_at_k` on empty ground truth returns `1.0`.** Nothing to miss is vacuously perfect — and it avoids dividing by zero.

**✅ Part 1 complete at `All 26 checks passed.`**

---

## The three experiments

These run automatically once the checks pass. **They're the actual content of this lab** — read the output, don't just watch it scroll.

### Experiment 1 — the recall–latency curve

```
    n_probe   recall@10   % scanned   speedup
          1        0.60        3.8%     26.6x
          2        0.85        6.6%     15.1x
          4        1.00       12.3%      8.1x
          8        1.00       24.2%      4.1x
         32        1.00      100.0%      1.0x
```

**Two things to take from this.** At `n_probe=2` you get 85% recall for a 15× speedup — that's the trade. And it **saturates at `n_probe=4`**: going to 32 costs 8× the work for zero extra recall.

Then answer:

1. **At what `n_probe` would you ship?** Justify it — and note the answer differs for a RAG system (which retrieves 20 chunks and tolerates misses) versus legal discovery (which cannot miss a document).
2. **Change `n_clusters` from 32 to 8, then to 128.** How does the curve move? What's the relationship between `n_clusters` and the recall you get per unit of work?
3. **Change `spread` in `make_clustered_corpus` from 0.15 to 0.4.** Clusters now overlap more. What happens to recall, and why?

### Experiment 2 — structure is what makes ANN work

```
  clustered (like real embeddings)   p1: 0.60  p2: 0.85  p4: 1.00  p8: 1.00
  random (no structure at all)       p1: 0.24  p2: 0.36  p4: 0.55  p8: 0.74
```

Same algorithm, same settings, very different results. **ANN indexes exploit the fact that real embeddings cluster by topic.** Remove the structure and clustering tells you nothing.

This is why published recall figures don't transfer between datasets — and why you measure on yours.

### Experiment 3 — post-filtering fails quietly

```
  corpus: 1000 docs, of which 10 are dept='legal' (1.0%)

  POST-filter, fetch top-10    -> 0 legal docs survive  [TOO FEW]
  POST-filter, fetch top-50    -> 2 legal docs survive  [TOO FEW]
  POST-filter, fetch top-200   -> 2 legal docs survive  [TOO FEW]
  POST-filter, fetch top-1000  -> 10 legal docs survive  [OK]

  PRE-filter                   -> 5 results  [OK]
```

Post-filtering had to fetch the **entire corpus** to return 5 results. Pre-filtering returned them immediately.

**And note the security implication**: post-filtering means documents outside the filter *were* retrieved into your process before being discarded. For a permissions filter, that's the difference between a design and a data breach (Module 7 §7.6).

---

## Part 2 — FAISS and Chroma (20 min)

Create `real_vector_db.py` in the repo root:

```python
"""real_vector_db.py - the same ideas with production tools."""

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "The stock market crashed today.",
    "Investors lost money in the financial crisis.",
    "LLMs can generate human-like text.",
    "Artificial intelligence writes like a human.",
    "Metformin is dosed up to 2000 mg per day.",
    "Insulin therapy requires careful glucose monitoring.",
]
metadatas = [
    {"topic": "animals", "year": 2023}, {"topic": "animals", "year": 2024},
    {"topic": "finance", "year": 2024}, {"topic": "finance", "year": 2022},
    {"topic": "ai", "year": 2024},      {"topic": "ai", "year": 2023},
    {"topic": "medical", "year": 2024}, {"topic": "medical", "year": 2023},
]

# normalize_embeddings=True so inner product == cosine similarity
embeddings = np.asarray(
    model.encode(documents, normalize_embeddings=True), dtype="float32")
dimension = embeddings.shape[1]
print(f"{len(documents)} documents -> {embeddings.shape}\n")


# ============================================================
# FAISS: Flat (exact) vs HNSW (approximate)
# ============================================================
import faiss

query = np.asarray([model.encode("cat on a mat", normalize_embeddings=True)],
                   dtype="float32")

flat = faiss.IndexFlatIP(dimension)          # exact: the ground truth
flat.add(embeddings)
exact_scores, exact_ids = flat.search(query, k=3)

print("--- FAISS Flat (exact) ---")
for score, idx in zip(exact_scores[0], exact_ids[0]):
    print(f"  {score:.3f}  {documents[idx]}")

hnsw = faiss.IndexHNSWFlat(dimension, 16)    # 16 = M, links per node
hnsw.hnsw.efConstruction = 64
hnsw.add(embeddings)                          # no .train() needed for HNSW
hnsw.hnsw.efSearch = 32
approx_scores, approx_ids = hnsw.search(query, k=3)

print("\n--- FAISS HNSW (approximate) ---")
for score, idx in zip(approx_scores[0], approx_ids[0]):
    print(f"  {score:.3f}  {documents[idx]}")

recall = len(set(approx_ids[0]) & set(exact_ids[0])) / len(exact_ids[0])
print(f"\n  recall@3 vs Flat: {recall:.2f}")
print("  (8 documents is far too few for HNSW to help - but this is HOW")
print("   you measure it, which is the transferable part)")


# ============================================================
# Chroma: documents, embeddings, metadata and filtering together
# ============================================================
import chromadb

client = chromadb.Client()          # in-memory; use PersistentClient to keep it
collection = client.get_or_create_collection(
    name="lab7", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings.tolist(),      # OUR vectors, so we know the model
    metadatas=metadatas,
)

print("\n--- Chroma, unfiltered ---")
results = collection.query(query_texts=["cat on a mat"], n_results=3)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  distance {dist:.3f}  {doc}")

print("\n--- Chroma, filtered to topic='medical' ---")
results = collection.query(
    query_texts=["cat on a mat"], n_results=3, where={"topic": "medical"})
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  distance {dist:.3f}  {doc}")
print("  (note: a poor semantic match, but it RESPECTED the filter)")

print("\n--- Chroma, combined filter ---")
results = collection.query(
    query_texts=["AI and text generation"], n_results=5,
    where={"$and": [{"year": {"$gte": 2024}}, {"topic": {"$in": ["ai", "finance"]}}]})
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['topic']} {meta['year']}]  {doc}")
```

```powershell
python real_vector_db.py
```

### Answer these

1. **Chroma returns `distances`, FAISS returns `scores`.** Which direction is "better" for each? (Module 7 §7.4 — and Module 3 §3.6 if you need the reminder.)
2. **Remove `normalize_embeddings=True`** and re-run the FAISS Flat search. What happens to the ranking, and why? (`IndexFlatIP` is inner product.)
3. **Cast `embeddings` to `float64`** and pass it to FAISS. What error do you get?
4. **Try the medical filter with a query that has no medical relevance.** Chroma still returns medical documents. Is that correct behaviour? What does it tell you about how a filter interacts with relevance?
5. **In the Chroma example, we passed our own `embeddings`.** What would break if we hadn't, and Chroma's default embedding model changed in a later version?

**✅ Part 2 complete when you have both tools working and can explain the distance/score direction.**

---

## 🚀 Stretch Challenges

### 1. Add IVF+PQ to your framework

Implement crude product quantization: split each vector into `m` sub-vectors, run k-means on each sub-space, store only the centroid ids.

Measure: memory saved versus recall lost. **You'll see why quantization is a memory play, not a speed play.**

### 2. Tune a real HNSW index properly

Build a 50,000-vector corpus with `make_clustered_corpus`, then sweep FAISS HNSW's `M`, `efConstruction` and `efSearch`. For each, measure recall against Flat and query latency.

Plot recall against latency. **You've just produced the curve every ANN benchmark shows** — and now you know it's dataset-specific.

### 3. Measure the memory cost for real

```python
import faiss, numpy as np
# Compare Flat vs HNSW vs IVF+PQ on the same vectors
print(faiss.serialize_index(index).nbytes / 1e6, "MB")
```

Does HNSW really cost 1.5–2× the raw vectors? Check the claim.

### 4. Build the recall regression test

Write a script that: loads a fixed set of 50 queries, computes exact results with Flat, computes approximate results with your production settings, asserts mean recall is above a threshold, and fails loudly otherwise.

**This is the single most valuable thing in this lab for real work.** It's the only way to catch silent retrieval degradation (Module 7 §7.10).

### 5. Find where a filter should skip the index entirely

At what filter selectivity does brute-forcing the subset beat ANN search? Measure it: filter to 0.1%, 1%, 10%, 50% of a 50,000-vector corpus and time both approaches.

**The crossover point is higher than most people expect** — and it means a lot of "we need a vector database for this" is really "we need a `WHERE` clause".

### 6. Prove the model-mismatch failure

Embed your documents with `all-MiniLM-L6-v2` and your query with `all-mpnet-base-v2`. Both work. Both produce numbers. The results are garbage.

**Note that nothing errors.** This is the failure mode from Module 3 §3.7 and Module 7 §7.10, and seeing it silently produce nonsense is more instructive than reading about it.

---

## When you're done

1. Attempt Part 1 before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why `n_probe=40` came out *slower* than brute force in the timing demo.
3. Run `python solution.py` for three demos with real timings, a storage-cost table, and an access-control demonstration.

**Next:** Module 8 — Retrieval-Augmented Generation. Everything in this module becomes the retrieval half of a RAG pipeline: chunking, hybrid search (the `SKU-4471` fix), re-ranking, and the full document Q&A bot with citations.
