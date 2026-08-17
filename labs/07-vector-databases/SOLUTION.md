# Lab 7 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `brute_force_search`

```python
def brute_force_search(query, vectors, k):
    scores = vectors @ query
    return np.argsort(scores)[::-1][:k]
```

One matrix-vector product scores the entire corpus. Because everything is normalised, these dot products *are* cosine similarities (Module 3 §3.6).

**Why this task exists at all.** It's the least interesting function in the lab and the most important: it's your **ground truth**. Without an exact baseline you cannot compute recall, and without recall you cannot tell a well-tuned index from a badly tuned one — because both return results and both look fine.

> **🔑 Build the Flat index before you need it.** Retrofitting a baseline after a quality complaint tells you nothing about whether the index was ever good.

Note that `[:k]` with `k` larger than the corpus is safe — Python slicing past the end just returns everything.

---

## Task 2 — `build_ivf_index`

```python
similarities = vectors @ centroids.T          # (n_vectors, n_clusters)
assignments = similarities.argmax(axis=1)

inverted_lists = {cluster: [] for cluster in range(len(centroids))}
for vector_index, cluster in enumerate(assignments):
    inverted_lists[int(cluster)].append(vector_index)
```

### Why pre-create every key

```python
# ❌ Only creates keys for clusters that got members
lists = {}
for i, c in enumerate(assignments):
    lists.setdefault(int(c), []).append(i)
```

A cluster can legitimately attract **zero** vectors — k-means initialises centroids from random points, and one may end up in a sparse region. When `ivf_search` probes that centroid, the dict lookup raises `KeyError`.

There's a check for it (`build_ivf_index includes empty clusters as keys`) using deliberately far-apart centroids so one is guaranteed empty.

This is a small instance of a general habit: **make the data structure total over its key space** so callers don't need to know which keys happen to exist.

### The name

"Inverted file" comes from text search. A normal index maps document → words; an *inverted* index maps word → documents. Here it's cluster → vectors, so a query about a cluster immediately yields its members.

---

## Task 3 — `ivf_search`

```python
centroid_scores = centroids @ query
probed = np.argsort(centroid_scores)[::-1][:n_probe]

candidates = [index for cluster in probed
              for index in inverted_lists[int(cluster)]]
if not candidates:
    return np.array([], dtype=int)

candidates = np.array(candidates, dtype=int)
candidate_scores = vectors[candidates] @ query
best_positions = np.argsort(candidate_scores)[::-1][:k]
return candidates[best_positions]           # <- the critical line
```

### Where the saving comes from

Two comparisons instead of one:

1. Query vs **centroids** — 32 comparisons instead of 2,000
2. Query vs **candidates in probed clusters** — a fraction of the corpus

With `n_probe=2` of 32 clusters, you touch roughly 6% of the data. That's the whole idea.

### The index-mapping trap

```python
return best_positions       # ❌ positions WITHIN candidates
return candidates[best_positions]   # ✅ positions in the corpus
```

`argsort` on `candidate_scores` returns indices into `candidate_scores` — which is a 4-element array if you had 4 candidates. Those numbers are `0..3`, not corpus indices.

**The bug returns valid-looking integers pointing at the wrong documents.** Nothing crashes. In a RAG system you'd retrieve the wrong text and answer confidently from it.

There's a dedicated check (`ivf_search returns ORIGINAL indices, not candidate positions`) constructed so the two interpretations give different answers — a symmetric test case would pass either way.

Task 6 has the identical trap.

### The `n_probe = n_clusters` invariant

One check asserts that probing *every* cluster gives exactly the brute-force result. That's a useful property to hold onto: **IVF with maximum `n_probe` degenerates to Flat.** If that check fails, your candidate gathering or index mapping is wrong, because with all clusters probed the candidate set is the whole corpus.

---

## Task 4 — `recall_at_k`

```python
exact_set = {int(i) for i in exact}
if not exact_set:
    return 1.0
approximate_set = {int(i) for i in approximate}
return len(approximate_set & exact_set) / len(exact_set)
```

### Why sets, and why order doesn't matter

Recall asks **which** results were found, not in what order. `[3, 2, 1]` versus `[1, 2, 3]` is perfect recall either way — there's a check for this.

Ranking *quality* is a different question, measured by NDCG or MRR. Those matter when the order of results affects the user. For RAG they matter less than you'd think: you're feeding the top-k to an LLM, which reads all of them.

### Why empty ground truth is `1.0`

Vacuously true — there was nothing to miss — and it avoids dividing by zero. The opposite case (empty *results* against non-empty ground truth) correctly gives `0.0`, and both are tested.

### Why this is the most important function here

Module 7 §7.5 makes the argument; here's the concrete version. A mistuned index:

- Throws no errors
- Has excellent latency (that's the problem — it's fast *because* it's skipping work)
- Returns plausible documents

**Every signal in your monitoring looks healthy.** The only thing that reveals the problem is comparing against exact results, which means you need this function and a Flat baseline. Stretch challenge 4 turns it into a regression test, and that's the version worth having in a real project.

---

## Task 5 — `filter_by_metadata`

```python
def _matches(record, key, condition):
    if key not in record:
        return False
    value = record[key]
    if isinstance(condition, dict):
        for operator, operand in condition.items():
            if operator == "$gte" and not value >= operand:
                return False
            ...
        return True
    return value == condition
```

Three design decisions:

**A missing key never matches.** `filter_by_metadata(metas, {"nonexistent": 1})` returns nothing rather than raising. For a filter, "this record has no such field" means "doesn't match" — raising would make heterogeneous metadata unusable.

**`all()` over an empty dict is `True`**, so no filters means everything matches. That falls out of Python's semantics rather than needing a special case, and it's the behaviour you want.

**Operators live in a nested dict.** `{"year": {"$gte": 2024}}` versus `{"year": 2024}`. This is the convention Chroma, MongoDB and most vector databases use, so the shape will look familiar when you meet the real thing.

---

## Task 6 — `search_with_filter`

```python
allowed = filter_by_metadata(metadatas, filters)
if len(allowed) == 0:
    return np.array([], dtype=int)
scores = vectors[allowed] @ query
best_positions = np.argsort(scores)[::-1][:k]
return allowed[best_positions]
```

Same index-mapping step as task 3, and one property worth naming explicitly:

> **Documents outside the filter are never scored.** Not scored and discarded — never touched.

That's the security difference between pre- and post-filtering. `solution.py`'s third demo makes it vivid: with post-filtering, secret documents are read into your process before being dropped, one log line or stack trace from leaking.

### The check that surprises people

```python
search_with_filter(query=[1, 0], ..., {"topic": "finance"}, k=2)  ->  [3, 2]
```

Not `[2, 3]`. Records 2 and 3 are the finance ones, but record 3's vector scores *higher* against the query. **Results are ranked by similarity, not index order** — obvious once stated, and easy to assume otherwise when writing the test.

(I made exactly that mistake writing this lab, and the test caught it. Which is the argument for tests with hand-computed expected values.)

---

## The experiments — discussion

### Experiment 1: the curve

```
    n_probe   recall@10   % scanned   speedup
          1        0.60        3.8%     26.6x
          2        0.85        6.6%     15.1x
          4        1.00       12.3%      8.1x
          8        1.00       24.2%      4.1x
         32        1.00      100.0%      1.0x
```

**1. Where would you ship?** `n_probe=4` looks tempting — perfect recall at an 8× speedup. But the honest answer depends entirely on the task:

| Task | Acceptable recall | Why |
|---|---|---|
| RAG feeding 20 chunks to an LLM | 0.80–0.90 | The LLM sees plenty of context; one missed chunk rarely changes the answer |
| Product search | 0.90+ | A missing top result is visible to the user |
| Legal/medical discovery | ~1.0, or Flat | Missing a document has consequences |
| De-duplication | 0.95+ | Missed duplicates accumulate |

**"Good enough" is a product decision, not a technical one.** The engineer's job is to *measure* the trade so someone can make that decision with numbers.

**2. Changing `n_clusters`.** Fewer clusters (8) means each is larger, so `n_probe=1` scans ~12% of the data — better recall per probe, less speedup. More clusters (128) means smaller clusters, so `n_probe=1` scans ~1% — much faster, much worse recall, and you need higher `n_probe`.

The heuristic `n_clusters ≈ √N` balances these. It's a starting point, not an answer — the point of the exercise is that you can now find the right value by measuring.

**3. Increasing `spread` to 0.4.** Recall drops noticeably. Overlapping clusters mean a query's true neighbours are more likely to sit in a *different* cluster than the query's own — exactly IVF's failure mode (Module 7 §7.4).

**This maps onto something real:** a corpus of clearly-separated topics indexes well; a corpus where everything is vaguely about the same subject indexes poorly. Your data's structure determines your achievable recall, and no amount of tuning fixes structureless data.

### Experiment 2: structure

```
  clustered (like real embeddings)   p1: 0.60  p2: 0.85  p4: 1.00  p8: 1.00
  random (no structure at all)       p1: 0.24  p2: 0.36  p4: 0.55  p8: 0.74
```

Same algorithm, same parameters. The only difference is whether the data has structure to exploit.

This is the honest foundation under every ANN benchmark: **the numbers are a property of the algorithm *and* the dataset together.** A blog post reporting "95% recall at `ef_search=64`" measured that on their corpus with their embedding model. Yours will differ.

There's also a deeper point about high-dimensional geometry here. In many dimensions, random points are nearly equidistant from each other — the "curse of dimensionality" — so there are no meaningful neighbourhoods to find. Real embeddings escape this because they occupy a low-dimensional structure *within* the high-dimensional space. ANN indexes work by finding that structure.

### Experiment 3: filtering

```
  POST-filter, fetch top-10    -> 0 legal docs survive
  POST-filter, fetch top-50    -> 2 legal docs survive
  POST-filter, fetch top-200   -> 2 legal docs survive
  POST-filter, fetch top-1000  -> 10 legal docs survive  [OK]

  PRE-filter                   -> 5 results  [OK]
```

Post-filtering had to retrieve the **entire corpus** to return 5 results. At that point you've done strictly more work than brute force *and* added complexity.

Note the plateau between top-50 and top-200 — both yield 2 documents. Over-fetching helps only until it doesn't; the surviving count depends on where the rare documents happen to rank, which you can't control.

**The practical rule** (Module 7 §7.6): post-filter for broad filters with over-fetching; pre-filter for narrow ones; and for very narrow filters, skip the vector index entirely and brute-force the subset. Stretch challenge 5 has you find the crossover point.

---

## Part 2 — Discussion

**1. Distance vs score direction.**

| Tool | Returns | Better is |
|---|---|---|
| FAISS `IndexFlatIP` | Inner product **score** | **Higher** |
| Chroma (cosine space) | **Distance** (1 − cosine similarity) | **Lower** |

Mixing these up is the silent-bug-from-Module-3 again: sort the wrong way and you return your worst matches with total confidence. **Always check which convention your tool uses**, and write a test that asserts a known-similar pair ranks first.

**2. Removing `normalize_embeddings=True`.** The ranking changes, and longer or more "energetic" documents start winning regardless of relevance. `IndexFlatIP` is *inner product*, which equals cosine similarity **only for unit vectors** (Module 3 §3.6). Without normalisation you're ranking by a mix of direction and magnitude.

**3. Passing `float64`.** FAISS raises an assertion or type error — it requires `float32`. NumPy's default is `float64`, so this bites almost everyone once. Cast explicitly: `.astype("float32")`.

**4. The medical filter with an irrelevant query.** Chroma returns medical documents with poor similarity scores. **That is correct behaviour** — the filter is a hard constraint, similarity is a ranking within it.

The lesson: **a filter doesn't make results relevant, it makes them permitted.** If your filter narrows to documents that don't answer the question, you'll get the least-bad of a bad set. In a RAG system that's a real failure mode, and it's why Module 8 covers checking whether retrieved context is actually relevant before generating from it.

**5. Not passing your own embeddings.** Chroma would use its default embedding function. If that default changes in a later library version, new queries get embedded by a different model than your stored documents — and the vectors become incomparable (Module 3 §3.7). **Nothing errors; results become nonsense.**

This is why Module 7 §7.8 says to always know which model produced your vectors, and to record it in metadata.

---

## 🚀 Stretch — Discussion

### 1. IVF+PQ

You'll find quantization saves large amounts of memory and costs recall, while doing relatively little for speed. **PQ is a memory play.** Reach for it when you cannot fit raw vectors in RAM; not otherwise.

### 3. Why `n_probe=40` came out *slower* than brute force

`solution.py`'s timing demo shows something worth explaining:

```
  index               recall@10   ms/query   speedup
  Flat (exact)             1.00       2.65      1.0x
  IVF n_probe=1            0.49       0.28      9.4x
  IVF n_probe=40           0.99       6.76      0.4x     <- SLOWER than Flat
```

At `n_probe=40` the IVF search is **2.5× slower than brute force** while returning worse results. Two reasons:

1. **Python-level candidate gathering.** The list comprehension collecting indices from 40 clusters runs in interpreted Python, while `vectors @ query` is a single optimised BLAS call. At this scale the overhead dominates the saving.
2. **20,000 vectors is small.** One matrix product over 20,000 × 128 floats is genuinely fast. There's not much to save.

**Both point at the same conclusion**, and it's the honest one: at small N, brute force wins. Module 7 §7.1 says "you probably don't need a vector database below ~100k vectors", and the timing demo demonstrates it rather than asserting it.

A production IVF implementation (FAISS's, written in C++ with SIMD) doesn't have the Python overhead — but it still can't beat brute force when brute force is already 2 ms.

### 4. The recall regression test

**The most valuable thing in this lab for real work.** A test that fails when retrieval quality drops is the only defence against silent degradation.

Two details for a real version: pin the query set and the ground truth (recompute ground truth only when the corpus changes intentionally), and assert on *mean* recall plus a *floor* on the worst query — a good average can hide one catastrophically bad query class.

### 5. The brute-force crossover

Higher than most people expect. Filtering to 1% of 50,000 vectors leaves 500 — brute-forcing 500 vectors takes well under a millisecond, and no index will beat that once you account for the filtered-search overhead.

**A lot of "we need a vector database" is really "we need a `WHERE` clause".** This is the practical case for `pgvector` in Module 7 §7.9: if your filters are usually narrow, the vector search is the easy part and the filtering is what matters — and a real database is very good at filtering.

### 6. The model-mismatch failure

The instructive part is that **nothing errors.** You get numbers, rankings, and confident nonsense. There's no runtime signal at all.

That's why the mitigations are procedural rather than technical: record the embedding model and version in your index metadata, and add a startup assertion that the query model matches the index model. You cannot detect this failure from the results alone.

---

## Ready for Module 8?

- [ ] You can explain what "approximate" means in ANN, and what you give up
- [ ] You can define recall@k and say why you need a Flat baseline
- [ ] You know why ANN depends on the data having structure
- [ ] You can explain why recall degradation is invisible in production
- [ ] You know when to pre-filter versus post-filter, and why post-filtering is unsafe for permissions
- [ ] You can say why an index is locked to its embedding model
- [ ] You know roughly when brute force is simply the better choice

**Next: Module 8 — Retrieval-Augmented Generation.** Everything here becomes the retrieval half of a real pipeline: chunking strategies, hybrid search (finally fixing `SKU-4471` from Lab 3), re-ranking, and the document Q&A bot with citations that's the course's first portfolio piece.

---

<div align="center">

**[⬅ Back to Lab 7](README.md)** · **[📖 Module 7](../../modules/07-vector-databases.md)** · **[🏠 README](../../README.md)**

</div>
