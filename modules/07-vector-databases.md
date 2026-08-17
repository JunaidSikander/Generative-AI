# Module 7: Embeddings & Vector Databases

> **By the end of this module** you'll know why brute-force search stops working, how approximate nearest-neighbour indexes buy speed by trading accuracy, how to measure that trade-off rather than hope, and how to store and query embeddings for real with FAISS and Chroma — including metadata filtering, which is the feature most tutorials skip and most real systems need.

| | |
|---|---|
| **Time** | ~2 hours (70 min reading, 50 min lab) |
| **Prerequisites** | [Module 3](03-tokens-embeddings-similarity.md) especially — you need embeddings and cosine similarity |
| **Packages** | `numpy`, `faiss-cpu`, `chromadb`, `sentence-transformers` |
| **Cost** | Free — everything runs locally |

---

## Contents

- [7.0 Why This Matters](#70-why-this-matters)
- [7.1 Where Brute Force Breaks](#71-where-brute-force-breaks)
- [7.2 What a Vector Database Actually Is](#72-what-a-vector-database-actually-is)
- [7.3 The Indexing Pipeline](#73-the-indexing-pipeline)
- [7.4 ANN Indexes](#74-ann-indexes)
- [7.5 The Recall–Latency Trade-off](#75-the-recalllatency-trade-off)
- [7.6 Metadata Filtering](#76-metadata-filtering)
- [7.7 FAISS in Practice](#77-faiss-in-practice)
- [7.8 Chroma in Practice](#78-chroma-in-practice)
- [7.9 Choosing a Vector Database](#79-choosing-a-vector-database)
- [7.10 Production Concerns](#710-production-concerns)
- [🧪 Hands-On Lab 7](#-hands-on-lab-7)
- [✅ Key Takeaways](#-key-takeaways)
- [⚠️ Common Mistakes & Misconceptions](#️-common-mistakes--misconceptions)
- [📚 Going Deeper](#-going-deeper)

---

## 7.0 Why This Matters

In Module 3 you built a semantic search engine. Here's the line that did the searching:

```python
scores = doc_embeddings @ query_embedding      # compare against EVERY document
```

That works beautifully for six documents. It works fine for six thousand. At six million it's a problem, and at six hundred million it's impossible.

This module is about the layer that fixes it — and about one idea that's easy to miss:

> **A vector database doesn't find the *nearest* neighbours. It finds *probably* the nearest neighbours, quickly.**

That word "probably" is the whole engineering trade. You give up a small amount of accuracy for an enormous amount of speed. Understanding what you gave up — and measuring it — is the difference between a retrieval system that works and one that quietly returns second-rate results forever.

**The failure mode here is silent.** A badly tuned index doesn't crash or throw. It returns plausible-looking documents that just aren't the best matches, and your RAG system in Module 8 answers slightly wrong questions with complete confidence. Nobody notices for months.

So the lab has you implement an ANN index yourself and then *measure its recall*. That measurement habit is the point of the module.

---

## 7.1 Where Brute Force Breaks

### The arithmetic

Brute force compares the query against every stored vector. Cost grows linearly:

| Documents | Vectors compared | Rough time (384-dim, NumPy) |
|---|---|---|
| 1,000 | 1,000 | < 1 ms |
| 100,000 | 100,000 | ~20 ms |
| 1,000,000 | 1,000,000 | ~200 ms |
| 100,000,000 | 100,000,000 | ~20 s |

The last row is the problem. Twenty seconds per query is not a product. And that's *one* query — a hundred concurrent users makes it far worse.

### Memory is the other wall

Vectors are not small.

```python
# Storage for N vectors of D dimensions, as 32-bit floats:
#   bytes = N * D * 4

    1_000_000 * 384  * 4   =  1.5 GB      # small embedding model
    1_000_000 * 1536 * 4   =  6.1 GB      # larger embedding model
   10_000_000 * 1536 * 4   = 61.4 GB      # now you need a strategy
```

**Two immediate practical consequences:** embedding dimension is a *cost* decision as much as a quality one (Module 3 §3.5), and beyond a few million vectors you can't just hold everything in one process's memory.

### Be honest about when you need this

> **📌 You probably don't need a vector database yet.** Up to roughly 100,000 vectors, a NumPy array in memory is genuinely fine — often *faster* than a database, because there's no network hop, no serialisation, and no index overhead. It's also simpler to debug and impossible to misconfigure.
>
> Reach for a real vector store when you have: more vectors than fit comfortably in memory, a need to persist across restarts, metadata filtering, concurrent writers, or more than one process needing access.

The rest of this module assumes you've crossed one of those lines. If you haven't, knowing *why* the tools exist is still worth your time — but don't add infrastructure you don't need.

---

## 7.2 What a Vector Database Actually Is

A database purpose-built to store high-dimensional vectors and find nearest neighbours fast.

### What makes it different from a normal database

| | **Relational / document DB** | **Vector database** |
|---|---|---|
| Query means | "Find rows where `status = 'open'`" | "Find the 10 vectors closest to *this* one" |
| Index type | B-tree, hash | **Approximate nearest neighbour** (HNSW, IVF) |
| Match quality | Exact — a row either matches or doesn't | **Approximate and ranked** by distance |
| Query input | Values and predicates | A vector |

The key difference is that last row combined with the third: **there is no exact answer, only a ranking, and the index doesn't guarantee it found the best one.**

### The four things it does

| Capability | What it means |
|---|---|
| **ANN indexing** | Structures the vectors so search doesn't scan everything |
| **Metadata filtering** | "Nearest neighbours *where* `department = 'legal'` and `year >= 2023`" |
| **CRUD and upserts** | Add, update and delete vectors without rebuilding the whole index |
| **Scaling** | Sharding, replication, persistence across restarts |

**Metadata filtering is the one people underestimate**, and §7.6 is about it. In practice, "search these documents, but only the ones this user is allowed to see" is a requirement of almost every real system, and it interacts with ANN indexing in a way that's genuinely tricky.

---

## 7.3 The Indexing Pipeline

Four stages, and only the middle two are new.

```
   ┌──────────────┐
   │ 1. RAW DATA  │   documents, PDFs, web pages, code, images
   └──────┬───────┘
          │        (Module 8 adds CHUNKING here)
          ▼
   ┌──────────────┐
   │ 2. EMBED     │   one vector per chunk
   └──────┬───────┘   e.g. all-MiniLM-L6-v2 -> 384 numbers
          │
          ▼
   ┌──────────────┐
   │ 3. INDEX     │   build an ANN structure over the vectors
   └──────┬───────┘   + store metadata alongside each one
          │
          ▼
   ┌──────────────┐
   │ 4. QUERY     │   embed the query -> ANN search -> top-k
   └──────────────┘
```

### The asymmetry that shapes everything

| | Indexing | Querying |
|---|---|---|
| How often | **Once** (plus updates) | **Every request** |
| Latency budget | Minutes to hours is fine | Milliseconds |
| Cost | One-off | Per query, forever |

Because indexing happens once and querying happens constantly, **it's worth spending a lot of build-time effort to make queries fast.** That's the entire justification for ANN indexes: they're expensive to construct and cheap to search.

It also explains the cost result from Lab 3's stretch challenge — embedding a corpus once is negligible compared to the ongoing cost of serving queries against it.

---

## 7.4 ANN Indexes

**ANN** = Approximate Nearest Neighbour. Four families, and you'll meet all four in vector-database configuration options.

### 1. Flat (brute force) — exact, and the baseline

Compare against everything. No index at all.

```
query ──▶ [v1] [v2] [v3] ... [vN]     compare all N
```

| | |
|---|---|
| **Recall** | 100% — exact by definition |
| **Speed** | O(N) — linear |
| **Use when** | Fewer than ~100k vectors, or you need guaranteed-exact results |

**Don't skip Flat.** It's the only way to know what your approximate index is missing — you need exact results to measure recall against. Every serious ANN evaluation compares to Flat.

### 2. IVF (Inverted File) — cluster, then search a few clusters

The idea is intuitive: group similar vectors together up front, then only search the groups near your query.

```
  BUILD:  run k-means -> C centroids; assign every vector to its nearest

     cluster 0: [v3, v17, v88, ...]
     cluster 1: [v1, v42, ...]
     cluster 2: [v9, v23, v51, ...]
     ...

  SEARCH: find the n_probe centroids closest to the query,
          then scan ONLY those clusters

     query ──▶ nearest centroids: [2, 7]  ──▶ scan ~2/C of the data
```

| | |
|---|---|
| **Tuning knob** | `n_probe` — how many clusters to search |
| **Speed** | Roughly `n_probe / n_clusters` of brute force |
| **Trade-off** | Higher `n_probe` = better recall, slower |
| **Failure mode** | A true neighbour sitting just across a cluster boundary gets missed |

**You'll build this in the lab**, and measure exactly how recall responds to `n_probe`. It's about 20 lines and it makes the whole ANN concept concrete.

A common heuristic is `n_clusters ≈ √N`, then tune `n_probe` upward until recall is acceptable.

### 3. HNSW — a navigable graph. The production default.

**Hierarchical Navigable Small World.** The most widely used index in modern vector databases.

Build a multi-layer graph where each vector links to its near neighbours. Upper layers are sparse (long-range links); lower layers are dense (local links).

```
  Layer 2   ●─────────────────────●            few nodes, long jumps
             \                   /
  Layer 1     ●───────●─────────●              more nodes
               \     / \       /
  Layer 0       ●───●───●─────●───●──●         every node, short hops

  SEARCH: enter at the top, greedily hop toward the query,
          descend a layer, repeat. Coarse to fine.
```

It's the same idea as finding a house: motorway, then main road, then street, then door.

| | |
|---|---|
| **Tuning knobs** | `M` (links per node), `ef_construction` (build quality), `ef_search` (search effort) |
| **Speed** | Roughly O(log N) — this is why it wins |
| **Recall** | Very high, often 95%+ with sensible settings |
| **Downside** | **Memory-hungry** (stores the graph as well as the vectors); slower to build |

**Default to HNSW** unless you have a specific reason not to. The usual reason is memory.

### 4. Quantization (PQ) and hashing (LSH) — compress the vectors

Rather than organising the vectors, shrink them.

**Product Quantization** splits each vector into sub-vectors and replaces each with a codebook entry — turning 1536 floats (6 KB) into a few dozen bytes.

| | |
|---|---|
| **Wins** | Massive memory reduction — often 10–50× |
| **Costs** | Accuracy loss, because the vectors are genuinely lossy now |
| **Use when** | You cannot fit the raw vectors in RAM |

Often combined: `IVF + PQ` is a common configuration for very large collections. HNSW can be quantized too.

### Comparison

| Index | Recall | Speed | Memory | Build time | Use when |
|---|---|---|---|---|---|
| **Flat** | 100% | Slow (O(N)) | Vectors only | None | < 100k, or measuring recall |
| **IVF** | Tunable | Fast | Vectors + centroids | Moderate | Large, memory-conscious |
| **HNSW** | Very high | Fastest (O(log N)) | **High** (graph) | Slow | **The default** |
| **IVF+PQ** | Lower | Fast | **Lowest** | Moderate | Billions of vectors |

---

## 7.5 The Recall–Latency Trade-off

This is the central skill of the module, and it's a measurement skill rather than a knowledge one.

### What recall means here

**Recall@k** = of the *true* top-k nearest neighbours, what fraction did my index actually return?

```python
def recall_at_k(approximate: list, exact: list) -> float:
    """What fraction of the true top-k did the approximate search find?"""
    if not exact:
        return 1.0
    return len(set(approximate) & set(exact)) / len(exact)
```

Recall of 0.9 means you found 9 of the true top 10 and one result is an impostor — a document that ranks lower than something you missed.

### The curve

Here's real measured output from the lab's IVF implementation — 2,000 vectors, 64 dimensions, 32 clusters, recall averaged over 25 queries:

```
  n_probe   recall@10   % scanned   speedup
        1        0.60        3.8%     26.6x
        2        0.85        6.6%     15.1x
        4        1.00       12.3%      8.1x
        8        1.00       24.2%      4.1x
       32        1.00      100.0%      1.0x     (= brute force)
```

**Look at `n_probe=2`: 85% recall while touching under 7% of the data.** That's a ~15× speedup for a 15% accuracy loss. Whether that trade is acceptable depends on your task — but it's the trade ANN indexes exist to offer.

And note it saturates. Going from `n_probe=4` to `32` costs 8× the work for *zero* additional recall. **Tuning past the point of diminishing returns is pure waste** — which you can only know by measuring.

### The caveat that matters

Run the same experiment on **random, structureless vectors** and the picture collapses. Both rows below are the same algorithm with the same settings, on the same number of vectors:

```
                                       n_probe=1  =2     =4     =8
  clustered (like real embeddings)          0.60  0.85   1.00   1.00
  random (no structure at all)              0.24  0.36   0.55   0.74
```

> **🔑 ANN indexes work because real embeddings have structure.** Documents cluster by topic, so a query's true neighbours tend to live in the same cluster. On genuinely structureless data, clustering tells you nothing and IVF degrades toward brute force.
>
> The practical implication: **recall figures don't transfer between datasets.** A blog post reporting 95% recall at `n_probe=8` measured that on *their* data. You have to measure on yours.

### How to actually tune an index

1. Build a **Flat** index as ground truth
2. Take 50–100 representative queries
3. For each candidate setting, compute recall against Flat and measure latency
4. Plot recall against latency
5. Pick the point where recall is good enough for your task, and stop

**"Good enough" is a product decision, not a technical one.** A RAG system retrieving 20 chunks to feed an LLM tolerates 85% recall easily — the LLM sees plenty of context either way. A legal-discovery system that must not miss a document needs Flat, or an exhaustive re-check.

> **⚠️ Recall degradation is invisible in production.** Nothing errors. Latency looks great. Your users get slightly worse answers forever. This is why you build the Flat baseline *before* you need it — retrofitting one after a quality complaint means you have no idea whether the index was ever good.

---

## 7.6 Metadata Filtering

The feature tutorials skip and production systems can't live without.

### Why you need it

Real queries are rarely just "find similar text":

- "Find similar support tickets **from this customer**"
- "Search the docs, **only the current version**"
- "Similar case law, **decided after 2020**, **in this jurisdiction**"
- "Search documents **this user has permission to read**" ← the important one

That last case is access control. **Getting it wrong is a data breach**, not a quality problem.

### Storing metadata

Every vector carries a payload:

```python
documents = [
    {
        "text": "Metformin dosing guidance for type 2 diabetes...",
        "vector": [0.12, -0.44, ...],
        "metadata": {
            "source": "clinical_guidelines_2024.pdf",
            "page": 47,
            "department": "endocrinology",
            "year": 2024,
            "access_level": "clinician",
        },
    },
]
```

Metadata earns its place three ways: **filtering** (narrow the search), **citation** (Module 8 needs `source` and `page` to attribute answers), and **access control**.

### Pre-filter vs post-filter — and why it matters

Two ways to combine filtering with vector search, and they fail differently.

**Post-filtering** — search first, then discard:

```
   ANN search for top-100  ──▶  drop anything failing the filter  ──▶  keep top-10
```

| ✅ | ❌ |
|---|---|
| Simple; works with any index | **You may end up with fewer than k results** — or none |

The failure is concrete: if only 0.1% of your corpus matches `department = 'legal'`, your top-100 semantic matches may contain *zero* legal documents. You return nothing, despite excellent matches existing.

**Pre-filtering** — restrict first, then search:

```
   filter to matching subset  ──▶  search only within it  ──▶  top-10
```

| ✅ | ❌ |
|---|---|
| **Always returns k results** if k exist | Can defeat the ANN index entirely |

The problem: an HNSW graph is built over *all* vectors. Removing most of them can disconnect the graph, so greedy traversal gets stranded and recall collapses. Modern databases handle this with filtered-search algorithms that walk the graph while respecting the filter — but the quality depends on the implementation, and it's worth knowing this is where vector databases genuinely differ.

### The practical rule

| Filter selectivity | Strategy |
|---|---|
| **Broad** (matches > ~20%) | Post-filter, over-fetching (`k * 5`) |
| **Narrow** (matches < ~1%) | Pre-filter, or brute-force the subset |
| **Very narrow** (a handful) | Skip vectors entirely — just fetch and rank the subset |

That last row is worth stating plainly: **if a filter narrows to 50 documents, brute-force those 50.** Reaching for an ANN index there is more complexity for slower results.

> **⚠️ Never rely on post-filtering for access control.** Retrieving documents the user can't see and filtering them afterwards means those documents *were* fetched into your process — one logging mistake or error message away from leaking. Filter for permissions at the query boundary, before the search.

---

## 7.7 FAISS in Practice

**FAISS** (Facebook AI Similarity Search) is a library, not a server. It runs in your process — no network, no daemon, extremely fast.

```powershell
pip install faiss-cpu
```

### A complete example

```python
"""FAISS: index and search, in one file."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "The stock market crashed today.",
    "Investors lost money in the financial crisis.",
    "LLMs can generate human-like text.",
    "Artificial intelligence writes like a human.",
]

# --- 1. Embed. Normalise so inner product == cosine similarity ---
embeddings = model.encode(documents, normalize_embeddings=True)
embeddings = np.asarray(embeddings, dtype="float32")   # FAISS requires float32
dimension = embeddings.shape[1]

# --- 2. Build a Flat index (exact) -----------------------------
# IndexFlatIP = exact search by Inner Product. With normalised vectors
# that IS cosine similarity (Module 3, section 3.6).
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Indexed {index.ntotal} vectors of dimension {dimension}")

# --- 3. Search -------------------------------------------------
query = model.encode("cat on a mat", normalize_embeddings=True)
query = np.asarray([query], dtype="float32")           # FAISS expects a 2-D batch

scores, indices = index.search(query, k=3)

for score, idx in zip(scores[0], indices[0]):
    print(f"  {score:.3f}  {documents[idx]}")
```

Three things to get right, because each causes a confusing failure:

| Requirement | Symptom if you get it wrong |
|---|---|
| **`float32`** | `TypeError`, or silently wrong results from a float64 array |
| **A 2-D query array** | Shape assertion error — FAISS always works in batches |
| **Normalise for cosine** | `IndexFlatIP` on unnormalised vectors ranks by magnitude, not meaning |

### Swapping in an ANN index

The interface stays identical — which is what makes measuring the trade-off easy:

```python
# --- IVF: cluster-based (Module 7, section 7.4) ---
n_clusters = 100
quantizer = faiss.IndexFlatIP(dimension)              # used to find centroids
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)

index.train(embeddings)      # IVF MUST be trained: this runs k-means
index.add(embeddings)

index.nprobe = 10            # THE tuning knob (section 7.5)
scores, indices = index.search(query, k=3)


# --- HNSW: graph-based, the usual production choice ---
index = faiss.IndexHNSWFlat(dimension, 32)            # 32 = M, links per node
index.hnsw.efConstruction = 200                       # build quality
index.add(embeddings)                                  # no training needed
index.hnsw.efSearch = 50                              # search effort
scores, indices = index.search(query, k=3)
```

> **⚠️ IVF requires `.train()` before `.add()`.** Skip it and FAISS raises an error about an untrained index. Flat and HNSW don't need training — Flat has no structure to learn, and HNSW builds its graph incrementally as you add.

### Saving and loading

```python
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")
```

> **⚠️ FAISS stores vectors, not your text.** The index gives you back *integer positions*. You must persist the documents and metadata yourself — a JSON file, a pickle, a database — keyed by the same positions. **Getting the two out of sync is the most common FAISS bug**, and it produces confidently mismatched results: right vector, wrong text.
>
> Chroma exists partly to solve this for you.

---

## 7.8 Chroma in Practice

**Chroma** is a lightweight vector database that stores documents, embeddings and metadata together, and can persist to disk.

```powershell
pip install chromadb
```

```python
"""Chroma: documents, embeddings and metadata in one place."""

import chromadb

# In-memory (gone on exit):     chromadb.Client()
# Persistent (survives restart):
client = chromadb.PersistentClient(path="./chroma_db")

# A collection is like a table. cosine must be set at creation time.
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"},
)

# --- Add documents. Chroma embeds them for you by default. ---
collection.add(
    ids=["doc1", "doc2", "doc3"],                 # your own stable IDs
    documents=[
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "The stock market crashed today.",
    ],
    metadatas=[
        {"topic": "animals", "year": 2023},
        {"topic": "animals", "year": 2024},
        {"topic": "finance", "year": 2024},
    ],
)

# --- Query ---
results = collection.query(query_texts=["cat on a mat"], n_results=2)

for doc, distance, meta in zip(results["documents"][0],
                              results["distances"][0],
                              results["metadatas"][0]):
    print(f"  distance {distance:.3f}  [{meta['topic']}]  {doc}")
```

### Metadata filtering

The reason to reach for Chroma over FAISS:

```python
# --- Filter on metadata ---
results = collection.query(
    query_texts=["cat on a mat"],
    n_results=2,
    where={"topic": "animals"},                    # metadata filter
)

# --- Operators and combinations ---
results = collection.query(
    query_texts=["market news"],
    n_results=5,
    where={
        "$and": [
            {"year": {"$gte": 2024}},
            {"topic": {"$in": ["finance", "economics"]}},
        ]
    },
)

# --- Filter on document CONTENT as well as metadata ---
results = collection.query(
    query_texts=["dosing"],
    n_results=5,
    where_document={"$contains": "metformin"},      # a crude keyword gate
)
```

That last one is a partial answer to Lab 3's `SKU-4471` problem — you can require an exact substring alongside semantic similarity. It's not full hybrid search (Module 8 builds that properly), but it's often enough.

### Bringing your own embeddings

Chroma's default embedding function is convenient and not always what you want:

```python
# Explicit is better: you control the model, and you know what it is.
embeddings = model.encode(documents, normalize_embeddings=True).tolist()

collection.add(
    ids=["doc1", "doc2"],
    documents=documents,           # stored for retrieval, not re-embedded
    embeddings=embeddings,         # your vectors, used for search
    metadatas=metadatas,
)
```

> **🔑 Always know which embedding model produced your vectors.** Module 3 §3.7: query and documents must be embedded by the *same* model or the scores are meaningless. If you let a library pick the default, and the default changes in a later version, your index silently becomes incompatible with new queries. Pin the model and record it.

### FAISS or Chroma?

| | **FAISS** | **Chroma** |
|---|---|---|
| What it is | A library, in-process | A database, in-process or client/server |
| Stores your text? | **No** — vectors only | **Yes** — text, vectors, metadata |
| Metadata filtering | Do it yourself | **Built in** |
| Persistence | Manual (`write_index` + your own store) | Built in |
| Raw speed | **Faster** | Fast enough |
| Index tuning | **Extensive** | Limited |
| Best for | Maximum control and speed | Getting something working quickly |

**Start with Chroma.** Move to FAISS when you need index tuning or the last increment of speed, and accept that you'll manage the document store yourself.

---

## 7.9 Choosing a Vector Database

Three families.

### 1. Dedicated vector databases

**Pinecone · Weaviate · Milvus · Qdrant · Chroma**

Built ground-up for vectors. Best performance, filtering and scale. Several are fully managed.

| ✅ | ❌ |
|---|---|
| Best-in-class ANN, filtering, scaling | Another system to run, monitor and pay for |
| Managed options remove ops burden | Data lives outside your primary database |

### 2. Vector search in a database you already run

**pgvector (Postgres) · Redis · Elasticsearch/OpenSearch · MongoDB Atlas**

| ✅ | ❌ |
|---|---|
| **One system.** Transactions across relational and vector data. | Usually slower at very large scale |
| No migration, no new ops, existing backups | Fewer index-tuning options |

> **💡 `pgvector` deserves more attention than it usually gets.** If you already run Postgres, it is very often the right answer up to several million vectors. You get `JOIN`s between vectors and your business tables, real transactions, one backup strategy, and no new infrastructure. Reaching straight for a dedicated vector database is a common case of premature specialisation.

### 3. Libraries

**FAISS · Annoy · ScaNN · hnswlib**

In-process, no server. Ideal for prototyping, research and read-heavy workloads with infrequent updates.

| ✅ | ❌ |
|---|---|
| Fastest — no network hop | No persistence, filtering or concurrency for free |
| Trivial to deploy — it's just a dependency | You build everything around it |

### A decision path

```
  Fewer than ~100k vectors, single process?
      -> NumPy array. Seriously. (section 7.1)

  Already running Postgres?
      -> pgvector. One system beats two.

  Prototyping, or need heavy index tuning?
      -> FAISS or Chroma locally.

  Need managed, scalable, multi-tenant, with filtering?
      -> A dedicated vector database.

  Billions of vectors?
      -> Milvus / Vespa, plus quantization, plus a real infrastructure budget.
```

**Weigh:** scale and latency, metadata filtering quality, hybrid search support, managed vs self-hosted, and cost.

---

## 7.10 Production Concerns

Five things that bite after the demo works.

### 1. Dimension and model lock-in

Module 3 said it; here's the operational cost.

**An index is tied to the embedding model that built it.** Change the model and every stored vector becomes meaningless — there's no conversion.

Re-indexing a large corpus means re-embedding everything: hours of compute, and a migration where old and new must coexist. The mitigations: store the model name and version in your metadata, and design for a rebuild-and-swap rather than an in-place upgrade.

**And it constrains model choice.** A better embedding model appears; adopting it costs a full re-index. That's a real reason to think about your embedding model more carefully than the rest of your stack.

### 2. Updates and deletions

| Index | Add | Delete |
|---|---|---|
| **Flat** | Trivial | Trivial |
| **HNSW** | Fine incrementally | **Awkward** — usually a tombstone, not a real removal |
| **IVF** | Fine | Fine, but clusters drift as data changes |

Two consequences. **Deletion is often soft**, so "deleted" vectors still occupy memory until a compaction — which matters for data-deletion requests under GDPR-style rules. And **IVF centroids go stale**: they were computed from the original distribution, so after enough churn recall degrades and you need to retrain.

**Schedule periodic rebuilds.** Treat an ANN index as a cache of a structure, not a permanent artifact.

### 3. Cost

| Cost | Driver |
|---|---|
| **Storage** | `N × D × 4` bytes, plus graph overhead for HNSW (often 1.5–2×) |
| **Memory** | Most ANN indexes want to be resident in RAM |
| **Embedding** | One-off for the corpus, ongoing for new documents and every query |
| **Queries** | Managed services usually bill per query |

The lever people miss: **embedding dimension**. Halving `D` halves storage and roughly halves search time. A 384-dimension model often retrieves nearly as well as a 1536-dimension one at a quarter of the cost — and Module 3's lab gave you the method for checking whether that's true on your data.

### 4. Measure recall continuously

Not once, at launch. Recall drifts as your data distribution changes.

Keep a small set of queries with known-correct answers and a Flat baseline, and run it on a schedule. **It is the only way to detect silent retrieval degradation** — nothing else in your monitoring will show it.

### 5. Where hybrid search comes in

Pure semantic search fails on exact identifiers — `SKU-4471`, error codes, product names. You saw this in Lab 3.

The fix is combining keyword and semantic scoring, and **Module 8 builds it**. Worth knowing now that this is a solved problem and a known limitation, not something you did wrong.

---

## 🧪 Hands-On Lab 7

**→ [Go to Lab 7: Build an ANN Index and Measure It](../labs/07-vector-databases/README.md)**

Implement brute-force search and a real IVF index from scratch, then measure the recall–latency curve yourself. Then do the same with FAISS and Chroma, with metadata filtering.

Part 1 is pure NumPy: no packages beyond NumPy, no API key, no cost. Budget 50 minutes.

---

## ✅ Key Takeaways

1. **A vector database finds *probably* the nearest neighbours, quickly.** The approximation is the product, not a defect.

2. **You probably don't need one below ~100k vectors.** A NumPy array is simpler and often faster.

3. **Indexing happens once; querying happens constantly.** That asymmetry justifies expensive index construction.

4. **Four index families:** Flat (exact, the baseline), IVF (cluster and probe), HNSW (graph, the default), quantization (compress when memory-bound).

5. **Always build a Flat baseline.** It's the only way to know what your approximate index is missing.

6. **Recall@k is the metric.** Measure it against Flat on *your* data — published figures don't transfer.

7. **ANN works because real embeddings have structure.** On structureless data, IVF degrades toward brute force.

8. **Tuning saturates.** Past a point, more effort buys zero recall. Measuring tells you where to stop.

9. **Recall degradation is silent.** No errors, good latency, quietly worse answers. Monitor it deliberately.

10. **Metadata filtering is a first-class requirement**, and pre- vs post-filtering fail differently. Never post-filter for access control.

11. **FAISS stores vectors, not text.** Keeping your document store in sync is your job, and getting it wrong yields right-vector-wrong-text.

12. **An index is locked to its embedding model.** Changing models means re-indexing everything. Record which model built the index.

---

## ⚠️ Common Mistakes & Misconceptions

<br>

> ### ❌ "I need a vector database for my 500 documents"
> **Reality:** 500 vectors is a NumPy array and a dot product. It'll be faster than a database, simpler to debug, and impossible to misconfigure. Add infrastructure when you have the problem it solves.

<br>

> ### ❌ Assuming ANN search returns the true nearest neighbours
> **Reality:** it returns *approximate* neighbours. Default settings might give 80% recall or 99% — you cannot know without measuring against an exact baseline.

<br>

> ### ❌ Never measuring recall
> **Reality:** the most consequential omission in this module. A mistuned index returns plausible-but-worse results silently, forever, and your RAG system confidently answers from second-rate context. Build the Flat baseline early.

<br>

> ### ❌ Copying `n_probe` or `ef_search` from a blog post
> **Reality:** those numbers were measured on someone else's data with someone else's embedding model. Recall–latency curves are dataset-specific. Measure yours.

<br>

> ### ❌ Forgetting `.train()` on an IVF index
> **Reality:** IVF must learn its centroids by running k-means before you add vectors. FAISS raises an error. Flat and HNSW need no training, which is why the mistake surprises people.

<br>

> ### ❌ Passing float64 arrays to FAISS
> **Reality:** FAISS wants `float32`. NumPy defaults to `float64`. Cast explicitly with `.astype("float32")`.

<br>

> ### ❌ Using `IndexFlatIP` without normalising
> **Reality:** inner product only equals cosine similarity on unit vectors. Skip normalisation and you rank by magnitude — meaning longer documents win regardless of relevance.

<br>

> ### ❌ Letting the FAISS index and document store drift apart
> **Reality:** FAISS returns integer positions. If you rebuild the index but not the position→document mapping, every result is the *right vector attached to the wrong text* — and it looks entirely plausible. Store them together, or use Chroma.

<br>

> ### ❌ Post-filtering for access control
> **Reality:** a security bug, not a performance one. Documents the user may not see are fetched into your process before being discarded — one logging line or stack trace from leaking. Filter at the query boundary.

<br>

> ### ❌ Expecting post-filtering to return k results
> **Reality:** if the filter is narrow, your top-100 semantic matches may contain zero matching documents, and you return nothing while good answers exist. Over-fetch for broad filters; pre-filter for narrow ones.

<br>

> ### ❌ Changing the embedding model without re-indexing
> **Reality:** old and new vectors live in unrelated coordinate spaces. Nothing errors; results become nonsense. Re-index everything, and record the model in your metadata so you can tell.

<br>

> ### ❌ "HNSW is strictly better, so always use it"
> **Reality:** it's usually the right default, and it's the most memory-hungry option — it stores a graph on top of your vectors. When memory is the binding constraint, IVF or quantization wins.

<br>

> ### ❌ Treating a bigger embedding dimension as free quality
> **Reality:** dimension drives storage, memory and query latency roughly linearly. Test whether a 384-dimension model is good enough for your task before paying 4× for 1536.

---

## 📚 Going Deeper

**Understand the algorithms**
- [*Efficient and robust approximate nearest neighbor search using HNSW*](https://arxiv.org/abs/1603.09320) — the HNSW paper; readable
- [Pinecone: Faiss tutorial series](https://www.pinecone.io/learn/series/faiss/) — the clearest practical explanation of IVF, PQ and HNSW
- [ANN Benchmarks](https://ann-benchmarks.com/) — recall-vs-speed curves across every library. Note how much they vary **by dataset**.

**Documentation**
- [FAISS wiki: guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Chroma docs](https://docs.trychroma.com/)
- [pgvector](https://github.com/pgvector/pgvector) — read this before adding a new database

**Reference**
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — embedding model quality, with dimensions listed so you can weigh the cost

---

<div align="center">

**[⬅ Module 6](06-langchain-chains.md)** · **[🧪 Do Lab 7](../labs/07-vector-databases/README.md)** · **[🏠 README](../README.md)** · **➡️ Module 8: Retrieval-Augmented Generation** *(coming next)*

</div>
