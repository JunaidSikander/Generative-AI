"""
starter.py - Lab 7: Build an ANN Index and Measure It

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-6) needs only numpy. You will implement brute-force search,
a real IVF index, recall measurement, and metadata filtering - then MEASURE
the recall/speed trade-off yourself.

PART 2 (in the lab brief) does the same with FAISS and Chroma.
"""

import numpy as np


# ======================================================================
# Helpers - provided, no need to change these
# ======================================================================

def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to unit length so dot product == cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def kmeans(vectors: np.ndarray, n_clusters: int, seed: int = 0,
           iterations: int = 25) -> np.ndarray:
    """Plain k-means. Returns the cluster centroids.

    Provided so you can focus on the INDEX rather than the clustering.
    Deterministic given a seed, so the tests are reproducible.
    """
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(len(vectors), n_clusters, replace=False)].copy()

    for _ in range(iterations):
        # Assign every vector to its nearest centroid. For unit-length vectors,
        # maximising cosine similarity is equivalent to minimising distance -
        # so one matrix product does the whole assignment step.
        #
        # The naive form, ((vectors[:, None, :] - centroids[None, :, :])**2).sum(-1),
        # allocates an (n_vectors x n_clusters x dimensions) array. At 20,000
        # vectors and 140 clusters that is ~2.9 GB PER ITERATION. The matmul
        # below allocates only (n_vectors x n_clusters).
        labels = (vectors @ centroids.T).argmax(axis=1)

        # Move each centroid to the mean of its members.
        for cluster in range(n_clusters):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(0)

        # Keep centroids on the unit sphere so the cosine shortcut stays valid.
        centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)

    return centroids


def make_clustered_corpus(n_vectors: int, dimensions: int, n_topics: int,
                          spread: float = 0.15, seed: int = 7) -> np.ndarray:
    """Generate synthetic vectors that CLUSTER, like real embeddings do.

    Real document embeddings group by topic. This matters enormously: ANN
    indexes exploit exactly that structure. On structureless random vectors
    they degrade toward brute force, which you will see for yourself in
    the experiment at the end.
    """
    rng = np.random.default_rng(seed)
    topic_centres = normalise(rng.normal(size=(n_topics, dimensions)))
    assignments = rng.integers(0, n_topics, size=n_vectors)
    points = topic_centres[assignments] + rng.normal(
        scale=spread, size=(n_vectors, dimensions))
    return normalise(points)


# ======================================================================
# TASK 1 - brute_force_search
# Module 7, section 7.4 (Flat index)
# ======================================================================

def brute_force_search(query: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Exact nearest-neighbour search. The ground truth.

    Compares the query against EVERY vector. Slow, and exactly right -
    which is why you need it: it is the only way to measure what an
    approximate index is missing.

    Args:
        query:   A single normalised vector, shape (dimensions,).
        vectors: Normalised corpus, shape (n_vectors, dimensions).
        k:       How many results to return.

    Returns:
        Indices of the k most similar vectors, best first.

    Examples:
        >>> V = normalise(np.array([[1., 0.], [0., 1.], [0.9, 0.1]]))
        >>> brute_force_search(np.array([1., 0.]), V, 2).tolist()
        [0, 2]
    """
    # TODO:
    #   1. Score every vector: vectors @ query
    #      (Both are normalised, so these ARE cosine similarities.)
    #   2. Return the indices of the k highest scores, best first.
    #      Reuse the argsort pattern from Lab 3, task 5.
    return np.array([], dtype=int)


# ======================================================================
# TASK 2 - build_ivf_index
# Module 7, section 7.4 (IVF)
# ======================================================================

def build_ivf_index(vectors: np.ndarray, centroids: np.ndarray) -> dict:
    """Assign every vector to its nearest centroid, building inverted lists.

    This is the "inverted file" in Inverted File Index: for each cluster,
    the list of vector indices belonging to it.

    Args:
        vectors:   Normalised corpus, shape (n_vectors, dimensions).
        centroids: Normalised centroids, shape (n_clusters, dimensions).

    Returns:
        {cluster_id: [vector indices]} for EVERY cluster id, including
        any that ended up empty (so callers can look up any id safely).

    Examples:
        >>> V = normalise(np.array([[1., 0.], [0.9, 0.1], [0., 1.]]))
        >>> C = normalise(np.array([[1., 0.], [0., 1.]]))
        >>> build_ivf_index(V, C)
        {0: [0, 1], 1: [2]}
    """
    # TODO:
    #   1. Score every vector against every centroid: vectors @ centroids.T
    #      -> shape (n_vectors, n_clusters)
    #   2. For each vector, find its best centroid with .argmax(axis=1)
    #   3. Build the dict. Start with an EMPTY LIST FOR EVERY cluster id,
    #      then append each vector's index to its cluster's list.
    #
    # Why pre-create empty lists? A cluster can legitimately end up with no
    # members, and ivf_search must be able to look up any id without a KeyError.
    return {}


# ======================================================================
# TASK 3 - ivf_search
# Module 7, sections 7.4 and 7.5
# ======================================================================

def ivf_search(query: np.ndarray, vectors: np.ndarray, centroids: np.ndarray,
               inverted_lists: dict, k: int, n_probe: int = 1) -> np.ndarray:
    """Approximate search: only scan the n_probe nearest clusters.

    Args:
        query:          A single normalised vector.
        vectors:        The full normalised corpus.
        centroids:      Normalised centroids.
        inverted_lists: Output of build_ivf_index.
        k:              How many results to return.
        n_probe:        How many clusters to search. THE tuning knob:
                        higher = better recall, slower.

    Returns:
        Indices of the k best candidates found, best first. May contain
        fewer than k if the probed clusters hold fewer than k vectors.
    """
    # TODO:
    #   1. Score the query against the CENTROIDS: centroids @ query
    #   2. Take the n_probe best centroid ids.
    #   3. Gather all vector indices from those clusters' lists.
    #   4. If no candidates, return an empty int array.
    #   5. Score ONLY the candidates: vectors[candidates] @ query
    #   6. Return the k best - remembering to map back to ORIGINAL indices.
    #
    # Step 6 is where this task usually goes wrong. argsort over the
    # candidate scores gives positions WITHIN the candidate list, not
    # positions in the corpus. You must index back through `candidates`.
    return np.array([], dtype=int)


# ======================================================================
# TASK 4 - recall_at_k
# Module 7, section 7.5
# ======================================================================

def recall_at_k(approximate: np.ndarray, exact: np.ndarray) -> float:
    """What fraction of the TRUE top-k did the approximate search find?

    The single most important measurement in this module. Without it you
    cannot tell a well-tuned index from a badly tuned one - both return
    results, and both look fine.

    Args:
        approximate: Indices from an approximate search.
        exact:       Indices from brute_force_search (the ground truth).

    Returns:
        A value from 0.0 to 1.0. Returns 1.0 when exact is empty
        (vacuously true - nothing to miss).

    Examples:
        >>> recall_at_k(np.array([1, 2, 3]), np.array([1, 2, 3]))
        1.0
        >>> recall_at_k(np.array([1, 2, 9]), np.array([1, 2, 3]))
        0.6666666666666666
        >>> recall_at_k(np.array([]), np.array([1, 2]))
        0.0
    """
    # TODO:
    #   1. Handle the empty-ground-truth case first: return 1.0.
    #   2. Otherwise return |intersection| / |exact|.
    # Hint: build Python sets of ints from the arrays. Order does not
    # matter - recall asks WHICH results were found, not in what order.
    return 0.0


# ======================================================================
# TASK 5 - filter_by_metadata
# Module 7, section 7.6
# ======================================================================

def filter_by_metadata(metadatas: list, filters: dict) -> np.ndarray:
    """Return the indices of records matching ALL the given filters.

    Supports exact matches and two range operators, which is enough to
    demonstrate the pre-filter vs post-filter distinction.

    Args:
        metadatas: One dict per vector, e.g. {"topic": "finance", "year": 2024}
        filters:   Conditions to apply. Values may be:
                     - a plain value          -> exact match
                     - {"$gte": n}            -> greater than or equal
                     - {"$lte": n}            -> less than or equal
                     - {"$in": [a, b]}        -> value is in the list

    Returns:
        Array of matching indices, ascending. Empty filters match everything.

    Examples:
        >>> metas = [{"topic": "a", "year": 2023}, {"topic": "b", "year": 2024}]
        >>> filter_by_metadata(metas, {"topic": "a"}).tolist()
        [0]
        >>> filter_by_metadata(metas, {"year": {"$gte": 2024}}).tolist()
        [1]
        >>> filter_by_metadata(metas, {}).tolist()
        [0, 1]
    """
    # TODO:
    #   Loop over metadatas with enumerate(). For each record, check EVERY
    #   filter key. Keep the index only if all conditions pass.
    #
    #   For each (key, condition):
    #     - if key not in the record        -> no match
    #     - if condition is a dict          -> apply $gte / $lte / $in
    #     - otherwise                       -> exact equality
    #
    #   Return np.array(matches, dtype=int).
    return np.array([], dtype=int)


# ======================================================================
# TASK 6 - search_with_filter
# Module 7, section 7.6
# ======================================================================

def search_with_filter(query: np.ndarray, vectors: np.ndarray, metadatas: list,
                       filters: dict, k: int) -> np.ndarray:
    """PRE-filtered exact search: narrow the corpus first, then search it.

    Pre-filtering always returns k results if k matching records exist -
    unlike post-filtering, which can return nothing when the filter is
    narrow (Module 7, section 7.6).

    Args:
        query:     A single normalised vector.
        vectors:   The full normalised corpus.
        metadatas: One metadata dict per vector.
        filters:   Conditions, as for filter_by_metadata.
        k:         How many results to return.

    Returns:
        Indices (into the FULL corpus) of the k best matching records.
    """
    # TODO:
    #   1. Use filter_by_metadata to get the allowed indices.
    #   2. If none, return an empty int array.
    #   3. Score only vectors[allowed] against the query.
    #   4. Take the k best and map back to original indices via `allowed`.
    #
    # Same index-mapping trap as task 3: argsort gives positions within
    # the filtered subset, not the full corpus.
    return np.array([], dtype=int)


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _run_self_test() -> int:
    checks = []

    def check(name, got, expected):
        checks.append((name, got, expected))

    # --- fixtures ---
    small = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]))
    centroids_small = normalise(np.array([[1.0, 0.0], [0.0, 1.0]]))

    # --- TASK 1 ---
    check("1. brute_force_search top-2",
          brute_force_search(np.array([1.0, 0.0]), small, 2).tolist(), [0, 1])
    check("1. brute_force_search top-1 on the other side",
          brute_force_search(np.array([0.0, 1.0]), small, 1).tolist(), [2])
    check("1. brute_force_search returns all when k > n",
          len(brute_force_search(np.array([1.0, 0.0]), small, 99)), 4)

    # --- TASK 2 ---
    check("2. build_ivf_index partitions correctly",
          build_ivf_index(small, centroids_small), {0: [0, 1], 1: [2, 3]})
    check("2. build_ivf_index covers every vector exactly once",
          sorted(i for lst in build_ivf_index(small, centroids_small).values()
                 for i in lst),
          [0, 1, 2, 3])

    # An empty cluster must still appear as a key.
    far_centroids = normalise(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    built = build_ivf_index(small, far_centroids)
    check("2. build_ivf_index includes empty clusters as keys",
          sorted(built.keys()) if isinstance(built, dict) else built, [0, 1, 2])

    # --- TASK 3 ---
    lists_small = build_ivf_index(small, centroids_small)
    check("3. ivf_search with n_probe=1 finds the right cluster",
          ivf_search(np.array([1.0, 0.0]), small, centroids_small,
                     lists_small, k=2, n_probe=1).tolist(), [0, 1])
    check("3. ivf_search with n_probe=all == brute force",
          ivf_search(np.array([1.0, 0.0]), small, centroids_small,
                     lists_small, k=4, n_probe=2).tolist(),
          brute_force_search(np.array([1.0, 0.0]), small, 4).tolist())
    check("3. ivf_search returns ORIGINAL indices, not candidate positions",
          ivf_search(np.array([0.0, 1.0]), small, centroids_small,
                     lists_small, k=1, n_probe=1).tolist(), [2])

    # --- TASK 4 ---
    check("4. recall_at_k perfect", recall_at_k(np.array([1, 2, 3]), np.array([1, 2, 3])), 1.0)
    check("4. recall_at_k partial",
          round(recall_at_k(np.array([1, 2, 9]), np.array([1, 2, 3])), 4), 0.6667)
    check("4. recall_at_k zero", recall_at_k(np.array([7, 8]), np.array([1, 2])), 0.0)
    check("4. recall_at_k ignores order",
          recall_at_k(np.array([3, 2, 1]), np.array([1, 2, 3])), 1.0)
    check("4. recall_at_k empty ground truth is 1.0",
          recall_at_k(np.array([]), np.array([])), 1.0)
    check("4. recall_at_k empty result is 0.0",
          recall_at_k(np.array([]), np.array([1, 2])), 0.0)

    # --- TASK 5 ---
    metas = [
        {"topic": "animals", "year": 2023, "level": "public"},
        {"topic": "animals", "year": 2024, "level": "internal"},
        {"topic": "finance", "year": 2024, "level": "public"},
        {"topic": "finance", "year": 2022, "level": "public"},
    ]
    check("5. filter exact match", filter_by_metadata(metas, {"topic": "animals"}).tolist(), [0, 1])
    check("5. filter $gte", filter_by_metadata(metas, {"year": {"$gte": 2024}}).tolist(), [1, 2])
    check("5. filter $lte", filter_by_metadata(metas, {"year": {"$lte": 2022}}).tolist(), [3])
    check("5. filter $in",
          filter_by_metadata(metas, {"topic": {"$in": ["finance"]}}).tolist(), [2, 3])
    check("5. filter combines with AND",
          filter_by_metadata(metas, {"topic": "finance", "year": {"$gte": 2024}}).tolist(), [2])
    check("5. empty filter matches everything",
          filter_by_metadata(metas, {}).tolist(), [0, 1, 2, 3])
    check("5. no matches -> empty",
          filter_by_metadata(metas, {"topic": "sports"}).tolist(), [])
    check("5. missing key -> no match",
          filter_by_metadata(metas, {"nonexistent": 1}).tolist(), [])

    # --- TASK 6 ---
    # Records 2 and 3 are the 'finance' ones. Against query [1, 0], record 3
    # ([0.1, 0.9]) scores higher than record 2 ([0, 1]) - so the result must be
    # [3, 2]: ranked by SIMILARITY, not by index order.
    check("6. search_with_filter restricts, and ranks by similarity",
          search_with_filter(np.array([1.0, 0.0]), small, metas,
                             {"topic": "finance"}, k=2).tolist(),
          [3, 2])
    check("6. search_with_filter returns k even when the filter is narrow",
          len(search_with_filter(np.array([1.0, 0.0]), small, metas,
                                 {"year": {"$lte": 2022}}, k=1)), 1)
    check("6. search_with_filter with no matches -> empty",
          search_with_filter(np.array([1.0, 0.0]), small, metas,
                             {"topic": "sports"}, k=2).tolist(), [])

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 7 SELF-TEST - ANN search from scratch")
    print("=" * 76)
    print()

    failures = 0
    for name, got, expected in checks:
        if got == expected:
            print(f"[ OK ]  {name}")
        else:
            failures += 1
            print(f"[FAIL]  {name}")
            print(f"          got:      {got!r}")
            print(f"          expected: {expected!r}")

    print()
    print("-" * 76)
    if failures == 0:
        print(f"  All {len(checks)} checks passed. Now run the experiment below -")
        print("  measuring the trade-off is the real point of this lab.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: task 1, then 2, then 3 (which needs both), then 4-6.")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# THE EXPERIMENT - measure the recall/speed trade-off
# ======================================================================

def experiment_recall_curve():
    """Measure recall against work done, across n_probe settings."""
    print("=" * 76)
    print("  EXPERIMENT 1: the recall/latency curve")
    print("=" * 76)
    print()

    n_vectors, dimensions, n_clusters, k = 2000, 64, 32, 10

    vectors = make_clustered_corpus(n_vectors, dimensions, n_topics=12, spread=0.15)
    centroids = normalise(kmeans(vectors, n_clusters, seed=1))
    lists = build_ivf_index(vectors, centroids)

    if not lists:
        print("  build_ivf_index not implemented yet.")
        print()
        return

    sizes = [len(v) for v in lists.values()]
    print(f"  corpus: {n_vectors} vectors, {dimensions} dims, {n_clusters} clusters")
    print(f"  cluster sizes: min {min(sizes)}, max {max(sizes)}")
    print()

    rng = np.random.default_rng(11)
    # Queries drawn NEAR real corpus points - like a real user query, which
    # resembles the documents it is looking for.
    queries = [normalise(vectors[rng.integers(0, n_vectors)]
                         + rng.normal(scale=0.08, size=dimensions))
               for _ in range(25)]

    print(f"  {'n_probe':>9}{'recall@10':>12}{'% scanned':>12}{'speedup':>10}")
    print("  " + "-" * 43)

    for n_probe in [1, 2, 4, 8, 16, 32]:
        recalls, scanned = [], []
        for query in queries:
            exact = brute_force_search(query, vectors, k)
            approx = ivf_search(query, vectors, centroids, lists, k, n_probe)
            recalls.append(recall_at_k(approx, exact))

            probed = np.argsort(centroids @ query)[::-1][:n_probe]
            scanned.append(sum(len(lists[int(c)]) for c in probed))

        mean_recall = float(np.mean(recalls))
        percent = 100 * float(np.mean(scanned)) / n_vectors
        print(f"  {n_probe:>9}{mean_recall:>12.2f}{percent:>11.1f}%"
              f"{n_vectors / max(np.mean(scanned), 1):>9.1f}x")

    print()
    print("  Read the n_probe=2 row. That is the trade this whole module is about:")
    print("  a large speedup for a small, MEASURED accuracy loss.")
    print()
    print("  Then note where the curve saturates. Past that point you are paying")
    print("  more work for zero extra recall - and the only way to know where")
    print("  that point is, is to measure it.")
    print()


def experiment_structure_matters():
    """Show that ANN depends on the data having cluster structure."""
    print("=" * 76)
    print("  EXPERIMENT 2: ANN only works if the data has structure")
    print("=" * 76)
    print()

    n_vectors, dimensions, n_clusters, k = 2000, 64, 32, 10
    rng = np.random.default_rng(5)

    datasets = {
        "clustered (like real embeddings)":
            make_clustered_corpus(n_vectors, dimensions, n_topics=12, spread=0.15),
        "random (no structure at all)":
            normalise(rng.normal(size=(n_vectors, dimensions))),
    }

    for label, vectors in datasets.items():
        centroids = normalise(kmeans(vectors, n_clusters, seed=1))
        lists = build_ivf_index(vectors, centroids)
        if not lists:
            print("  build_ivf_index not implemented yet.")
            return

        query_rng = np.random.default_rng(11)
        queries = [normalise(vectors[query_rng.integers(0, n_vectors)]
                             + query_rng.normal(scale=0.08, size=dimensions))
                   for _ in range(25)]

        row = []
        for n_probe in [1, 2, 4, 8]:
            recalls = [recall_at_k(
                ivf_search(q, vectors, centroids, lists, k, n_probe),
                brute_force_search(q, vectors, k)) for q in queries]
            row.append(f"p{n_probe}: {np.mean(recalls):.2f}")

        print(f"  {label:<34} {'  '.join(row)}")

    print()
    print("  Same algorithm, same settings, very different results.")
    print()
    print("  ANN indexes exploit the fact that real embeddings CLUSTER by topic,")
    print("  so a query's true neighbours tend to share its cluster. Remove the")
    print("  structure and clustering tells you nothing.")
    print()
    print("  This is why published recall numbers do not transfer between")
    print("  datasets - and why you measure on your own data.")
    print()


def experiment_filtering():
    """Show how post-filtering can return nothing while pre-filtering works."""
    print("=" * 76)
    print("  EXPERIMENT 3: pre-filter vs post-filter")
    print("=" * 76)
    print()

    n_vectors, dimensions = 1000, 32
    vectors = make_clustered_corpus(n_vectors, dimensions, n_topics=8, spread=0.2)

    rng = np.random.default_rng(3)
    # Deliberately RARE category: only ~1% of the corpus.
    metadatas = [
        {"dept": "legal" if rng.random() < 0.01 else "general", "year": 2024}
        for _ in range(n_vectors)
    ]
    n_legal = sum(1 for m in metadatas if m["dept"] == "legal")

    query = normalise(rng.normal(size=dimensions))
    k = 5

    print(f"  corpus: {n_vectors} docs, of which {n_legal} are dept='legal' "
          f"({100*n_legal/n_vectors:.1f}%)")
    print()

    for fetch in [10, 50, 200, 1000]:
        top = brute_force_search(query, vectors, fetch)
        survivors = [int(i) for i in top if metadatas[int(i)]["dept"] == "legal"]
        status = "OK" if len(survivors) >= k else "TOO FEW"
        print(f"  POST-filter, fetch top-{fetch:<5} -> {len(survivors)} legal docs "
              f"survive  [{status}]")

    pre = search_with_filter(query, vectors, metadatas, {"dept": "legal"}, k)
    print()
    print(f"  PRE-filter                   -> {len(pre)} results  [OK]")
    print()
    print("  Post-filtering on a rare category returns too few results, or none,")
    print("  no matter how good the semantic matches are - because the filter is")
    print("  applied AFTER the search has already chosen its candidates.")
    print()
    print("  Pre-filtering searches only the matching subset, so it always")
    print("  returns k results when k exist. With a 1% category, brute-forcing")
    print("  that subset is both simpler and faster than any index.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_recall_curve()
        experiment_structure_matters()
        experiment_filtering()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
