"""
solution.py - Lab 7 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import numpy as np


# ======================================================================
# Helpers
# ======================================================================

def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to unit length so dot product == cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def kmeans(vectors: np.ndarray, n_clusters: int, seed: int = 0,
           iterations: int = 25) -> np.ndarray:
    """Plain k-means. Returns the cluster centroids."""
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(len(vectors), n_clusters, replace=False)].copy()

    for _ in range(iterations):
        # For unit-length vectors, maximising cosine == minimising distance,
        # so one matrix product does the assignment. The naive broadcast form
        # allocates (n_vectors x n_clusters x dimensions) - about 2.9 GB per
        # iteration at 20,000 vectors and 140 clusters. This allocates
        # (n_vectors x n_clusters), roughly 1000x less.
        labels = (vectors @ centroids.T).argmax(axis=1)

        for cluster in range(n_clusters):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(0)

        # Keep centroids on the unit sphere so the cosine shortcut holds.
        centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)

    return centroids


def make_clustered_corpus(n_vectors: int, dimensions: int, n_topics: int,
                          spread: float = 0.15, seed: int = 7) -> np.ndarray:
    """Generate synthetic vectors that CLUSTER, like real embeddings do."""
    rng = np.random.default_rng(seed)
    topic_centres = normalise(rng.normal(size=(n_topics, dimensions)))
    assignments = rng.integers(0, n_topics, size=n_vectors)
    points = topic_centres[assignments] + rng.normal(
        scale=spread, size=(n_vectors, dimensions))
    return normalise(points)


# ======================================================================
# TASK 1 - brute_force_search
# ======================================================================

def brute_force_search(query: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Exact nearest-neighbour search. The ground truth."""
    # One matrix-vector product scores the whole corpus. Because everything
    # is normalised, these dot products ARE cosine similarities.
    scores = vectors @ query

    # argsort is ascending, so reverse for best-first, then take k.
    # Slicing past the end is safe: k > n just returns everything.
    return np.argsort(scores)[::-1][:k]


# ======================================================================
# TASK 2 - build_ivf_index
# ======================================================================

def build_ivf_index(vectors: np.ndarray, centroids: np.ndarray) -> dict:
    """Assign every vector to its nearest centroid, building inverted lists."""
    # (n_vectors, dims) @ (dims, n_clusters) -> (n_vectors, n_clusters)
    similarities = vectors @ centroids.T
    assignments = similarities.argmax(axis=1)

    # Pre-create a list for EVERY cluster, including ones that end up empty.
    # A cluster can legitimately attract no vectors, and ivf_search must be
    # able to look up any centroid id without a KeyError.
    inverted_lists = {cluster: [] for cluster in range(len(centroids))}

    for vector_index, cluster in enumerate(assignments):
        inverted_lists[int(cluster)].append(vector_index)

    return inverted_lists


# ======================================================================
# TASK 3 - ivf_search
# ======================================================================

def ivf_search(query: np.ndarray, vectors: np.ndarray, centroids: np.ndarray,
               inverted_lists: dict, k: int, n_probe: int = 1) -> np.ndarray:
    """Approximate search: only scan the n_probe nearest clusters."""
    # STEP 1: which clusters look promising? Comparing against a handful of
    # centroids is cheap - that is the entire saving.
    centroid_scores = centroids @ query
    probed = np.argsort(centroid_scores)[::-1][:n_probe]

    # STEP 2: gather the candidate vector indices from those clusters.
    candidates = [index for cluster in probed
                  for index in inverted_lists[int(cluster)]]

    if not candidates:
        return np.array([], dtype=int)

    candidates = np.array(candidates, dtype=int)

    # STEP 3: exact search, but only over the candidates.
    candidate_scores = vectors[candidates] @ query

    # STEP 4: map back to ORIGINAL corpus indices.
    # argsort here gives positions WITHIN `candidates`, so we must index
    # through `candidates` to recover real corpus indices. Forgetting this
    # is the classic bug in this task - and it returns plausible garbage.
    best_positions = np.argsort(candidate_scores)[::-1][:k]
    return candidates[best_positions]


# ======================================================================
# TASK 4 - recall_at_k
# ======================================================================

def recall_at_k(approximate: np.ndarray, exact: np.ndarray) -> float:
    """What fraction of the TRUE top-k did the approximate search find?"""
    exact_set = {int(i) for i in exact}

    # Nothing to miss: vacuously perfect. Also avoids dividing by zero.
    if not exact_set:
        return 1.0

    approximate_set = {int(i) for i in approximate}

    # Set intersection - recall asks WHICH results were found, not in what
    # order. Ranking quality is a different metric (NDCG, MRR).
    return len(approximate_set & exact_set) / len(exact_set)


# ======================================================================
# TASK 5 - filter_by_metadata
# ======================================================================

def _matches(record: dict, key: str, condition) -> bool:
    """Does one record satisfy one condition?"""
    if key not in record:
        return False

    value = record[key]

    # A dict condition means an operator; anything else is exact equality.
    if isinstance(condition, dict):
        for operator, operand in condition.items():
            if operator == "$gte" and not value >= operand:
                return False
            if operator == "$lte" and not value <= operand:
                return False
            if operator == "$in" and value not in operand:
                return False
        return True

    return value == condition


def filter_by_metadata(metadatas: list, filters: dict) -> np.ndarray:
    """Return the indices of records matching ALL the given filters."""
    matches = []

    for index, record in enumerate(metadatas):
        # all() over an empty filters dict is True, so no filters means
        # everything matches - which is the behaviour we want.
        if all(_matches(record, key, condition) for key, condition in filters.items()):
            matches.append(index)

    return np.array(matches, dtype=int)


# ======================================================================
# TASK 6 - search_with_filter
# ======================================================================

def search_with_filter(query: np.ndarray, vectors: np.ndarray, metadatas: list,
                       filters: dict, k: int) -> np.ndarray:
    """PRE-filtered exact search: narrow the corpus first, then search it."""
    allowed = filter_by_metadata(metadatas, filters)

    if len(allowed) == 0:
        return np.array([], dtype=int)

    # Score ONLY the permitted subset. Note the security property: documents
    # outside the filter are never even scored, let alone returned.
    scores = vectors[allowed] @ query

    # Same index-mapping step as ivf_search.
    best_positions = np.argsort(scores)[::-1][:k]
    return allowed[best_positions]


# ======================================================================
# Demonstrations
# ======================================================================

def demo_index_comparison():
    print("=" * 76)
    print("  FLAT vs IVF - what the approximation costs")
    print("=" * 76)
    print()

    import time

    n_vectors, dimensions, k = 20_000, 128, 10
    vectors = make_clustered_corpus(n_vectors, dimensions, n_topics=40, spread=0.15)
    centroids = normalise(kmeans(vectors, 140, seed=1))
    lists = build_ivf_index(vectors, centroids)

    rng = np.random.default_rng(11)
    queries = [normalise(vectors[rng.integers(0, n_vectors)]
                         + rng.normal(scale=0.08, size=dimensions))
               for _ in range(30)]

    exact_results = []
    start = time.perf_counter()
    for query in queries:
        exact_results.append(brute_force_search(query, vectors, k))
    flat_ms = (time.perf_counter() - start) / len(queries) * 1000

    print(f"  corpus: {n_vectors:,} vectors x {dimensions} dims, "
          f"{len(centroids)} clusters")
    print()
    print(f"  {'index':<18}{'recall@10':>11}{'ms/query':>11}{'speedup':>10}")
    print("  " + "-" * 50)
    print(f"  {'Flat (exact)':<18}{1.00:>11.2f}{flat_ms:>11.2f}{'1.0x':>10}")

    for n_probe in [1, 4, 12, 40]:
        start = time.perf_counter()
        recalls = []
        for query, exact in zip(queries, exact_results):
            approx = ivf_search(query, vectors, centroids, lists, k, n_probe)
            recalls.append(recall_at_k(approx, exact))
        ivf_ms = (time.perf_counter() - start) / len(queries) * 1000

        label = f"IVF n_probe={n_probe}"
        print(f"  {label:<18}{np.mean(recalls):>11.2f}{ivf_ms:>11.2f}"
              f"{flat_ms / max(ivf_ms, 1e-9):>9.1f}x")

    print()
    print("  Real timings on 20,000 vectors. Note that at this size Flat is")
    print("  already fast - which is exactly the point of section 7.1. The")
    print("  approximation only earns its complexity once N is large.")
    print()


def demo_dimension_cost():
    print("=" * 76)
    print("  WHY EMBEDDING DIMENSION IS A COST DECISION")
    print("=" * 76)
    print()

    print(f"  {'vectors':>12}{'dim 384':>12}{'dim 768':>12}{'dim 1536':>12}"
          f"{'dim 3072':>12}")
    print("  " + "-" * 60)

    for n in [100_000, 1_000_000, 10_000_000, 100_000_000]:
        row = f"  {n:>12,}"
        for dimensions in [384, 768, 1536, 3072]:
            gigabytes = n * dimensions * 4 / 1e9      # float32 = 4 bytes
            row += f"{gigabytes:>11.1f}G"
        print(row)

    print()
    print("  Raw vector storage only - HNSW adds roughly another 1.5-2x for")
    print("  its graph, and most ANN indexes want to be resident in RAM.")
    print()
    print("  Halving the dimension halves storage AND roughly halves search")
    print("  time. Test whether a 384-dim model is good enough before paying")
    print("  4x for 1536 (Module 3, section 3.5).")
    print()


def demo_access_control():
    print("=" * 76)
    print("  WHY POST-FILTERING IS NOT ACCESS CONTROL")
    print("=" * 76)
    print()

    dimensions = 16
    rng = np.random.default_rng(1)
    vectors = normalise(rng.normal(size=(6, dimensions)))

    documents = [
        "Public: company holiday schedule",
        "SECRET: acquisition plan for competitor",
        "Public: office wifi password policy",
        "SECRET: executive compensation review",
        "Public: onboarding checklist",
        "SECRET: pending litigation summary",
    ]
    metadatas = [{"level": "secret" if "SECRET" in d else "public"} for d in documents]

    # A user cleared only for public documents.
    query = normalise(rng.normal(size=dimensions))

    print("  POST-filter (WRONG for permissions):")
    fetched = brute_force_search(query, vectors, 6)
    print(f"    step 1 - retrieved {len(fetched)} docs into the process, including:")
    for i in fetched[:3]:
        print(f"             - {documents[int(i)]}")
    print("    step 2 - now discard the secret ones")
    print()
    print("    The secret documents were READ into memory before being dropped.")
    print("    One log line, error message or debugger session away from leaking.")
    print()

    print("  PRE-filter (correct):")
    allowed = search_with_filter(query, vectors, metadatas, {"level": "public"}, 3)
    for i in allowed:
        print(f"    - {documents[int(i)]}")
    print()
    print("    Secret documents were never scored, never fetched, never present.")
    print("    Filter for permissions at the query boundary, not afterwards.")
    print()


if __name__ == "__main__":
    demo_index_comparison()
    demo_dimension_cost()
    demo_access_control()
