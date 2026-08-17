"""
starter.py - Lab 3: From Text to Meaning

Replace each TODO with working code. The self-test at the bottom checks your work.

    python starter.py

PART 1 (tasks 1-5) needs only numpy, so the self-test always runs.
PART 2 (tasks 6-7) needs tiktoken and sentence-transformers; those sections
skip gracefully with a message if the packages are not installed.

    pip install numpy tiktoken sentence-transformers
"""

import numpy as np


# ======================================================================
# PART 1 - THE MATH (numpy only)
# ======================================================================

# ----------------------------------------------------------------------
# TASK 1 - cosine_similarity
# Module 3, section 3.6
# ----------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors - the angle, ignoring length.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        A value from -1 (opposite) through 0 (unrelated) to 1 (same direction).

    Examples:
        >>> cosine_similarity(np.array([1., 2., 3.]), np.array([2., 4., 6.]))
        1.0
        >>> cosine_similarity(np.array([1., 0.]), np.array([0., 1.]))
        0.0
    """
    # TODO: divide the dot product by the product of the two magnitudes.
    # Hint: np.dot(a, b) and np.linalg.norm(a)
    return 0.0


# ----------------------------------------------------------------------
# TASK 2 - dot_product
# Module 3, section 3.6
# ----------------------------------------------------------------------

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product: multiply element-wise, then sum.

    Unlike cosine similarity this is sensitive to magnitude, so longer
    vectors produce larger scores.

    Examples:
        >>> dot_product(np.array([1., 2., 3.]), np.array([4., 5., 6.]))
        32.0
    """
    # TODO: return the dot product of a and b.
    return 0.0


# ----------------------------------------------------------------------
# TASK 3 - euclidean_distance
# Module 3, section 3.6
# ----------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two points.

    NOTE the direction: LOWER means more similar. This is the opposite of
    cosine similarity, and mixing them up is a classic silent bug.

    Examples:
        >>> euclidean_distance(np.array([0., 0.]), np.array([3., 4.]))
        5.0
    """
    # TODO: return the length of the difference vector (a - b).
    return 0.0


# ----------------------------------------------------------------------
# TASK 4 - normalise
# Module 3, section 3.6
# ----------------------------------------------------------------------

def normalise(v: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1 while keeping its direction.

    Once vectors are normalised, cosine similarity and dot product are
    identical - which is why production systems normalise on the way in.

    Examples:
        >>> normalise(np.array([3., 4.]))
        array([0.6, 0.8])
    """
    # TODO: divide v by its own magnitude.
    return v


# ----------------------------------------------------------------------
# TASK 5 - top_k_indices
# Module 3, section 3.7
# ----------------------------------------------------------------------

def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k highest scores, best first.

    Args:
        scores: 1-D array of similarity scores.
        k:      How many indices to return.

    Returns:
        Array of k indices, ordered from highest score to lowest.

    Examples:
        >>> top_k_indices(np.array([0.1, 0.9, 0.5, 0.3]), 2)
        array([1, 2])
    """
    # TODO:
    #   np.argsort sorts LOWEST to highest and returns indices.
    #   Reverse it with [::-1], then take the first k.
    return np.array([])


# ======================================================================
# PART 2 - REAL TEXT (needs tiktoken / sentence-transformers)
# ======================================================================

# ----------------------------------------------------------------------
# TASK 6 - count_tokens
# Module 3, section 3.3
# ----------------------------------------------------------------------

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens exactly, using a real tokenizer.

    Args:
        text:          The text to measure.
        encoding_name: Which vocabulary to use. "cl100k_base" is the
                       GPT-4 / GPT-3.5 family.

    Returns:
        The exact number of tokens.
    """
    import tiktoken

    # TODO:
    #   1. Get the encoder:  tiktoken.get_encoding(encoding_name)
    #   2. Encode the text, and return how many IDs came back.
    return 0


# ----------------------------------------------------------------------
# TASK 7 - search
# Module 3, section 3.7 - THE most important function in this lab
# ----------------------------------------------------------------------

def search(query: str, documents: list, doc_embeddings: np.ndarray,
           model, top_k: int = 3) -> list:
    """Find the documents most similar in meaning to a query.

    Args:
        query:          The search text.
        documents:      The original document strings.
        doc_embeddings: Pre-computed, NORMALISED document embeddings,
                        shape (n_documents, embedding_dim).
        model:          The SentenceTransformer used for doc_embeddings.
                        You MUST use the same model for the query.
        top_k:          How many results to return.

    Returns:
        A list of (document_text, score) tuples, best match first.
    """
    # TODO:
    #   1. Embed the query with the SAME model, normalised:
    #        model.encode(query, normalize_embeddings=True)
    #   2. Score it against every document. Because everything is
    #      normalised, a dot product IS the cosine similarity:
    #        doc_embeddings @ query_embedding
    #   3. Use your top_k_indices() to find the best ones.
    #   4. Return a list of (documents[i], float(scores[i])) tuples.
    return []


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _close(got, expected, tolerance=1e-6):
    """Compare floats or arrays allowing for tiny rounding differences."""
    try:
        return np.allclose(np.asarray(got, dtype=float),
                           np.asarray(expected, dtype=float), atol=tolerance)
    except Exception:
        return False


def _run_self_test() -> int:
    checks = [
        ("1. cosine_similarity (parallel -> 1.0)",
         cosine_similarity(np.array([1., 2., 3.]), np.array([2., 4., 6.])), 1.0),
        ("1. cosine_similarity (perpendicular -> 0.0)",
         cosine_similarity(np.array([1., 0.]), np.array([0., 1.])), 0.0),
        ("1. cosine_similarity (opposite -> -1.0)",
         cosine_similarity(np.array([1., 0.]), np.array([-1., 0.])), -1.0),
        ("1. cosine_similarity ignores magnitude",
         cosine_similarity(np.array([1., 1.]), np.array([50., 50.])), 1.0),

        ("2. dot_product",
         dot_product(np.array([1., 2., 3.]), np.array([4., 5., 6.])), 32.0),

        ("3. euclidean_distance (3-4-5 triangle)",
         euclidean_distance(np.array([0., 0.]), np.array([3., 4.])), 5.0),
        ("3. euclidean_distance (identical -> 0)",
         euclidean_distance(np.array([2., 7.]), np.array([2., 7.])), 0.0),

        ("4. normalise (3-4 -> 0.6-0.8)",
         normalise(np.array([3., 4.])), [0.6, 0.8]),
        ("4. normalise gives length 1",
         np.linalg.norm(normalise(np.array([5., -12., 3.]))), 1.0),

        ("5. top_k_indices (k=2)",
         top_k_indices(np.array([0.1, 0.9, 0.5, 0.3]), 2), [1, 2]),
        ("5. top_k_indices (k=1)",
         top_k_indices(np.array([0.1, 0.9, 0.5, 0.3]), 1), [1]),
        ("5. top_k_indices (k=4, full ranking)",
         top_k_indices(np.array([0.1, 0.9, 0.5, 0.3]), 4), [1, 2, 3, 0]),

        ("BONUS: normalised dot == cosine",
         dot_product(normalise(np.array([3., 1., 4.])), normalise(np.array([1., 5., 9.]))),
         cosine_similarity(np.array([3., 1., 4.]), np.array([1., 5., 9.]))),
    ]

    print()
    print("=" * 70)
    print("  LAB 3 SELF-TEST - Part 1 (the math)")
    print("=" * 70)
    print()

    failures = 0
    for name, got, expected in checks:
        if _close(got, expected):
            print(f"[ OK ]  {name}")
        else:
            failures += 1
            print(f"[FAIL]  {name}")
            print(f"          got:      {np.asarray(got)}")
            print(f"          expected: {np.asarray(expected)}")

    print()
    print("-" * 70)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  That last one is worth pausing on: once vectors are normalised,")
        print("  the dot product IS cosine similarity. Same ranking, less work.")
    else:
        print(f"  {failures} of {len(checks)} failing. Keep going.")
    print("-" * 70)
    print()
    return failures


# ======================================================================
# PART 2 demos - skip gracefully if packages are missing
# ======================================================================

def _demo_tokenizer():
    print("=" * 70)
    print("  PART 2a - EXACT TOKEN COUNTING")
    print("=" * 70)
    print()

    try:
        import tiktoken
    except ImportError:
        print("  tiktoken not installed - skipping.")
        print("    pip install tiktoken")
        print()
        return

    samples = {
        "Plain English": "The quick brown fox jumps over the lazy dog.",
        "Python code": "def f(x): return {k: v for k, v in x.items()}",
        "Long number": "The total was 1234567890 dollars.",
        "Technical": "Retrieval-augmented generation with reranking.",
        "Spanish": "La inteligencia artificial esta transformando el mundo.",
    }

    print(f"  {'Sample':<16}{'chars':>7}{'tokens':>8}{'chars/token':>13}")
    print("  " + "-" * 44)
    for label, text in samples.items():
        n = count_tokens(text)
        if n == 0:
            print("  count_tokens not implemented yet - finish TASK 6.")
            print()
            return
        print(f"  {label:<16}{len(text):>7}{n:>8}{len(text)/n:>13.2f}")

    print()
    print("  Notice how far code and numbers fall below 4 chars/token.")
    print("  The rule of thumb is an English-prose rule of thumb.")
    print()


def _demo_search():
    print("=" * 70)
    print("  PART 2b - SEMANTIC SEARCH")
    print("=" * 70)
    print()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed - skipping.")
        print("    pip install sentence-transformers")
        print()
        return

    print("  Loading all-MiniLM-L6-v2 (downloads ~90 MB the first time)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "The stock market crashed today.",
        "Investors lost money in the financial crisis.",
        "LLMs can generate human-like text.",
        "Artificial intelligence writes like a human.",
    ]

    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    print(f"  Indexed {len(documents)} documents as {doc_embeddings.shape} vectors")
    print()

    for query in ["cat on a mat", "AI writing", "money problems"]:
        results = search(query, documents, doc_embeddings, model, top_k=3)
        print(f"  Query: {query!r}")
        if not results:
            print("    search() not implemented yet - finish TASK 7.")
            print()
            return
        for text, score in results:
            print(f"    {score:.3f}  {text}")
        print()

    print("  For 'cat on a mat', look at the SECOND result. It shares no words")
    print("  with the query at all - found purely by meaning. That is the whole")
    print("  point, and it is the engine inside every RAG system.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        _demo_tokenizer()
        _demo_search()
    else:
        print("  Fix Part 1 first, then the Part 2 demos will run.")
        print()
