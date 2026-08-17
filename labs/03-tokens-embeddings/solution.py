"""
solution.py - Lab 3 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import numpy as np


# ======================================================================
# PART 1 - THE MATH
# ======================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors - the angle, ignoring length."""
    # Dividing by both magnitudes cancels out length, leaving only direction.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product: multiply element-wise, then sum."""
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance. LOWER means more similar."""
    # np.linalg.norm of the difference is exactly sqrt(sum((a-b)^2)).
    return float(np.linalg.norm(a - b))


def normalise(v: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1 while keeping its direction."""
    return v / np.linalg.norm(v)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k highest scores, best first."""
    # argsort is ascending, so reverse it to get descending, then take k.
    return np.argsort(scores)[::-1][:k]


# ======================================================================
# PART 2 - REAL TEXT
# ======================================================================

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens exactly, using a real tokenizer."""
    import tiktoken

    encoder = tiktoken.get_encoding(encoding_name)
    return len(encoder.encode(text))


def search(query: str, documents: list, doc_embeddings: np.ndarray,
           model, top_k: int = 3) -> list:
    """Find the documents most similar in meaning to a query."""
    # CRITICAL: the query must be embedded by the SAME model as the documents,
    # or the two sets of vectors live in unrelated coordinate systems and the
    # scores are meaningless. Nothing errors - you just get wrong answers.
    query_embedding = model.encode(query, normalize_embeddings=True)

    # One matrix multiply scores the query against every document at once.
    # Everything is normalised, so these dot products ARE cosine similarities.
    scores = doc_embeddings @ query_embedding

    best = top_k_indices(scores, top_k)
    return [(documents[i], float(scores[i])) for i in best]


# ======================================================================
# Demonstrations
# ======================================================================

def demo_tokenizer():
    print("=" * 70)
    print("  EXACT TOKEN COUNTING")
    print("=" * 70)
    print()

    try:
        import tiktoken
    except ImportError:
        print("  tiktoken not installed. pip install tiktoken")
        print()
        return

    encoder = tiktoken.get_encoding("cl100k_base")

    # --- See the actual pieces ---
    text = "Explain embeddings simply."
    ids = encoder.encode(text)
    print(f"  {text!r}  ->  {len(ids)} tokens")
    for token_id in ids:
        # !r shows the repr, so leading spaces are visible rather than lost.
        print(f"    {token_id:>7}  ->  {encoder.decode([token_id])!r}")
    print()

    # --- Where the 4-chars-per-token rule holds and breaks ---
    samples = {
        "Plain English": "The quick brown fox jumps over the lazy dog.",
        "Python code": "def f(x): return {k: v for k, v in x.items()}",
        "Long number": "The total was 1234567890 dollars.",
        "Technical": "Retrieval-augmented generation with reranking.",
        "Spanish": "La inteligencia artificial esta transformando el mundo.",
    }

    print(f"  {'Sample':<16}{'chars':>7}{'tokens':>8}{'chars/token':>13}")
    print("  " + "-" * 44)
    for label, sample in samples.items():
        n = count_tokens(sample)
        print(f"  {label:<16}{len(sample):>7}{n:>8}{len(sample)/n:>13.2f}")
    print()

    # --- Tokenization is lossless ---
    original = "Tokenization is lossless."
    print(f"  Round-trip identical: {encoder.decode(encoder.encode(original)) == original}")
    print()


def demo_search():
    print("=" * 70)
    print("  SEMANTIC SEARCH")
    print("=" * 70)
    print()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed.")
        print("    pip install sentence-transformers")
        print()
        return

    print("  Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "The stock market crashed today.",
        "Investors lost money in the financial crisis.",
        "LLMs can generate human-like text.",
        "Artificial intelligence writes like a human.",
    ]

    # INDEXING: embed every document once, up front.
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    print(f"  Indexed {len(documents)} docs -> {doc_embeddings.shape}")
    print()

    for query in ["cat on a mat", "AI writing", "money problems", "SKU-4471"]:
        print(f"  Query: {query!r}")
        for text, score in search(query, documents, doc_embeddings, model, top_k=3):
            print(f"    {score:.3f}  {text}")
        print()

    print("  Two things to notice:")
    print("   - 'cat on a mat' finds 'A kitten rested on the rug' despite ZERO")
    print("     shared words. Meaning, not spelling.")
    print("   - 'SKU-4471' returns nonsense with middling confidence. Semantic")
    print("     search has no idea what an exact product code is. This is why")
    print("     production systems use HYBRID search - Module 8.")
    print()


def demo_metrics_agree():
    print("=" * 70)
    print("  WHY NORMALISATION MATTERS")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    query = rng.normal(size=8)
    docs = rng.normal(size=(5, 8))

    print("  RAW vectors - the three metrics disagree on ranking:")
    cos_rank = np.argsort([cosine_similarity(query, d) for d in docs])[::-1]
    dot_rank = np.argsort([dot_product(query, d) for d in docs])[::-1]
    euc_rank = np.argsort([euclidean_distance(query, d) for d in docs])  # ascending!
    print(f"    by cosine    : {cos_rank.tolist()}")
    print(f"    by dot        : {dot_rank.tolist()}")
    print(f"    by euclidean  : {euc_rank.tolist()}")
    print()

    query_n = normalise(query)
    docs_n = np.array([normalise(d) for d in docs])

    print("  NORMALISED vectors - all three now agree:")
    cos_rank = np.argsort([cosine_similarity(query_n, d) for d in docs_n])[::-1]
    dot_rank = np.argsort([dot_product(query_n, d) for d in docs_n])[::-1]
    euc_rank = np.argsort([euclidean_distance(query_n, d) for d in docs_n])
    print(f"    by cosine    : {cos_rank.tolist()}")
    print(f"    by dot        : {dot_rank.tolist()}")
    print(f"    by euclidean  : {euc_rank.tolist()}")
    print()
    print("  Note euclidean is sorted ASCENDING - lower distance is better.")
    print("  Sort it descending and you retrieve your WORST matches, silently.")
    print()


if __name__ == "__main__":
    demo_tokenizer()
    demo_search()
    demo_metrics_agree()
