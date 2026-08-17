"""
solution.py - Lab 8 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import math
import re
from collections import Counter


# ======================================================================
# TASK 1 - chunk_fixed
# ======================================================================

def chunk_fixed(text: str, chunk_size: int, overlap: int = 0) -> list:
    """Split text every chunk_size characters, with optional overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # step = chunk_size - overlap. With overlap >= chunk_size the step is zero
    # or negative, and range() with a non-positive step either yields nothing
    # or (with hand-rolled loops) never advances. Guard it explicitly rather
    # than shipping a pipeline that hangs on a config typo.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


# ======================================================================
# TASK 2 - chunk_recursive
# ======================================================================

def chunk_recursive(text: str, chunk_size: int,
                    separators: tuple = ("\n\n", "\n", " ", "")) -> list:
    """Split on the largest separator that keeps pieces under chunk_size."""
    if not text:
        return []

    # Base case: it already fits, so leave it whole.
    if len(text) <= chunk_size:
        return [text]

    # Out of separators, or explicitly asked for a hard split.
    # This MUST come before any .split() call: text.split("") raises
    # ValueError: empty separator.
    if not separators or separators[0] == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, remaining = separators[0], separators[1:]

    chunks = []
    for piece in text.split(separator):
        if not piece:
            continue                      # drop the empties split() produces
        if len(piece) <= chunk_size:
            chunks.append(piece)
        else:
            # Still too big - try the next, finer separator.
            chunks.extend(chunk_recursive(piece, chunk_size, remaining))

    return chunks


# ======================================================================
# TASK 3 - tokenize_words
# ======================================================================

def tokenize_words(text: str) -> list:
    """Split text into lowercase alphanumeric tokens."""
    if not text:
        return []
    # Lowercase first, so the pattern only needs the lowercase range.
    return re.findall(r"[a-z0-9]+", text.lower())


# ======================================================================
# TASK 4 - bm25_scores
# ======================================================================

def bm25_scores(query: str, documents: list,
                k1: float = 1.5, b: float = 0.75) -> list:
    """Score every document against the query using BM25."""
    if not documents:
        return []

    doc_tokens = [tokenize_words(d) for d in documents]
    n_documents = len(documents)

    total_length = sum(len(tokens) for tokens in doc_tokens)
    average_length = total_length / n_documents if n_documents else 0.0

    # Document frequency: in how many DOCUMENTS does each term appear?
    # set(tokens) is essential - a term appearing 5 times in one document
    # must still count as one document, or IDF is wrong.
    document_frequency = Counter()
    for tokens in doc_tokens:
        document_frequency.update(set(tokens))

    query_terms = tokenize_words(query)

    scores = []
    for tokens in doc_tokens:
        term_frequency = Counter(tokens)
        score = 0.0

        for term in query_terms:
            # A term the document does not contain contributes exactly 0,
            # so skipping it is both correct and faster.
            if term not in term_frequency:
                continue

            # IDF: rare terms carry more information than common ones.
            n_containing = document_frequency[term]
            idf = math.log(
                (n_documents - n_containing + 0.5) / (n_containing + 0.5) + 1)

            frequency = term_frequency[term]

            # The k1 term saturates: the 10th occurrence adds far less than
            # the 2nd. The b term normalises by length, so a match in a short
            # document counts for more than the same match in a long one.
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / average_length)

            score += idf * numerator / denominator

        scores.append(score)

    return scores


# ======================================================================
# TASK 5 - reciprocal_rank_fusion
# ======================================================================

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> list:
    """Merge several ranked lists into one, using rank position only."""
    fused = {}

    for ranking in rankings:
        # start=1 because a rank of 0 would make 1/(k+0) = 1/k, treating the
        # top result as infinitely good when k is 0.
        for position, document_id in enumerate(ranking, start=1):
            fused[document_id] = fused.get(document_id, 0.0) + 1.0 / (k + position)

    # Descending by fused score. Python's sort is stable, so documents with
    # identical scores keep the order they were first seen in.
    return sorted(fused, key=lambda document_id: -fused[document_id])


# ======================================================================
# TASK 6 - build_grounded_prompt
# ======================================================================

def build_grounded_prompt(question: str, chunks: list) -> str:
    """Build a prompt that answers ONLY from the given context."""
    # 1-based numbering gives the model a citation vocabulary ("[2]") that
    # your code can parse back to a source.
    context = "\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))

    return (
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, reply exactly: I don't know.\n"
        "Cite the sources you used as [1], [2], etc.\n\n"
        f"CONTEXT:\n{context}\n\n"
        # The question goes LAST: recall is strongest at the end of a context
        # (Module 3, section 3.9), and it sits right before generation begins.
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


RAG_SYSTEM_PROMPT = """You are a documentation assistant. You answer questions
strictly from the provided context.

RULES:
- Use ONLY information present in the context. Never add outside knowledge.
- Cite every claim with the bracketed number of its source, e.g. [2].
- If the context does not answer the question, reply exactly: I don't know.
- If sources conflict, say so and cite both.
- Quote exact figures, names and dates rather than paraphrasing them.
- Do not speculate, and do not offer advice beyond the context.
"""


# ======================================================================
# TASK 7 - citations
# ======================================================================

def extract_cited_indices(answer: str) -> list:
    """Pull the [n] citation markers out of an answer."""
    if not answer:
        return []
    # A set de-duplicates repeated markers; sorted() gives a stable order.
    return sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})


def validate_citations(answer: str, n_chunks: int) -> tuple:
    """Check that citation markers are present and in range."""
    problems = []
    cited = extract_cited_indices(answer)

    # An answer that makes claims but cites nothing usually means the model
    # ignored the context and answered from training knowledge - the exact
    # failure RAG exists to prevent. A refusal is legitimately uncited.
    if not cited and "i don't know" not in answer.lower():
        problems.append("answer makes claims but cites no sources")

    # Markers are 1-based, so anything outside 1..n_chunks was invented.
    out_of_range = [index for index in cited if not 1 <= index <= n_chunks]
    if out_of_range:
        problems.append(f"cited non-existent sources: {out_of_range}")

    return (not problems, problems)


def attach_sources(answer: str, chunks: list) -> dict:
    """Resolve citation markers back to their source documents.

    Args:
        answer: The model's answer, containing [n] markers.
        chunks: The chunks that were supplied, as {text, metadata} dicts.

    Returns:
        {answer, sources} where each source has marker, source, page, excerpt.
    """
    sources = []

    for index in extract_cited_indices(answer):
        # Skip invented markers rather than raising: validate_citations already
        # reported them, and a partly-attributed answer beats a crash.
        if not 1 <= index <= len(chunks):
            continue

        chunk = chunks[index - 1]
        metadata = chunk.get("metadata", {})
        sources.append({
            "marker": index,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            # Showing an excerpt is what makes a citation verifiable in
            # two seconds rather than requiring trust.
            "excerpt": chunk["text"][:200],
        })

    return {"answer": answer, "sources": sources}


# ======================================================================
# Demonstrations
# ======================================================================

SAMPLE_CHUNKS = [
    "Metformin is a first-line treatment for type 2 diabetes mellitus.",
    "The maximum daily dose of metformin is 2000 mg for immediate-release "
    "formulations, or up to 2550 mg with careful monitoring.",
    "Extended-release metformin is dosed once daily, with a maximum of 2000 mg.",
    "Metformin is contraindicated when eGFR falls below 30 mL/min/1.73m2.",
    "Common side effects of metformin include nausea and transient GI discomfort.",
    "Product SKU-4471 is our 500 mg metformin tablet, 60-count bottle.",
    "Insulin therapy requires careful blood glucose monitoring and dose titration.",
    "Error E1042 indicates a failed prescription validation in the ordering system.",
]


def demo_overlap_cost():
    print("=" * 76)
    print("  THE COST OF OVERLAP")
    print("=" * 76)
    print()

    document = " ".join(f"sentence{i} about dosing guidance." for i in range(120))
    chunk_size = 200

    print(f"  document: {len(document)} characters, chunk_size = {chunk_size}")
    print()
    print(f"  {'overlap':>9}{'%':>6}{'chunks':>9}{'total chars':>14}{'storage vs 0%':>16}")
    print("  " + "-" * 54)

    baseline = None
    for overlap in [0, 20, 40, 100, 150]:
        chunks = chunk_fixed(document, chunk_size, overlap)
        total = sum(len(c) for c in chunks)
        if baseline is None:
            baseline = total
        percent = 100 * overlap / chunk_size
        print(f"  {overlap:>9}{percent:>5.0f}%{len(chunks):>9}{total:>14,}"
              f"{total / baseline:>15.2f}x")

    print()
    print("  At 50% overlap you store twice the text - and your top-5 retrieved")
    print("  chunks are really about two distinct passages, because neighbouring")
    print("  chunks are near-duplicates of each other.")
    print()
    print("  10-20% is the useful range: enough to stop an answer being split")
    print("  across a boundary, not enough to fill your results with repeats.")
    print()


def demo_chunk_size_tradeoff():
    print("=" * 76)
    print("  CHUNK SIZE: PRECISION vs CONTEXT")
    print("=" * 76)
    print()

    document = (
        "Metformin Dosing.\n\n"
        "The maximum daily dose is 2000 mg for immediate-release tablets. "
        "Doses above this require careful monitoring of renal function and "
        "vitamin B12 levels.\n\n"
        "Contraindications.\n\n"
        "Metformin is contraindicated when eGFR falls below 30. Use with "
        "caution between 30 and 45."
    )
    question = "What is the maximum daily dose?"

    for chunk_size in [40, 100, 250, 1000]:
        chunks = chunk_recursive(document, chunk_size)
        scores = bm25_scores(question, chunks)
        best = scores.index(max(scores)) if max(scores) > 0 else None

        print(f"  chunk_size = {chunk_size:<5} -> {len(chunks)} chunks")
        if best is None:
            print("    best match: (nothing matched)")
        else:
            preview = chunks[best].replace("\n", " ")[:66]
            answers = "2000" in chunks[best]
            print(f"    best match: {preview!r}")
            print(f"    contains the answer: {answers}")
        print()

    print("  Too small and the chunk holding '2000 mg' loses the words that")
    print("  make it findable. Too large and the retrieved chunk carries a lot")
    print("  of text that has nothing to do with the question.")
    print()
    print("  There is no universally right size - which is why you need an")
    print("  evaluation set to choose one (Module 8, section 8.11).")
    print()


def demo_rrf_behaviour():
    print("=" * 76)
    print("  WHY RRF USES RANKS, NOT SCORES")
    print("=" * 76)
    print()

    # Two retrievers on wildly different scales.
    dense_scores = {0: 0.81, 1: 0.79, 2: 0.77, 3: 0.40}
    sparse_scores = {3: 14.2, 2: 11.9, 0: 0.6, 1: 0.0}

    print("  dense (cosine, 0-1):   ", {k: round(v, 2) for k, v in dense_scores.items()})
    print("  sparse (BM25, unbounded):", sparse_scores)
    print()

    naive = sorted({d: dense_scores.get(d, 0) + sparse_scores.get(d, 0)
                    for d in range(4)}.items(), key=lambda x: -x[1])
    print(f"  NAIVE score addition -> {[d for d, _ in naive]}")
    print("    BM25 dominates completely: its 14.2 swamps every cosine value,")
    print("    so the dense retriever effectively gets no vote at all.")
    print()

    dense_ranking = sorted(dense_scores, key=lambda d: -dense_scores[d])
    sparse_ranking = sorted(sparse_scores, key=lambda d: -sparse_scores[d])
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    print(f"  RRF fusion           -> {fused}")
    print("    Both retrievers get an equal vote, because only POSITION counts.")
    print("    Documents ranked well by both rise; a runaway score cannot")
    print("    hijack the result.")
    print()


def demo_end_to_end_offline():
    print("=" * 76)
    print("  A COMPLETE RETRIEVAL PASS (offline, no model call)")
    print("=" * 76)
    print()

    # Give the chunks metadata, as a real loader would.
    chunks = [
        {"text": text, "metadata": {"source": "formulary.pdf", "page": 10 + i}}
        for i, text in enumerate(SAMPLE_CHUNKS)
    ]
    texts = [c["text"] for c in chunks]

    question = "What is the maximum daily dose of metformin?"

    # Keyword retrieval standing in for the full hybrid pass.
    scores = bm25_scores(question, texts)
    ranking = sorted(range(len(texts)), key=lambda i: -scores[i])
    top = ranking[:3]
    retrieved = [chunks[i] for i in top]

    print(f"  question: {question}")
    print()
    print("  retrieved:")
    for rank, i in enumerate(top, 1):
        print(f"    {rank}. (score {scores[i]:.2f}) {texts[i][:62]}")
    print()

    prompt = build_grounded_prompt(question, [c["text"] for c in retrieved])
    print(f"  prompt length: {len(prompt)} characters")
    print()

    # Simulate a well-behaved model response.
    simulated_answer = (
        "The maximum daily dose of metformin is 2000 mg for immediate-release "
        "formulations, or up to 2550 mg with careful monitoring [1]. "
        "Extended-release is capped at 2000 mg once daily [2]."
    )

    is_valid, problems = validate_citations(simulated_answer, len(retrieved))
    result = attach_sources(simulated_answer, retrieved)

    print(f"  answer valid: {is_valid}   problems: {problems or '-'}")
    print()
    print("  answer:")
    print(f"    {result['answer']}")
    print()
    print("  sources (this is what makes it verifiable):")
    for source in result["sources"]:
        print(f"    [{source['marker']}] {source['source']} p.{source['page']}")
        print(f"        \"{source['excerpt'][:70]}...\"")
    print()


if __name__ == "__main__":
    demo_overlap_cost()
    demo_chunk_size_tradeoff()
    demo_rrf_behaviour()
    demo_end_to_end_offline()
