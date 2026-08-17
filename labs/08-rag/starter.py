"""
starter.py - Lab 8: Build a Document Q&A Bot

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-7) is pure standard library - no packages, no API key.
You will implement chunking, BM25 keyword scoring, reciprocal rank fusion,
grounded prompting and citation validation from scratch.

PART 2 (in the lab brief) assembles them into a working RAG bot over your
own PDFs.
"""

import math
import re
from collections import Counter


# ======================================================================
# TASK 1 - chunk_fixed
# Module 8, section 8.4 (strategy 1)
# ======================================================================

def chunk_fixed(text: str, chunk_size: int, overlap: int = 0) -> list:
    """Split text every chunk_size characters, with optional overlap.

    Args:
        text:       The text to split.
        chunk_size: Characters per chunk.
        overlap:    Characters repeated from the end of the previous chunk.

    Returns:
        A list of chunks. Empty text gives an empty list.

    Raises:
        ValueError: if chunk_size <= 0, or overlap >= chunk_size.

    Examples:
        >>> chunk_fixed("abcdefghij", 4, 0)
        ['abcd', 'efgh', 'ij']
        >>> chunk_fixed("abcdefghij", 4, 2)
        ['abcd', 'cdef', 'efgh', 'ghij', 'ij']
    """
    # TODO:
    #   1. Raise ValueError if chunk_size <= 0.
    #   2. Raise ValueError if overlap >= chunk_size.
    #      WHY: step = chunk_size - overlap. If overlap >= chunk_size the step
    #      is zero or negative and range() either loops forever or yields
    #      nothing. This is a real bug that hangs a pipeline, so guard it.
    #   3. Return [] for empty text.
    #   4. step = chunk_size - overlap, then slice at each step.
    return []


# ======================================================================
# TASK 2 - chunk_recursive
# Module 8, section 8.4 (strategy 2) - the sensible default
# ======================================================================

def chunk_recursive(text: str, chunk_size: int,
                    separators: tuple = ("\n\n", "\n", " ", "")) -> list:
    """Split on the largest separator that keeps pieces under chunk_size.

    Tries paragraph breaks first, then line breaks, then spaces, then a hard
    character split - so natural boundaries survive wherever possible. This
    is the algorithm behind LangChain's RecursiveCharacterTextSplitter.

    Args:
        text:       The text to split.
        chunk_size: Maximum characters per chunk.
        separators: Separators to try, coarsest first. "" means hard split.

    Returns:
        A list of chunks, none longer than chunk_size.

    Examples:
        >>> chunk_recursive("aaa\\n\\nbbb\\n\\nccc", 5)
        ['aaa', 'bbb', 'ccc']
        >>> chunk_recursive("hello world foo", 6)
        ['hello', 'world', 'foo']
        >>> chunk_recursive("short", 10)
        ['short']
    """
    # TODO:
    #   1. Empty text -> []
    #   2. len(text) <= chunk_size -> [text]        (base case)
    #   3. If separators is empty, OR separators[0] == "":
    #        hard-split into chunk_size pieces and return.
    #      IMPORTANT: text.split("") raises ValueError in Python, so the ""
    #      case must be handled BEFORE any call to .split().
    #   4. Otherwise: separator, remaining = separators[0], separators[1:]
    #        - split text on separator, dropping empty pieces
    #        - for each piece: keep it if it fits, else RECURSE with `remaining`
    #        - return the flattened list
    return []


# ======================================================================
# TASK 3 - tokenize_words
# Module 8, section 8.6
# ======================================================================

def tokenize_words(text: str) -> list:
    """Split text into lowercase alphanumeric tokens.

    Crude on purpose - no stemming, no stop-word removal. Good enough for
    BM25, and it keeps the next task focused on the scoring formula.

    Examples:
        >>> tokenize_words("Metformin: max 2000 mg/day!")
        ['metformin', 'max', '2000', 'mg', 'day']
        >>> tokenize_words("")
        []
    """
    # TODO: lowercase the text, then return all runs of letters/digits.
    # Hint: re.findall(r"[a-z0-9]+", text.lower())
    return []


# ======================================================================
# TASK 4 - bm25_scores
# Module 8, section 8.6 - keyword relevance
# ======================================================================

def bm25_scores(query: str, documents: list,
                k1: float = 1.5, b: float = 0.75) -> list:
    """Score every document against the query using BM25.

        score(D,Q) = SUM over q in Q of
                     IDF(q) * (f * (k1+1)) / (f + k1*(1 - b + b*|D|/avgdl))

    where f is how often q appears in D, |D| is D's token count, and
        IDF(q) = ln((N - n + 0.5) / (n + 0.5) + 1)
    with N documents total and n containing q.

    Args:
        query:     The search query.
        documents: The documents to score.
        k1:        Term-frequency saturation. Higher = repeats count more.
        b:         Length normalisation, 0 to 1.

    Returns:
        One score per document, in the same order. A document containing
        none of the query terms scores exactly 0.0.

    Examples:
        >>> round(bm25_scores("cat", ["the cat sat", "the dog sat", "birds fly"])[0], 4)
        0.9808
        >>> bm25_scores("aspirin", ["the cat sat", "the dog sat"])
        [0.0, 0.0]
    """
    # TODO:
    #   1. Tokenize every document once (reuse tokenize_words).
    #   2. n_documents = len(documents); average_length = mean token count.
    #   3. Build a document-frequency Counter. Use set(tokens) so a term
    #      appearing 5 times in one document still counts as ONE document.
    #   4. For each document:
    #        - term_frequency = Counter(its tokens)
    #        - for each query term present in it, add the BM25 contribution
    #        - skip query terms the document does not contain (they add 0)
    #   5. Return the list of scores.
    return [0.0] * len(documents)


# ======================================================================
# TASK 5 - reciprocal_rank_fusion
# Module 8, section 8.6 - THE hybrid search algorithm
# ======================================================================

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> list:
    """Merge several ranked lists into one.

        RRF(d) = SUM over rankings of 1 / (k + rank_of_d_in_that_ranking)

    Ranks are 1-based. This uses ONLY rank position, never the underlying
    scores - which is exactly why it can fuse cosine similarity and BM25
    without normalising anything.

    Args:
        rankings: Each inner list holds document ids, best first.
        k:        Damping constant. 60 is the value from the original paper.

    Returns:
        Document ids sorted by fused score, best first. Every id appearing
        in any input list appears exactly once in the output.

    Examples:
        >>> reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]], k=0)
        [1, 0, 2]
        >>> reciprocal_rank_fusion([[5, 1], [5, 2]])[0]
        5
        >>> reciprocal_rank_fusion([])
        []
    """
    # TODO:
    #   1. Build a dict of {document_id: fused_score}.
    #   2. For each ranking, enumerate(ranking, start=1) to get 1-based ranks,
    #      and ADD 1/(k + rank) to that document's running score.
    #      Use .get(doc, 0.0) so the first sighting starts from zero.
    #   3. Return the ids sorted by score, DESCENDING.
    return []


# ======================================================================
# TASK 6 - build_grounded_prompt
# Module 8, section 8.9
# ======================================================================

def build_grounded_prompt(question: str, chunks: list) -> str:
    """Build a prompt that answers ONLY from the given context.

    Must produce EXACTLY this shape (the self-test compares it literally):

        Answer the question using ONLY the context below.
        If the context does not contain the answer, reply exactly: I don't know.
        Cite the sources you used as [1], [2], etc.

        CONTEXT:
        [1] first chunk
        [2] second chunk

        QUESTION: the question

        ANSWER:

    Args:
        question: The user's question.
        chunks:   Retrieved chunk texts, best first.

    Returns:
        The assembled prompt.
    """
    # TODO:
    #   1. Number the chunks 1-based: "[1] text", one per line.
    #      Hint: enumerate(chunks, start=1) and "\n".join(...)
    #   2. Assemble the full prompt exactly as shown above.
    #
    # Why 1-based numbering? It gives the model a citation vocabulary it can
    # use ("[2]") and your code can parse. Without numbers you get "the first
    # document", which you cannot resolve back to a source.
    return ""


# ======================================================================
# TASK 7 - citations
# Module 8, section 8.10
# ======================================================================

def extract_cited_indices(answer: str) -> list:
    """Pull the [n] citation markers out of an answer.

    Returns:
        Sorted, de-duplicated 1-based indices. Empty list if none found.

    Examples:
        >>> extract_cited_indices("The dose is 2000 mg [1] daily [3].")
        [1, 3]
        >>> extract_cited_indices("[2][2][1]")
        [1, 2]
        >>> extract_cited_indices("No citations here.")
        []
    """
    # TODO:
    #   Guard empty/None input, then find every [digits] group.
    #   Return them as a sorted set of ints (de-duplicated, ascending).
    # Hint: re.findall(r"\[(\d+)\]", answer)
    return []


def validate_citations(answer: str, n_chunks: int) -> tuple:
    """Check that citation markers are present and in range.

    Args:
        answer:   The model's answer.
        n_chunks: How many chunks were supplied (so valid markers are 1..n).

    Returns:
        (is_valid, problems) where problems is a list of strings.

    Two things are checked:
      - An answer that makes claims but cites NOTHING is suspicious: the
        model is likely answering from its own knowledge, not your documents.
        An "I don't know" answer is exempt.
      - A marker outside 1..n_chunks was invented by the model.

    Examples:
        >>> validate_citations("The dose is 2000 mg [1].", 3)
        (True, [])
        >>> ok, problems = validate_citations("The dose is 2000 mg.", 3)
        >>> ok, len(problems)
        (False, 1)
        >>> ok, problems = validate_citations("See [7].", 3)
        >>> ok, len(problems)
        (False, 1)
    """
    problems = []

    # TODO:
    #   1. cited = extract_cited_indices(answer)
    #   2. If nothing was cited AND "i don't know" is not in answer.lower(),
    #      append a problem describing it.
    #   3. Collect any cited index outside 1..n_chunks; if any, append ONE
    #      problem listing them.
    #   4. Return (not problems, problems).

    return (not problems, problems)


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _run_self_test() -> int:
    checks = []

    def check(name, got, expected):
        checks.append((name, got, expected))

    def check_raises(name, fn, exception=ValueError):
        try:
            fn()
            checks.append((name, "did not raise", f"raised {exception.__name__}"))
        except exception:
            checks.append((name, f"raised {exception.__name__}",
                           f"raised {exception.__name__}"))
        except Exception as exc:
            checks.append((name, f"raised {type(exc).__name__}",
                           f"raised {exception.__name__}"))

    # --- TASK 1 ---
    check("1. chunk_fixed no overlap", chunk_fixed("abcdefghij", 4, 0),
          ["abcd", "efgh", "ij"])
    check("1. chunk_fixed with overlap", chunk_fixed("abcdefghij", 4, 2),
          ["abcd", "cdef", "efgh", "ghij", "ij"])
    check("1. chunk_fixed empty text", chunk_fixed("", 4, 0), [])
    check("1. chunk_fixed text shorter than chunk", chunk_fixed("ab", 4, 0), ["ab"])
    check_raises("1. chunk_fixed rejects overlap >= chunk_size",
                 lambda: chunk_fixed("abcdef", 3, 3))
    check_raises("1. chunk_fixed rejects chunk_size <= 0",
                 lambda: chunk_fixed("abcdef", 0, 0))

    # --- TASK 2 ---
    check("2. chunk_recursive splits on paragraphs",
          chunk_recursive("aaa\n\nbbb\n\nccc", 5), ["aaa", "bbb", "ccc"])
    check("2. chunk_recursive falls through to spaces",
          chunk_recursive("hello world foo", 6), ["hello", "world", "foo"])
    check("2. chunk_recursive keeps short text whole",
          chunk_recursive("short", 10), ["short"])
    check("2. chunk_recursive empty text", chunk_recursive("", 10), [])
    check("2. chunk_recursive hard-splits when out of separators",
          chunk_recursive("aaaaaaaa", 3, ("",)), ["aaa", "aaa", "aa"])
    check("2. chunk_recursive mixes strategies",
          chunk_recursive("aa\n\nbbbbbbbbbb", 4), ["aa", "bbbb", "bbbb", "bb"])
    check("2. chunk_recursive never exceeds chunk_size",
          max((len(c) for c in chunk_recursive(
              "the quick brown fox jumps over the lazy dog again and again", 10)),
              default=0) <= 10, True)

    # --- TASK 3 ---
    check("3. tokenize_words strips punctuation",
          tokenize_words("Metformin: max 2000 mg/day!"),
          ["metformin", "max", "2000", "mg", "day"])
    check("3. tokenize_words lowercases", tokenize_words("ABC def"), ["abc", "def"])
    check("3. tokenize_words empty", tokenize_words(""), [])

    # --- TASK 4 ---
    docs = ["the cat sat", "the dog sat", "birds fly high"]
    check("4. bm25 scores the matching document",
          round(bm25_scores("cat", docs)[0], 4), 0.9808)
    check("4. bm25 gives 0 to non-matching documents",
          [round(s, 4) for s in bm25_scores("cat", docs)[1:]], [0.0, 0.0])
    check("4. bm25 all zero when no term matches",
          bm25_scores("aspirin", docs), [0.0, 0.0, 0.0])
    check("4. bm25 shared term scores equally",
          [round(s, 4) for s in bm25_scores("sat", docs)], [0.47, 0.47, 0.0])

    medical = [
        "Metformin maximum daily dose is 2000 mg",
        "Metformin is contraindicated in renal failure",
        "Insulin therapy requires glucose monitoring",
        "The maximum dose of metformin metformin metformin",
    ]
    scores = bm25_scores("metformin dose", medical)
    check("4. bm25 ranks repeated terms highest",
          scores.index(max(scores)), 3)
    check("4. bm25 finds the only insulin document",
          bm25_scores("insulin", medical).index(max(bm25_scores("insulin", medical))), 2)

    # --- TASK 5 ---
    check("5. rrf fuses two rankings (k=0 for clean values)",
          reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]], k=0), [1, 0, 2])
    check("5. rrf promotes a doc ranked well in BOTH",
          reciprocal_rank_fusion([[5, 1], [5, 2]])[0], 5)
    check("5. rrf includes every id exactly once",
          sorted(reciprocal_rank_fusion([[5, 1], [5, 2]])), [1, 2, 5])
    check("5. rrf single ranking", reciprocal_rank_fusion([[0]]), [0])
    check("5. rrf empty input", reciprocal_rank_fusion([]), [])
    check("5. rrf handles rankings of different lengths",
          len(reciprocal_rank_fusion([[1, 2, 3], [4]])), 4)

    # --- TASK 6 ---
    expected_prompt = (
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, reply exactly: I don't know.\n"
        "Cite the sources you used as [1], [2], etc.\n\n"
        "CONTEXT:\n"
        "[1] one\n"
        "[2] two\n\n"
        "QUESTION: Q?\n\n"
        "ANSWER:"
    )
    check("6. build_grounded_prompt exact format",
          build_grounded_prompt("Q?", ["one", "two"]), expected_prompt)
    check("6. build_grounded_prompt numbers from 1",
          "[1] first" in build_grounded_prompt("Q?", ["first"]), True)

    # --- TASK 7 ---
    check("7. extract_cited_indices finds markers",
          extract_cited_indices("The dose is 2000 mg [1] daily [3]."), [1, 3])
    check("7. extract_cited_indices de-duplicates and sorts",
          extract_cited_indices("[2][2][1]"), [1, 2])
    check("7. extract_cited_indices none found",
          extract_cited_indices("No citations here."), [])
    check("7. extract_cited_indices empty input", extract_cited_indices(""), [])
    check("7. extract_cited_indices multi-digit",
          extract_cited_indices("[10] and [2]"), [2, 10])

    check("7. validate_citations accepts a cited answer",
          validate_citations("The dose is 2000 mg [1].", 3), (True, []))

    ok, problems = validate_citations("The dose is 2000 mg.", 3)
    check("7. validate_citations flags an uncited claim", (ok, len(problems)), (False, 1))

    ok, problems = validate_citations("I don't know.", 3)
    check("7. validate_citations exempts 'I don't know'", ok, True)

    ok, problems = validate_citations("See [7].", 3)
    check("7. validate_citations flags an invented marker",
          (ok, len(problems)), (False, 1))

    ok, problems = validate_citations("See [1] and [9].", 3)
    check("7. validate_citations flags only the out-of-range marker",
          (ok, len(problems)), (False, 1))

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 8 SELF-TEST - the RAG toolkit")
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
        print(f"  All {len(checks)} checks passed.")
        print("  You now have every piece of a RAG pipeline except the model call.")
        print("  Part 2 in the lab brief assembles them over your own documents.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1, 2, then 3 (needed by 4), 4, 5, 6, 7.")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
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


def experiment_chunking():
    print("=" * 76)
    print("  EXPERIMENT 1: fixed vs recursive chunking")
    print("=" * 76)
    print()

    document = (
        "Metformin Dosing Guidance\n\n"
        "The maximum daily dose is 2000 mg for immediate-release tablets. "
        "Higher doses require monitoring.\n\n"
        "Contraindications include severe renal impairment."
    )

    for label, chunks in [
        ("FIXED (size=60, no overlap)", chunk_fixed(document, 60, 0)),
        ("FIXED (size=60, overlap=15)", chunk_fixed(document, 60, 15)),
        ("RECURSIVE (size=60)", chunk_recursive(document, 60)),
    ]:
        if not chunks:
            print("  chunking not implemented yet.")
            print()
            return
        print(f"  {label}  ->  {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk.replace("\n", "\\n")
            print(f"    {i}. {preview!r}")
        print()

    print("  Look at where FIXED cuts. It splits mid-sentence and mid-word,")
    print("  so a chunk can contain half an idea and answer nothing.")
    print()
    print("  RECURSIVE splits on paragraph and word boundaries instead, so each")
    print("  chunk is a readable unit. That is why it is the default.")
    print()


def experiment_hybrid_search():
    """Show what BM25 alone gets right and wrong - the case for hybrid search."""
    print("=" * 76)
    print("  EXPERIMENT 2: what keyword search alone can and cannot do")
    print("=" * 76)
    print()
    print("  This experiment uses BM25 only. The semantic half needs real")
    print("  embeddings, which Part 2 of the lab brief supplies - so here we")
    print("  establish HALF the argument honestly, rather than faking the other.")
    print()

    # (label, query, the text the CORRECT chunk must contain)
    queries = [
        ("exact identifier", "SKU-4471", "SKU-4471"),
        ("exact identifier", "E1042", "E1042"),
        ("shares key terms", "maximum daily dose of metformin", "maximum daily dose"),
        ("partial overlap", "what is the highest amount of metformin per day",
         "maximum daily dose"),
        ("pure paraphrase", "how much can a patient take each day",
         "maximum daily dose"),
    ]

    print(f"  {'query type':<20}{'correct chunk on top?':<24}{'score':>8}")
    print("  " + "-" * 54)

    for label, query, expected in queries:
        scores = bm25_scores(query, SAMPLE_CHUNKS)
        if not scores or max(scores) == 0:
            print(f"  {label:<20}{'NO - nothing matched':<24}{0.0:>8.2f}")
            continue
        best = scores.index(max(scores))
        correct = expected in SAMPLE_CHUNKS[best]
        verdict = "yes" if correct else "NO - wrong chunk"
        print(f"  {label:<20}{verdict:<24}{scores[best]:>8.2f}")

    print()
    print("  Read the last row. 'how much can a patient take each day' shares no")
    print("  distinctive words with 'maximum daily dose', so BM25 has nothing to")
    print("  match on and returns the wrong chunk with a weak score.")
    print()
    print("  Now read the first two rows. On exact identifiers BM25 is precise")
    print("  and confident - and these are exactly the queries where semantic")
    print("  similarity fails, as you saw in Lab 3 with 'SKU-4471'.")
    print()
    print("  Neither retriever covers both. That is the entire argument for")
    print("  hybrid search, and reciprocal_rank_fusion is how you combine them")
    print("  without having to reconcile their incompatible score scales.")
    print()


def experiment_prompt_and_citations():
    print("=" * 76)
    print("  EXPERIMENT 3: grounded prompts and citation validation")
    print("=" * 76)
    print()

    question = "What is the maximum daily dose of metformin?"
    retrieved = [SAMPLE_CHUNKS[1], SAMPLE_CHUNKS[2], SAMPLE_CHUNKS[3]]

    prompt = build_grounded_prompt(question, retrieved)
    if not prompt:
        print("  build_grounded_prompt not implemented yet.")
        print()
        return

    print("  The prompt sent to the model:")
    print("  " + "-" * 72)
    for line in prompt.splitlines():
        print(f"  | {line}")
    print("  " + "-" * 72)
    print()

    # Answers a model might plausibly return.
    candidate_answers = [
        ("well-grounded",
         "The maximum daily dose is 2000 mg for immediate-release, or up to "
         "2550 mg with monitoring [1]. Extended-release is capped at 2000 mg [2]."),
        ("no citations (answering from memory?)",
         "The maximum daily dose of metformin is 2000 mg."),
        ("invented a source",
         "The dose is 2000 mg [1], and it interacts with warfarin [7]."),
        ("correct refusal",
         "I don't know."),
    ]

    print(f"  {'answer type':<38}{'valid':>7}   problems")
    print("  " + "-" * 72)
    for label, answer in candidate_answers:
        ok, problems = validate_citations(answer, len(retrieved))
        print(f"  {label:<38}{str(ok):>7}   {'; '.join(problems) or '-'}")
    print()
    print("  The second row is the important one. An answer that makes claims")
    print("  and cites nothing usually means the model ignored your context and")
    print("  answered from training knowledge - the exact failure RAG exists to")
    print("  prevent. It costs one regex to detect.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_chunking()
        experiment_hybrid_search()
        experiment_prompt_and_citations()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
