# Lab 8 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `chunk_fixed`

```python
if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")
if overlap >= chunk_size:
    raise ValueError("overlap must be smaller than chunk_size")
if not text:
    return []

step = chunk_size - overlap
return [text[i:i + chunk_size] for i in range(0, len(text), step)]
```

### Why the overlap guard is a real bug, not defensive padding

```python
step = chunk_size - overlap     # overlap=4, chunk_size=4  ->  step = 0
range(0, len(text), 0)          # ValueError: range() arg 3 must not be zero
```

With `range` you get an exception, which is survivable. Write the same loop by hand — `while i < len(text): ... ; i += step` — and it **never advances**. Your indexing job appends the same chunk forever until memory runs out.

There's a test expecting `ValueError`, because a config typo (`overlap: 500, chunk_size: 500`) should not be able to hang a pipeline. **Guard the inputs that can produce non-terminating behaviour**, not just the ones that can produce wrong answers.

### Reading the overlap output

```python
chunk_fixed("abcdefghij", 4, 2)  ->  ['abcd', 'cdef', 'efgh', 'ghij', 'ij']
```

`step = 2`, so each chunk repeats the last 2 characters of the previous one. Note the final chunk is short — that's expected and fine; a real pipeline may drop chunks below a minimum length.

---

## Task 2 — `chunk_recursive`

```python
if not text:
    return []
if len(text) <= chunk_size:
    return [text]

if not separators or separators[0] == "":
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

separator, remaining = separators[0], separators[1:]

chunks = []
for piece in text.split(separator):
    if not piece:
        continue
    if len(piece) <= chunk_size:
        chunks.append(piece)
    else:
        chunks.extend(chunk_recursive(piece, chunk_size, remaining))
return chunks
```

### The `""` trap

```python
"hello".split("")     # ValueError: empty separator
```

The empty string in the separator list means "give up on boundaries and hard-split". It must be handled **before** any `.split()` call. Put the check after the split and every document hitting the last separator crashes.

### Why the recursion terminates

Each recursive call passes `remaining` — a strictly shorter separator tuple. Eventually either a piece fits (base case) or the separators run out and the hard-split branch fires. **No path recurses with the same arguments**, which is what you check when convincing yourself a recursive function is safe.

### What this version deliberately omits

Real implementations (LangChain's `RecursiveCharacterTextSplitter`) also **merge** small adjacent pieces back up toward `chunk_size`. Without merging, this:

```python
chunk_recursive("aaa\n\nbbb\n\nccc", 5)  ->  ['aaa', 'bbb', 'ccc']
```

produces three tiny chunks where one chunk of `"aaa\n\nbbb\n\nccc"` would have been better — it's only 13 characters.

**Why leave it out?** Merging roughly doubles the algorithm's length and obscures the recursive idea, which is the thing worth learning. But know that it matters in practice: unmerged recursive splitting on a document with many short paragraphs gives you a pile of near-useless single-line chunks.

Adding merging is a good exercise: after building the piece list, greedily concatenate adjacent pieces (with the separator between them) while the result stays under `chunk_size`.

---

## Task 3 — `tokenize_words`

```python
return re.findall(r"[a-z0-9]+", text.lower())
```

Lowercase **first**, so the pattern only needs the lowercase range. `"Metformin: max 2000 mg/day!"` → `['metformin', 'max', '2000', 'mg', 'day']`.

**What this throws away, and why that's acceptable here:** no stemming (so `"dose"` and `"dosing"` are different terms), no stop-word removal, and hyphenated terms split (`"SKU-4471"` → `['sku', '4471']`).

That last one is interesting: BM25 still finds the SKU chunk, because `'4471'` is a rare token with high IDF. **The hyphen doesn't matter as much as you'd expect**, because rarity does the work.

Real search systems add stemming and language-specific handling. For learning BM25, the crude version keeps the focus on the formula.

---

## Task 4 — `bm25_scores`

### The `set()` that matters

```python
document_frequency = Counter()
for tokens in doc_tokens:
    document_frequency.update(set(tokens))     # set() is load-bearing
```

Document frequency counts **how many documents contain a term**, not how many times it appears. Drop the `set()` and a term appearing five times in one document counts as five documents — so its IDF collapses and common terms stop being penalised.

The bug is silent: scores are still produced, still ordered, just wrong in a way you'd never notice without a reference implementation.

### Why the formula has the shape it does

```python
numerator = frequency * (k1 + 1)
denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
```

**The `k1` part saturates.** As `frequency` grows, the ratio approaches `k1 + 1` and stops climbing. A document mentioning your term 50 times isn't 50× more relevant than one mentioning it once — which naive TF-IDF wrongly implies.

You can see it in the test data:

```
"The maximum dose of metformin metformin metformin"   -> 1.2348
"Metformin maximum daily dose is 2000 mg"             -> 0.9960
```

Three mentions of `metformin` beats one — but only by ~24%, not 3×.

**The `b` part normalises length.** `b=0.75` means mostly-normalise: a match in a short document counts for more, because there's less chance it's incidental.

### The clean signal you get for free

A document containing none of the query terms scores exactly `0.0` — the loop skips every term. Test: `bm25_scores("aspirin", docs)` → `[0.0, 0.0, 0.0]`.

That's a genuinely useful signal: **`max(scores) == 0` means keyword search has nothing to offer for this query**, which is exactly when you'd lean on the dense retriever.

---

## Task 5 — `reciprocal_rank_fusion`

```python
fused = {}
for ranking in rankings:
    for position, document_id in enumerate(ranking, start=1):
        fused[document_id] = fused.get(document_id, 0.0) + 1.0 / (k + position)
return sorted(fused, key=lambda document_id: -fused[document_id])
```

Ten lines, and it's the algorithm that makes hybrid search practical.

### Why it beats adding scores — with numbers

`solution.py`'s third demo makes this concrete. Two retrievers on different scales:

```
  dense (cosine, 0-1):     {0: 0.81, 1: 0.79, 2: 0.77, 3: 0.4}
  sparse (BM25, unbounded): {3: 14.2, 2: 11.9, 0: 0.6, 1: 0.0}

  NAIVE score addition -> [3, 2, 0, 1]
  RRF fusion           -> [2, 0, 3, 1]
```

**Naive addition hands the decision entirely to BM25.** Its 14.2 swamps every cosine value, so document 3 — which the dense retriever ranked *last* — comes first. The dense retriever effectively got no vote.

RRF ignores the magnitudes. Document 2 wins because it ranked well in **both** (3rd dense, 2nd sparse), which is exactly the signal you want from an ensemble.

### Why `start=1`

```python
enumerate(ranking, start=1)
```

With `start=0`, the top result scores `1/(k+0)`. When `k=0` that's a division by zero; when `k=60` it over-weights rank 1 relative to rank 2 more than intended. Ranks are 1-based in the original paper, and the arithmetic depends on it.

### What `k` does

`k=60` (the paper's value) damps the difference between top ranks:

| Rank | `1/(60+rank)` | `1/(0+rank)` |
|---|---|---|
| 1 | 0.0164 | 1.000 |
| 2 | 0.0161 | 0.500 |
| 3 | 0.0159 | 0.333 |

With `k=60`, ranks 1 and 2 are nearly equivalent — so being #1 in one retriever doesn't automatically beat being #2 in both. With `k=0`, rank 1 dominates completely.

**Larger `k` = more consensus-driven. Smaller `k` = more winner-takes-all.** The tests use `k=0` for exactly computable values and `k=60` for default behaviour.

---

## Task 6 — `build_grounded_prompt`

```python
context = "\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))

return (
    "Answer the question using ONLY the context below.\n"
    "If the context does not contain the answer, reply exactly: I don't know.\n"
    "Cite the sources you used as [1], [2], etc.\n\n"
    f"CONTEXT:\n{context}\n\n"
    f"QUESTION: {question}\n\n"
    "ANSWER:"
)
```

Four choices, each doing real work:

**1. "ONLY the context."** Without it the model blends retrieved text with training knowledge, and you lose the ability to verify anything. The answer might be right — you just can't tell why.

**2. `"reply exactly: I don't know"`** — Module 5 §5.4. "Don't hallucinate" is unactionable; the model has no internal "am I making this up?" signal. This is a *behaviour it can perform*, and because the string is exact, **your code can detect it**.

**3. Numbered chunks.** `[1]`, `[2]` give the model a citation vocabulary. Without numbers you get "according to the first document", which you cannot parse back to a source.

**4. Question last.** Module 3 §3.9: recall is strongest at the start and end of a context. Putting the question immediately before `ANSWER:` places it in a high-attention position at the moment generation begins.

The test compares the string **byte for byte**. That's not pedantry — a prompt is the specification of your application's behaviour, and one that varies unpredictably between calls produces output that varies unpredictably.

---

## Task 7 — Citations

```python
def extract_cited_indices(answer):
    if not answer:
        return []
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
```

A set de-duplicates (`"[2][2][1]"` → `[1, 2]`); `sorted()` gives a stable order. The `(\d+)` capture group means `findall` returns just the digits, not the brackets.

### `validate_citations` — the two checks

```python
if not cited and "i don't know" not in answer.lower():
    problems.append("answer makes claims but cites no sources")

out_of_range = [i for i in cited if not 1 <= i <= n_chunks]
```

**Check 1 is the valuable one.** An answer that makes claims and cites nothing usually means **the model ignored your context and answered from training knowledge** — the exact failure RAG exists to prevent.

And it's cheap. One regex, one condition, and you catch a failure that's otherwise invisible: the answer looks fluent and confident, and might even be correct, but it isn't *grounded* — so you have no basis for trusting it and no source to check.

The `"i don't know"` exemption matters: a legitimate refusal has nothing to cite.

**Check 2** catches invented markers. Models do cite `[7]` when given five chunks — a small, real hallucination that would otherwise resolve to nothing or crash your source-attachment code.

### Why `attach_sources` skips invalid markers rather than raising

```python
if not 1 <= index <= len(chunks):
    continue
```

`validate_citations` already reported the problem. A partly-attributed answer with a warning is more useful to a user than an exception — **degrade, don't crash**, when the failure is expected rather than exceptional.

---

## The experiments — discussion

### Experiment 1: chunking

Fixed-size splits mid-word. Recursive splits on paragraph and word boundaries. With `chunk_size=60` on a document with paragraph breaks, recursive produces readable units and fixed produces fragments like `"maximum daily dose is 2000 mg for immediate-release tabl"`.

**Why that fragment is a problem:** it might still be retrieved, and it might still contain the answer — but the chunk boundary destroyed information (`"tablets"` → `"tabl"`), and a chunk that ends mid-word reads as broken to whoever checks the citation.

### Experiment 2: a deliberately half-finished argument

```
  query type          correct chunk on top?      score
  exact identifier    yes                         3.67
  exact identifier    yes                         1.90
  shares key terms    yes                         3.94
  partial overlap     yes                         2.34
  pure paraphrase     NO - wrong chunk            1.00
```

BM25 succeeds on four of five and fails on the pure paraphrase — `"how much can a patient take each day"` shares no distinctive terms with `"maximum daily dose"`, so there's nothing to match on. Note the score of `1.00` versus 2.34–3.94 for the successes: **the low score is itself a signal** that keyword matching found nothing solid.

> **A note on how this experiment was built.** An earlier version included a "pseudo-semantic" ranker to show both halves. It scored by word overlap — which is keyword matching under a different name — so it *also* nailed `SKU-4471`, and the demo appeared to show semantic search succeeding where it actually fails.
>
> That would have been a rigged demonstration: making the point by constructing data that produces the desired output. It was removed. **Half an honest demonstration beats a whole dishonest one**, and Part 2 supplies the semantic half with real embeddings.

### Experiment 3: citation validation

Four plausible answers, and only the validator distinguishes them:

| Answer | Verdict | Why it matters |
|---|---|---|
| Well-grounded with `[1]` `[2]` | ✅ valid | The good case |
| Correct facts, **no citations** | ❌ | Probably answered from memory — unverifiable |
| Cites `[7]` of 3 chunks | ❌ | Invented a source |
| `"I don't know."` | ✅ valid | Correct refusal, legitimately uncited |

**Row 2 is the one to internalise.** The answer is *factually correct*. It's still a failure, because you have no way to verify it and no reason to believe the next one will be right. Correctness and groundedness are different properties.

---

## Part 2 — Discussion

### Why Step 3 ("make it fail") is the real work

A RAG demo works on the first question you try, because you chose a question you knew the documents answered. Finding the failures is where the engineering is:

| Failure you'll probably find | Likely cause | Fix |
|---|---|---|
| Says "I don't know" about something covered | Retrieval miss | Larger `top_k`, smaller chunks, re-ranking |
| Confident answer from the wrong section | Retrieval found plausible-but-wrong chunks | Re-ranking (§8.7) |
| Can't find a specific number or code | Dense-only behaviour dominating | Check BM25 is contributing |
| Answer split across two chunks, gets neither | Chunk boundary | Overlap, or parent-document retrieval |
| Empty or garbled extraction | **Loader problem** | Check the extracted text (§8.3) |

That last row is why `rag.py` prints the first 300 characters before indexing. **A scanned PDF yields empty text and a silently useless index** — and it's the first thing to rule out when a RAG system seems inexplicably bad.

### Why `evaluate.py` measures retrieval only

This is Module 8 §8.11's central argument. If you only measure end to end, a bad answer tells you nothing about *where* it went wrong — and retrieval failures and generation failures need opposite fixes.

`evaluate.py` asks one question: **was the right chunk in the top-k?** If no, stop tuning your prompt — nothing in the prompt can recover a chunk that was never retrieved.

`must_contain` is a deliberately crude oracle. It's a substring check, so it can't tell you whether the chunk *answers* the question, only that the key text was present. That's fine: it's cheap, deterministic, needs no LLM, and catches the failure mode that matters most. Module 11 replaces it with proper metrics.

### The tuning table

The value isn't in any single number — it's in seeing that **the parameters interact**:

- Smaller chunks improve precision but split answers across boundaries
- Larger `top_k` improves recall but dilutes the context and costs tokens
- Dense-only fails on identifiers; keyword-only fails on paraphrase
- Larger `fetch_k` gives fusion more to work with, up to a point

**Two rules for this table:** change one thing at a time, and write the numbers down. Four simultaneous changes that "seem better" teach you nothing about which helped — and you can't undo the one that hurt.

---

## 🚀 Stretch — Discussion

### 1. The cross-encoder re-ranker

Module 8 §8.7 claims this is often the highest-return single improvement. **Verify it on your data rather than believing it.**

Expect a meaningful recall improvement at `top_k=4`, because the cross-encoder reads query and chunk *together* and can tell that this specific chunk answers this specific question — something no independent embedding can do.

The counter-intuitive part: it lets you **retrieve more and send less**. Fetch 20, re-rank, send 4. Better context in fewer tokens, so quality goes up and cost goes down together.

### 2. Parent-document retrieval

This *resolves* the chunk-size trade-off rather than compromising on it — small chunks for precise matching, large parents for context. Often the single best structural change available.

One implementation detail: **de-duplicate the parents.** Five matching small chunks may share one parent, and sending it five times wastes your entire context budget on repetition.

### 3. Heading context

Cheap, and the gains are real. A chunk reading `"Refunds are processed within 14 days"` doesn't contain the word "billing" — so a question about "the billing refund policy" may miss it. Prefix `"Billing > Refunds"` and it matches.

### 5. Conflict detection

Genuinely hard, and most RAG systems fail it silently. Two contradictory chunks retrieved together produce a confident answer that picks one, with no indication a conflict existed.

The `"if sources conflict, say so"` rule helps and isn't reliable. Real approaches add document dates to metadata and prefer recent sources, or detect contradiction as a separate step. **Worth knowing that this is an open problem rather than something you've configured wrong.**

---

## Ready for Module 9?

- [ ] You can explain why retrieval quality is the ceiling on a RAG system
- [ ] You can state the difference between fine-tuning and RAG in one sentence
- [ ] You know why `overlap >= chunk_size` must raise
- [ ] You can explain why RRF uses ranks rather than scores
- [ ] You know why semantic search fails on `SKU-4471` and BM25 fails on paraphrase
- [ ] You can explain why an uncited answer is a red flag even when it's correct
- [ ] You know why retrieval and generation must be evaluated separately
- [ ] **You have a working bot and a retrieval recall number for your own documents**

That last box is the milestone. If it's ticked, you've built something real.

**Next: Module 9 — AI Agents & Tool Use.** Your bot answers from a fixed corpus. An agent *decides what to do* — search, calculate, call an API — and chains those decisions. The capability jump is large, and so is the risk, which is why Module 9 spends as much time on guardrails as on capability.

---

<div align="center">

**[⬅ Back to Lab 8](README.md)** · **[📖 Module 8](../../modules/08-rag.md)** · **[🏠 README](../../README.md)**

</div>
