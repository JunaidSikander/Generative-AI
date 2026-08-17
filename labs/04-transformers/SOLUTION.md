# Lab 4 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `softmax`

```python
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)
```

Three lines, and two of them are subtle.

### Why subtract the max

Mathematically it's a no-op:

$$\frac{e^{a_i - c}}{\sum_j e^{a_j - c}} = \frac{e^{a_i} \cdot e^{-c}}{e^{-c}\sum_j e^{a_j}} = \frac{e^{a_i}}{\sum_j e^{a_j}}$$

The `e^{-c}` cancels. So why bother?

```python
np.exp(1000)          # inf  -> overflow
np.exp(1000 - 1000)   # 1.0  -> fine
```

Attention scores can be large, especially before scaling. Without the shift you get `inf / inf` = `nan`, and `nan` propagates through everything silently. **The self-test checks `softmax([1000, 1000])` returns `[0.5, 0.5]`** — it fails with `nan` if you skip the shift.

After shifting, the largest value is always exactly 0, so `exp` never exceeds 1. This is the standard numerically-stable softmax, and you'll see it in every real implementation.

### Why `axis=-1` and `keepdims=True`

`axis=-1` means "the last axis" — across each row.

```python
scores = np.array([[1.0, 2.0],
                   [3.0, 4.0]])

softmax(scores)                    # axis=-1: each ROW sums to 1  ✅
# [[0.269, 0.731],
#  [0.269, 0.731]]
```

Get the axis wrong and you normalise down columns instead — so instead of "how much does token *i* attend to each other token", you'd compute something meaningless about competition between tokens.

**The dangerous part: the shape is still right.** No exception, no warning, plausible-looking numbers, completely wrong model. The self-test has a dedicated check (`softmax per-ROW not per-column`) using an asymmetric input, because a symmetric test input would pass either way.

`keepdims=True` preserves the dimension so broadcasting works:

```python
np.max(x, axis=-1)                    # shape (2,)   -> broadcasting misaligns
np.max(x, axis=-1, keepdims=True)     # shape (2, 1) -> divides row-wise  ✅
```

Drop `keepdims` and you'll get either a broadcasting error or, worse, silently transposed arithmetic.

---

## Task 2 — `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Step 1 — `Q @ K.T`

`(n_tokens, d_k) @ (d_k, n_tokens)` → `(n_tokens, n_tokens)`.

`scores[i][j]` is how much token *i* cares about token *j*. Because a dot product is large when vectors point the same way (Module 3 §3.6), this measures relevance.

**This is where O(n²) comes from,** and it's worth making concrete:

```python
for n in [100, 1_000, 10_000]:
    print(f"n={n:>6}: {(n * n * 8) / 1e6:>10.1f} MB")
# n=   100:        0.1 MB
# n=  1000:        8.0 MB
# n= 10000:      800.0 MB
```

At 10,000 tokens, one attention matrix is ~800 MB — **and there's one per head, per layer**. With 32 heads and 32 layers you're deep into terabyte territory if you materialise them all.

You've just derived, with arithmetic, why long context windows are hard and expensive, and why FlashAttention (which never materialises the full matrix) and sliding-window attention exist.

### Step 2 — `/ np.sqrt(d_k)`

Run `solution.py` and read the scaling demo:

```
   d_k    max |score| unscaled   max weight unscaled    scaled
       4                    9.96                0.7125    0.4867
      64                   19.47                0.8423    0.3006
     512                   49.43                1.0000    0.5861
```

As `d_k` grows, the raw dot products grow with it (roughly as `√d_k` for random vectors). Feed those into softmax and it **saturates**: at `d_k=512` the top weight is `1.0000`. The "weighted blend" has become a hard pick.

Two problems with that:

1. **No blending.** Attention's whole value is combining information from several tokens.
2. **Vanishing gradients.** Softmax is nearly flat when saturated, so almost no learning signal flows back and training stalls.

Dividing by `√d_k` keeps score variance roughly constant regardless of dimension. **One division, and it's the difference between a model that trains and one that doesn't.**

### Step 3 — the mask, added *before* softmax

Order matters enormously here. Masking must happen before softmax so `exp(-inf) = 0` removes those positions *and* the surviving weights renormalise to sum to 1. Mask after softmax and you'd have rows that no longer sum to 1.

### Step 4/5 — softmax, then blend

`weights @ V` — each token's output is a weighted average of every token's value vector.

The `uniform attention -> mean of V` check makes this concrete: with all-zero Q and K, every score is 0, so every weight is `1/3`, so the output is exactly the mean of V's rows. Attention *is* a weighted average; uniform weights give you the plain average.

### The shape detail

Output shape follows `V`, not `Q`:

```python
Q: (5, 8)    K: (5, 8)    V: (5, 3)    ->    output: (5, 3)
```

`Q` and `K` must share `d_k` (they're dot-producted together). `V` can have its own `d_v`. In practice `d_k == d_v`, but keeping them distinct in your head prevents shape confusion later.

---

## Task 3 — `causal_mask`

```python
def causal_mask(n_tokens):
    future = np.triu(np.ones((n_tokens, n_tokens)), k=1)
    return np.where(future == 1, -np.inf, 0.0)
```

`np.triu(..., k=1)` selects the strictly-upper triangle. `k=1` means "start one above the diagonal" — the diagonal itself (a token attending to *itself*) must stay allowed.

```python
causal_mask(3)
# [[  0., -inf, -inf],
#  [  0.,   0., -inf],
#  [  0.,   0.,   0.]]
```

### Why `-inf` and not `0`

This is the most important detail in the task.

The mask is **added to the scores before softmax**:

| Mask value | After `exp()` | Effect |
|---|---|---|
| `-inf` | `exp(-inf)` = **0** | ✅ Position completely removed |
| `0` | `exp(0)` = **1** | ❌ A substantial weight — the future leaks |

Using `0` produces a model that scores brilliantly during training — because it can see the answers — and fails completely at inference, when the future doesn't exist. **A silent, catastrophic bug**, and one of the classic mistakes when implementing a transformer by hand.

### What the checks prove

The three combined checks are worth understanding:

- **`first token attends only to itself`** → `[1.0, 0, 0, 0]`. Token 0 has no past, so all its weight must go to itself. If your mask is off by one, this fails.
- **`no attention leaks to the future`** → `np.triu(weights, k=1).sum() == 0.0`. Exactly zero, not merely small.
- **`causal rows still sum to 1`** → masking removes positions without breaking normalisation. This is what would fail if you masked *after* softmax.

---

## Task 4 — `multi_head_attention`

```python
def multi_head_attention(X, W_Q_list, W_K_list, W_V_list, W_O):
    head_outputs = []
    for W_Q, W_K, W_V in zip(W_Q_list, W_K_list, W_V_list):
        Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        output, _ = scaled_dot_product_attention(Q, K, V)
        head_outputs.append(output)

    combined = np.concatenate(head_outputs, axis=-1)
    return combined @ W_O
```

**Every head sees the same `X`** but projects it through its own weights, so each ends up examining a different kind of relationship.

### Why concatenate *then* project

Concatenation just stacks the heads' findings side by side — `(n, d_head)` × 4 heads → `(n, 4 * d_head)`. That's a pile of separate opinions with no interaction.

`W_O` is what lets the model **mix** them, learning how much each head's perspective should influence the result. Without it you'd have four independent analyses and no synthesis.

### The dimension arithmetic

With `d_model = 8` and `n_heads = 2`, each head works in `d_head = 4` dimensions. `d_model` is **divided** among heads, not multiplied:

```
d_model = 512, n_heads = 8   ->   d_head = 64
```

**This is why multi-head attention costs roughly the same as single-head.** You get eight perspectives for the price of one by splitting the space rather than enlarging it. It also explains why "more heads" isn't automatically better — past a point each head has too few dimensions to represent anything useful, and studies have found many heads can be pruned with negligible loss.

The `actually transforms X` check exists to catch the stub-returning-input case.

---

## The experiments — discussion

### Experiment 1 — one token, two contexts

The output vector for `"bit"` differs between sentences, with a cosine similarity near zero. **The input vector was identical.** Only the neighbours changed.

This is the direct resolution of Module 3's cliffhanger. Static embeddings (word2vec, GloVe) give `"bit"` one vector forever and cannot distinguish the verb from the quantity. Attention produces a *different vector per occurrence*.

**The caveat matters, and the lab says so in its own output.** These weight matrices are random. The two vectors differ, but not *meaningfully* — nothing has taught this toy model that *dog* and *man* imply an action. In a trained model, `W_Q`, `W_K` and `W_V` have been shaped by billions of examples so that the difference tracks actual semantics.

**What you're seeing is the mechanism, not the meaning.** That distinction is worth holding onto, because it's the honest version of what "attention" does: it provides the *machinery* for context-dependence. Training supplies the content.

### Experiment 2 — causal masking

Look at the first token's row in the masked version: `1.00, 0.00, 0.00, 0.00`. With nothing but itself to attend to, softmax must give it all the weight.

The rest of the upper-right triangle is exact zeros. Compare the unmasked version, where every token draws on the whole sentence.

That difference is the whole architectural distinction between the two families (Module 4 §4.8):

| | Unmasked (bidirectional) | Masked (causal) |
|---|---|---|
| Family | Encoder-only — BERT | Decoder-only — GPT |
| Each token sees | The entire sequence | Itself and the past |
| Trained by | Filling in masked blanks | Predicting the next token |
| Good at | Understanding, **embeddings** | **Generating text** |

And this is why you used an *encoder* model (`all-MiniLM-L6-v2`) for embeddings in Lab 3 and call a *decoder* model for chat. Different masks, different jobs.

---

## 🚀 Stretch — Discussion

### 2. Removing residual connections

If you tried it: stacking ten blocks without `X +` makes activations explode or collapse toward zero, and the output stops depending meaningfully on the input.

**Residuals are what make depth possible.** Two reasons:

1. Each block learns a *refinement* to the representation rather than rebuilding it from scratch
2. Gradients flow straight back through the `+`, so early layers still receive signal

Before residual connections (ResNet, 2015), networks beyond ~20 layers were essentially untrainable. Modern LLMs stack 100+.

### 3. Positional encoding

The two sentences — `"The dog bit the man"` and `"The man bit the dog"` — contain **identical multisets of tokens**. Without positional encoding, attention produces the same set of output vectors, merely reordered, so the two sentences are indistinguishable in aggregate.

Add positional encoding and the vectors genuinely differ.

This is the cleanest demonstration that **attention is order-blind on its own**. Look back at the formula: it's dot products and a weighted sum. Nothing in it refers to position. Order has to be injected, either by adding it to the input (sinusoidal, learned) or by rotating Q and K inside each layer (RoPE, which is what modern models use).

### 4. The memory arithmetic

800 MB for one attention matrix at 10,000 tokens, per head, per layer.

This single calculation explains a lot of modern LLM engineering:

- **FlashAttention** computes attention in tiles without ever materialising the full matrix
- **Sliding-window attention** restricts each token to a local neighbourhood, making cost linear
- **Multi-query / grouped-query attention** shares keys and values across heads to cut memory
- **KV caching** stores past keys and values during generation so you don't recompute them every step

All of them are attacks on the same O(n²) wall you just measured.

### 5. Real attention maps

The honest finding: **some heads are interpretable, many aren't.**

You'll likely see a head that tracks subject–verb relationships and one that links determiners to their nouns. You'll also see heads that dump most of their weight on the first token or on punctuation — these act as approximate no-ops, a documented phenomenon sometimes called "attention sinks".

This is why treating attention maps as *explanations* of model reasoning is a well-known interpretability trap. They show where information flowed. They don't show why, and they aren't a faithful account of the model's decision.

---

## Ready for Module 5?

- [ ] You can write the attention formula from memory and say what each term does
- [ ] You can explain why we subtract the max in softmax, and why `axis=-1` matters
- [ ] You can explain why we divide by `√d_k`
- [ ] You can explain why the mask uses `-inf` and why it's applied before softmax
- [ ] You can say where O(n²) comes from and what it costs in megabytes
- [ ] You can explain why encoder models are used for embeddings and decoder models for chat
- [ ] You can state the difference between fine-tuning and RAG in one sentence

**Next: Module 5 — Prompt Engineering.** Back to practical work — but everything there rests on what you now know about attention, position and the context window. When Module 5 says placement within a prompt matters, you'll know precisely why.

---

<div align="center">

**[⬅ Back to Lab 4](README.md)** · **[📖 Module 4](../../modules/04-transformers.md)** · **[🏠 README](../../README.md)**

</div>
