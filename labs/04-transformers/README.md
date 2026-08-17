# 🧪 Lab 4: Build Attention From Scratch

**Module:** [4 — Transformers & Model Architecture](../../modules/04-transformers.md)

---

## Objective

Implement the computation that runs inside every LLM in this course — in pure NumPy, in about 20 lines — then watch it do the thing Module 3 said was impossible with static embeddings.

By the end you will have:

1. **Implemented** softmax, scaled dot-product attention, causal masking and multi-head attention
2. **Proved** why the `√d_k` scaling exists, by measuring what happens without it
3. **Watched** the token `"bit"` produce two different vectors in two different sentences
4. **Seen** causal masking zero out the future, and understood why GPT and BERT differ

## Expected outcome

`python starter.py` reports **17 of 17 checks passing**, then runs two experiments: an attention-weight breakdown for `"bit"` in both sentences, and a side-by-side comparison of masked versus unmasked attention patterns.

## Requirements

| | |
|---|---|
| **Packages** | `numpy`. That's all. |
| **API key** | **None** |
| **Cost** | **Free** — no API calls whatsoever |
| **Time** | ~45 minutes |

```powershell
pip install numpy
```

**Files:**

| File | Purpose |
|---|---|
| `starter.py` | **Your work.** 4 `TODO`s + 17-check self-test + 2 experiments. |
| `solution.py` | Reference solution, plus a full transformer block and a scaling demo. |
| `SOLUTION.md` | The reasoning behind each answer. |

> **💡 This is the most satisfying lab in the course.** Attention has a reputation for being impenetrable. It's four lines of matrix arithmetic. Writing it yourself dissolves the mystery permanently.

---

## The tasks

Run it first to see where you stand:

```powershell
python labs/04-transformers/starter.py
```

### Task 1 — `softmax` (Module 4 §4.3, step 3)

Turn scores into weights that sum to 1.

**Two things to get right:**

- **Subtract the row max first.** `x - np.max(x, axis=-1, keepdims=True)`. Mathematically this changes nothing; numerically it's essential — `np.exp(1000)` overflows to `inf` and everything becomes `nan`. There's a test for exactly this.
- **`axis=-1` and `keepdims=True`.** Softmax must be applied **per row**. Get the axis wrong and your code runs, returns the right *shape*, and is completely wrong. There's a test for this too, because it's the single most common bug in hand-written attention.

Do this one first — tasks 2, 3 and 4 all depend on it.

### Task 2 — `scaled_dot_product_attention` (§4.3)

The formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Six lines:

```python
d_k = Q.shape[-1]
scores = Q @ K.T                    # 1. score every pair
scores = scores / np.sqrt(d_k)      # 2. scale
if mask is not None:
    scores = scores + mask          # 3. mask (optional)
weights = softmax(scores)           # 4. normalise into weights
output = weights @ V                # 5. blend the values
return output, weights
```

Return **both** the output and the weights — you need the weights for the experiments, and inspecting them is how you debug attention.

> **📌 One shape detail:** the output shape follows `V`, not `Q`. With `Q` of shape `(5, 8)` and `V` of shape `(5, 3)`, the output is `(5, 3)`. There's a test for it, because getting `d_k` and `d_v` confused is a common stumble.

### Task 3 — `causal_mask` (§4.7)

Build the additive mask that stops a token seeing its own future.

```python
future = np.triu(np.ones((n, n)), k=1)     # 1s strictly above the diagonal
return np.where(future == 1, -np.inf, 0.0)
```

**Why `-inf` and not `0`?** The mask is added *before* softmax. `exp(-inf) = 0`, so those positions get exactly zero weight and the remaining weights still sum to 1. Using `0` gives `exp(0) = 1` — a substantial weight — which leaks the future. That model looks brilliant in training and fails completely at inference.

### Task 4 — `multi_head_attention` (§4.4)

Run several heads in parallel, concatenate, project.

```python
for W_Q, W_K, W_V in zip(W_Q_list, W_K_list, W_V_list):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    output, _ = scaled_dot_product_attention(Q, K, V)
    head_outputs.append(output)

combined = np.concatenate(head_outputs, axis=-1)
return combined @ W_O
```

Every head sees the **same input** but projects it with **its own weights**, so each examines a different kind of relationship. `W_O` then mixes their findings.

**✅ Complete at `All 17 checks passed.`**

---

## The experiments

These run automatically once all checks pass.

### Experiment 1 — One token, two contexts

The payoff. `"bit"` is embedded **identically** in both sentences, but attention gives it different output vectors because its neighbours differ.

You'll see attention weights per sentence, then the cosine similarity between the two output vectors — a number close to zero, meaning the two representations have genuinely diverged.

**That's a contextual embedding**, and it's precisely what static embeddings cannot produce (Module 3 §3.4).

> **⚠️ Read the honest caveat in the output.** These weight matrices are **random, not trained**. The difference is real but not yet *meaningful* — nothing has taught this toy model that *dog* and *man* imply a verb. Training is what turns the mechanism into meaning. What you're seeing is the mechanism, and that's the point.

### Experiment 2 — Causal masking

Prints two attention matrices side by side:

```
WITHOUT mask (BERT-style)          WITH causal mask (GPT-style)
        The  cat  sat down                 The  cat  sat down
  The  0.43 0.16 0.29 0.12           The  1.00 0.00 0.00 0.00
  cat  0.29 0.44 0.18 0.10           cat  0.40 0.60 0.00 0.00
  sat  0.03 0.01 0.83 0.13           sat  0.03 0.01 0.96 0.00
  down 0.03 0.01 0.31 0.65           down 0.03 0.01 0.31 0.65
```

Look at the upper-right triangle. Exact zeros on the right. **That single difference is what separates a generative model from an understanding-only one** (§4.8).

Note also that the first token's row becomes `1.00` — with nothing but itself to attend to, it must give itself all the weight.

---

## 🚀 Stretch Challenges

### 1. Prove the scaling matters

Run `python solution.py` and read the `sqrt(d_k)` demo, then rebuild it yourself: for `d_k` in 4, 64, 512, compare the maximum softmax weight with and without scaling.

Unscaled, the max weight climbs toward `1.0000` as `d_k` grows — softmax has saturated into a hard pick, and gradients vanish. **You've just demonstrated why one division by a square root is load-bearing.**

### 2. Build a full transformer block

`solution.py` has `layer_norm`, `feed_forward` and `transformer_block`. Write them yourself, then stack three blocks and confirm the output shape never changes.

**Then test the residual connection.** Remove the `X +` and stack ten blocks. Watch the values explode or collapse. That's why residuals exist — without them, deep networks simply don't train.

### 3. Add positional encoding

Implement the sinusoidal encoding from §4.5, then run this test:

```python
sentence_a = ["The", "dog", "bit", "the", "man"]
sentence_b = ["The", "man", "bit", "the", "dog"]
```

**Without** positional encoding, attention produces the same set of output vectors for both — just reordered. **With** it, they genuinely differ.

This is the cleanest possible demonstration that attention is order-blind on its own.

### 4. Make attention efficient

Your implementation builds the full `(n, n)` score matrix. Measure how memory grows for `n` = 100, 1,000, 10,000:

```python
import numpy as np
for n in [100, 1_000, 10_000]:
    print(f"n={n:>6}: {(n * n * 8) / 1e6:>10.1f} MB for one attention matrix")
```

At 10,000 tokens one matrix is ~800 MB — **and there's one per head per layer.** You've just derived, with arithmetic, why long context windows are hard and why FlashAttention and sliding-window attention exist.

### 5. Visualise attention on a real model

```powershell
pip install transformers torch
```

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer("The dog bit the man", return_tensors="pt")
outputs = model(**inputs)

# One attention matrix per layer; each is (batch, heads, tokens, tokens)
print(f"{len(outputs.attentions)} layers, shape {outputs.attentions[0].shape}")
```

Plot a few heads as heatmaps. **Some are interpretable — one tracking subject/verb, another resolving determiners. Many aren't**, attending mostly to punctuation or the first token as a no-op. That mix is worth seeing directly: it's the honest picture of what attention maps do and don't tell you, and it's the first entry in Module 4's "Common Mistakes" section.

---

## When you're done

1. Attempt everything before opening the answers.
2. Read **[`SOLUTION.md`](SOLUTION.md)** — including why the axis bug is so dangerous and what "O(n²)" actually costs.
3. Run `python solution.py` for the scaling demo and a stacked transformer block.

**Then consider [Karpathy's *Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY).** With this lab behind you it's genuinely followable — you've already written the hardest part.

**Next:** Module 5 — Prompt Engineering. Back to practical work, and everything you build from here rests on what you now understand about attention, position and context.
