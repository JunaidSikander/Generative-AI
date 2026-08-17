"""
starter.py - Lab 4: Build Attention From Scratch

Replace each TODO with working code. The self-test checks your work.

    python starter.py

Needs ONLY numpy. No API key, no downloads, no cost.

By the end you will have implemented the same computation that runs inside
every LLM in this course - and watched one token produce two different
vectors depending on its neighbours.
"""

import numpy as np


# ======================================================================
# TASK 1 - softmax
# Module 4, section 4.3 step 3
# ======================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax: turn scores into weights that sum to 1.

    Args:
        x: Array of scores. For a 2-D array, softmax is applied to EACH ROW
           independently, so every row of the result sums to 1.

    Returns:
        Array of the same shape, with each row summing to 1.

    Examples:
        >>> softmax(np.array([0.0, 0.0]))
        array([0.5, 0.5])
        >>> softmax(np.array([0.0, np.log(3)]))
        array([0.25, 0.75])
    """
    # TODO:
    #   1. Subtract the row max:  x - np.max(x, axis=-1, keepdims=True)
    #      This is mathematically a no-op but prevents exp() overflowing.
    #   2. Exponentiate.
    #   3. Divide by the row sum:  np.sum(..., axis=-1, keepdims=True)
    #
    # CRITICAL: axis=-1 and keepdims=True. Getting the axis wrong produces
    # code that runs, has the right shape, and is completely wrong.
    return x


# ======================================================================
# TASK 2 - scaled_dot_product_attention
# Module 4, section 4.3 - THE core computation
# ======================================================================

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray = None):
    """Scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q:    Queries, shape (n_tokens, d_k)
        K:    Keys,    shape (n_tokens, d_k)
        V:    Values,  shape (n_tokens, d_v)
        mask: Optional additive mask, shape (n_tokens, n_tokens). Contains 0
              where attention is allowed and -inf where it is blocked.

    Returns:
        (output, weights) where output is (n_tokens, d_v) and weights is
        (n_tokens, n_tokens) with each row summing to 1.

    Examples:
        >>> Q = np.array([[1.0, 0.0]])
        >>> K = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> V = np.array([[10.0, 0.0], [0.0, 10.0]])
        >>> out, w = scaled_dot_product_attention(Q, K, V)
        >>> w.round(4)
        array([[0.6698, 0.3302]])
    """
    # TODO:
    #   1. d_k = Q.shape[-1]
    #   2. scores = Q @ K.T
    #   3. scores = scores / np.sqrt(d_k)
    #   4. if mask is not None: scores = scores + mask
    #   5. weights = softmax(scores)
    #   6. output = weights @ V
    #   7. return output, weights
    return V, np.zeros((Q.shape[0], K.shape[0]))


# ======================================================================
# TASK 3 - causal_mask
# Module 4, section 4.7
# ======================================================================

def causal_mask(n_tokens: int) -> np.ndarray:
    """Build an additive mask that hides future positions.

    Args:
        n_tokens: Sequence length.

    Returns:
        (n_tokens, n_tokens) array with 0.0 on and below the diagonal
        (allowed) and -inf above it (blocked).

    Examples:
        >>> causal_mask(3)
        array([[  0., -inf, -inf],
               [  0.,   0., -inf],
               [  0.,   0.,   0.]])
    """
    # TODO:
    #   np.triu(np.ones((n, n)), k=1) gives 1s strictly ABOVE the diagonal -
    #   exactly the "future" positions. Turn those into -inf and the rest 0.0.
    # Hint: np.where(condition, value_if_true, value_if_false)
    #
    # Why -inf and not 0? The mask is added BEFORE softmax, and exp(-inf) = 0.
    # Using 0 would give exp(0) = 1, a substantial weight - leaking the future.
    return np.zeros((n_tokens, n_tokens))


# ======================================================================
# TASK 4 - multi_head_attention
# Module 4, section 4.4
# ======================================================================

def multi_head_attention(X: np.ndarray, W_Q_list: list, W_K_list: list,
                         W_V_list: list, W_O: np.ndarray) -> np.ndarray:
    """Run several attention heads in parallel and combine them.

    Args:
        X:        Input embeddings, shape (n_tokens, d_model)
        W_Q_list: One query weight matrix per head, each (d_model, d_k)
        W_K_list: One key weight matrix per head
        W_V_list: One value weight matrix per head, each (d_model, d_v)
        W_O:      Output projection, shape (n_heads * d_v, d_model)

    Returns:
        Combined output, shape (n_tokens, d_model)
    """
    # TODO:
    #   1. For each head, project X into Q, K, V:  X @ W_Q  etc.
    #   2. Run scaled_dot_product_attention on each; collect the outputs.
    #   3. Join them side by side:  np.concatenate(outputs, axis=-1)
    #   4. Mix them with the learned output projection:  combined @ W_O
    return X


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _close(got, expected, tolerance=1e-4):
    try:
        got_array = np.asarray(got, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        if got_array.shape != expected_array.shape:
            return False
        # equal_nan handles the -inf entries in the causal mask.
        return np.allclose(got_array, expected_array, atol=tolerance, equal_nan=True)
    except Exception:
        return False


def _run_self_test() -> int:
    checks = []

    # --- TASK 1: softmax ---
    checks.append(("1. softmax (equal scores -> equal weights)",
                   softmax(np.array([0.0, 0.0])), [0.5, 0.5]))
    checks.append(("1. softmax (known values)",
                   softmax(np.array([0.0, np.log(3)])), [0.25, 0.75]))
    checks.append(("1. softmax rows sum to 1 (2-D)",
                   np.sum(softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])), axis=-1),
                   [1.0, 1.0]))
    checks.append(("1. softmax is numerically stable (no overflow)",
                   softmax(np.array([1000.0, 1000.0])), [0.5, 0.5]))
    checks.append(("1. softmax per-ROW not per-column",
                   softmax(np.array([[0.0, 0.0], [0.0, np.log(3)]])),
                   [[0.5, 0.5], [0.25, 0.75]]))

    # --- TASK 2: attention ---
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    out, weights = scaled_dot_product_attention(Q, K, V)
    checks.append(("2. attention weights (known values)",
                   weights, [[0.669762, 0.330238]]))
    checks.append(("2. attention output (known values)",
                   out, [[6.697615, 3.302385]]))

    # Uniform Q/K means uniform attention, so output = mean of V.
    Q2 = np.zeros((3, 4))
    K2 = np.zeros((3, 4))
    V2 = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    out2, weights2 = scaled_dot_product_attention(Q2, K2, V2)
    checks.append(("2. uniform scores -> uniform weights",
                   weights2, np.full((3, 3), 1 / 3)))
    checks.append(("2. uniform attention -> mean of V",
                   out2, [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]))
    checks.append(("2. output shape follows V, not Q",
                   np.array(scaled_dot_product_attention(
                       np.zeros((5, 8)), np.zeros((5, 8)), np.zeros((5, 3)))[0].shape),
                   [5, 3]))

    # --- TASK 3: causal mask ---
    checks.append(("3. causal_mask(3) shape and values",
                   causal_mask(3),
                   [[0.0, -np.inf, -np.inf], [0.0, 0.0, -np.inf], [0.0, 0.0, 0.0]]))
    checks.append(("3. causal_mask(1) is a single zero",
                   causal_mask(1), [[0.0]]))

    # --- TASK 2 + 3 together: causal attention ---
    rng = np.random.default_rng(0)
    Qc = rng.normal(size=(4, 6))
    Kc = rng.normal(size=(4, 6))
    Vc = rng.normal(size=(4, 6))
    _, causal_weights = scaled_dot_product_attention(Qc, Kc, Vc, mask=causal_mask(4))
    checks.append(("2+3. first token attends only to itself",
                   causal_weights[0], [1.0, 0.0, 0.0, 0.0]))
    checks.append(("2+3. no attention leaks to the future",
                   np.triu(causal_weights, k=1).sum(), 0.0))
    checks.append(("2+3. causal rows still sum to 1",
                   causal_weights.sum(axis=-1), [1.0, 1.0, 1.0, 1.0]))

    # --- TASK 4: multi-head ---
    n_tokens, d_model, n_heads = 4, 8, 2
    d_head = d_model // n_heads
    rng4 = np.random.default_rng(1)
    X = rng4.normal(size=(n_tokens, d_model))
    W_Q_list = [rng4.normal(size=(d_model, d_head)) for _ in range(n_heads)]
    W_K_list = [rng4.normal(size=(d_model, d_head)) for _ in range(n_heads)]
    W_V_list = [rng4.normal(size=(d_model, d_head)) for _ in range(n_heads)]
    W_O = rng4.normal(size=(n_heads * d_head, d_model))
    mha_out = multi_head_attention(X, W_Q_list, W_K_list, W_V_list, W_O)
    checks.append(("4. multi_head_attention output shape",
                   np.array(mha_out.shape), [n_tokens, d_model]))
    checks.append(("4. multi_head_attention actually transforms X",
                   float(np.allclose(mha_out, X)), 0.0))

    print()
    print("=" * 72)
    print("  LAB 4 SELF-TEST - attention from scratch")
    print("=" * 72)
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
    print("-" * 72)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  You have just implemented the core of every transformer.")
        print("  Now run the experiments below.")
    else:
        print(f"  {failures} of {len(checks)} failing. Keep going.")
        print("  Start with softmax - tasks 2, 3 and 4 all depend on it.")
    print("-" * 72)
    print()
    return failures


# ======================================================================
# EXPERIMENT - the "bit" demonstration from Module 4, section 4.1
# ======================================================================

def experiment_bit():
    """Show the SAME token producing DIFFERENT vectors in different contexts."""
    print("=" * 72)
    print("  EXPERIMENT: one token, two contexts")
    print("=" * 72)
    print()

    d_model = 16
    rng = np.random.default_rng(42)

    # A tiny hand-built vocabulary. Each word gets a random embedding -
    # in a real model these would be learned.
    vocabulary = ["the", "dog", "bit", "man", "a", "little", "of", "water"]
    embeddings = {word: rng.normal(size=d_model) for word in vocabulary}

    # Random weight matrices, standing in for learned ones.
    W_Q = rng.normal(size=(d_model, d_model)) * 0.3
    W_K = rng.normal(size=(d_model, d_model)) * 0.3
    W_V = rng.normal(size=(d_model, d_model)) * 0.3

    sentence_a = ["the", "dog", "bit", "the", "man"]
    sentence_b = ["a", "little", "bit", "of", "water"]

    outputs = {}
    for label, words in [("A", sentence_a), ("B", sentence_b)]:
        X = np.array([embeddings[w] for w in words])
        Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        output, weights = scaled_dot_product_attention(Q, K, V)

        bit_index = words.index("bit")
        outputs[label] = output[bit_index]

        print(f"  Sentence {label}: {' '.join(words)}")
        print(f"  How 'bit' attends to each word:")
        for word, weight in zip(words, weights[bit_index]):
            bar = "#" * int(weight * 40)
            print(f"    {word:<8} {weight:.3f}  {bar}")
        print()

    # The input vector for "bit" was IDENTICAL in both sentences.
    similarity = float(
        np.dot(outputs["A"], outputs["B"])
        / (np.linalg.norm(outputs["A"]) * np.linalg.norm(outputs["B"]))
    )

    print("  " + "-" * 68)
    print(f"  Input vector for 'bit'  : identical in both sentences")
    print(f"  Output vector for 'bit' : cosine similarity = {similarity:.3f}")
    print()
    print("  The same token produced two different vectors, purely because its")
    print("  neighbours differed. That is a CONTEXTUAL embedding, and it is what")
    print("  static embeddings (word2vec, GloVe) fundamentally cannot do.")
    print()
    print("  HONEST CAVEAT: these weight matrices are random, not trained. So the")
    print("  difference here is real but not yet MEANINGFUL - nothing has taught")
    print("  this toy model that dog/man imply a verb. Training is what turns the")
    print("  mechanism into meaning. What you are seeing is the mechanism.")
    print()


def experiment_causal():
    """Show what causal masking does to the attention pattern."""
    print("=" * 72)
    print("  EXPERIMENT: causal masking")
    print("=" * 72)
    print()

    words = ["The", "cat", "sat", "down"]
    rng = np.random.default_rng(7)
    X = rng.normal(size=(len(words), 8))
    Q = K = V = X

    for label, mask in [("WITHOUT mask (BERT-style)", None),
                        ("WITH causal mask (GPT-style)", causal_mask(len(words)))]:
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        print(f"  {label}")
        print("           " + "".join(f"{w:>8}" for w in words))
        for i, word in enumerate(words):
            row = "".join(f"{weights[i][j]:>8.2f}" for j in range(len(words)))
            print(f"    {word:<8}{row}")
        print()

    print("  Read the upper-right triangle. Without the mask every token sees")
    print("  the whole sentence. With it, those entries are exactly 0.00 - no")
    print("  token can see its own future. That single difference is what")
    print("  separates a generative model from an understanding-only one.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_bit()
        experiment_causal()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
