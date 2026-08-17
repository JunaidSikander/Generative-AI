"""
solution.py - Lab 4 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import numpy as np


# ======================================================================
# TASK 1 - softmax
# ======================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax: turn scores into weights that sum to 1."""
    # Subtracting the row max is mathematically a no-op: exp(a-c)/sum(exp(a-c))
    # equals exp(a)/sum(exp(a)). Numerically it is essential - without it,
    # exp(1000) overflows to inf and you get nan.
    shifted = x - np.max(x, axis=-1, keepdims=True)

    exponentiated = np.exp(shifted)

    # axis=-1 sums along the LAST axis (across each row).
    # keepdims=True preserves the shape so broadcasting divides row-wise.
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


# ======================================================================
# TASK 2 - scaled_dot_product_attention
# ======================================================================

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray = None):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V."""
    d_k = Q.shape[-1]

    # STEP 1: score every query against every key.
    # (n_tokens, d_k) @ (d_k, n_tokens) -> (n_tokens, n_tokens)
    # This all-pairs comparison is the source of O(n^2) cost.
    scores = Q @ K.T

    # STEP 2: scale. Dot products in high dimensions grow large, which pushes
    # softmax into a saturated region where gradients vanish.
    scores = scores / np.sqrt(d_k)

    # STEP 3 (optional): mask. Additive, and applied BEFORE softmax so that
    # exp(-inf) = 0 removes those positions entirely.
    if mask is not None:
        scores = scores + mask

    # STEP 4: normalise each row into weights summing to 1.
    weights = softmax(scores)

    # STEP 5: blend the values. Each token's output is a weighted average of
    # every token's value vector.
    output = weights @ V

    return output, weights


# ======================================================================
# TASK 3 - causal_mask
# ======================================================================

def causal_mask(n_tokens: int) -> np.ndarray:
    """Build an additive mask that hides future positions."""
    # np.triu(..., k=1) keeps the strictly-upper triangle: exactly the
    # positions that come AFTER the current one.
    future = np.triu(np.ones((n_tokens, n_tokens)), k=1)

    # -inf where blocked, 0.0 where allowed. Adding 0 changes nothing;
    # adding -inf drives that weight to exactly zero after softmax.
    return np.where(future == 1, -np.inf, 0.0)


# ======================================================================
# TASK 4 - multi_head_attention
# ======================================================================

def multi_head_attention(X: np.ndarray, W_Q_list: list, W_K_list: list,
                         W_V_list: list, W_O: np.ndarray) -> np.ndarray:
    """Run several attention heads in parallel and combine them."""
    head_outputs = []

    # Every head sees the SAME input X but projects it with its own weights,
    # so each ends up examining a different kind of relationship.
    for W_Q, W_K, W_V in zip(W_Q_list, W_K_list, W_V_list):
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        output, _ = scaled_dot_product_attention(Q, K, V)
        head_outputs.append(output)

    # Concatenation just stacks the heads' findings side by side...
    combined = np.concatenate(head_outputs, axis=-1)

    # ...and W_O lets the model MIX them, weighting each head's contribution.
    return combined @ W_O


# ======================================================================
# BONUS - a complete transformer block
# ======================================================================

def layer_norm(x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Normalise each token's vector to mean 0, variance 1."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # epsilon guards against dividing by zero when variance is tiny.
    return (x - mean) / np.sqrt(variance + epsilon)


def feed_forward(x: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    """Position-wise FFN: expand, apply a non-linearity, contract."""
    # np.maximum(0, ...) is ReLU. Applied to each token independently -
    # no information moves between tokens here.
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


def transformer_block(X, attention_weights, ffn_weights, mask=None):
    """One full transformer block: attention + FFN, with residuals.

    Args:
        X:                 Input, shape (n_tokens, d_model)
        attention_weights: (W_Q_list, W_K_list, W_V_list, W_O)
        ffn_weights:       (W1, b1, W2, b2)
        mask:              Optional causal mask.

    Returns:
        Output of the same shape as X - which is what lets blocks stack.
    """
    W_Q_list, W_K_list, W_V_list, W_O = attention_weights
    W1, b1, W2, b2 = ffn_weights

    # --- Sub-layer 1: attention, with a residual connection ---
    # Note the "X +" : the block learns a REFINEMENT to X, not a replacement.
    # This is what makes 100-layer networks trainable at all.
    normalised = layer_norm(X)
    attention_out = multi_head_attention(normalised, W_Q_list, W_K_list, W_V_list, W_O)
    X = X + attention_out

    # --- Sub-layer 2: feed-forward, also with a residual ---
    normalised = layer_norm(X)
    X = X + feed_forward(normalised, W1, b1, W2, b2)

    return X


# ======================================================================
# Demonstrations
# ======================================================================

def demo_attention_basics():
    print("=" * 72)
    print("  THE FOUR STEPS, SHOWN")
    print("=" * 72)
    print()

    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    d_k = Q.shape[-1]

    raw = Q @ K.T
    scaled = raw / np.sqrt(d_k)
    weights = softmax(scaled)
    output = weights @ V

    print(f"  Q = {Q.tolist()}")
    print(f"  K = {K.tolist()}")
    print(f"  V = {V.tolist()}")
    print()
    print(f"  1. Q @ K.T          = {raw.round(4).tolist()}")
    print(f"  2. / sqrt(d_k={d_k})    = {scaled.round(4).tolist()}")
    print(f"  3. softmax          = {weights.round(4).tolist()}   (sums to {weights.sum():.1f})")
    print(f"  4. @ V              = {output.round(4).tolist()}")
    print()
    print("  The query pointed at K's first row, so that row won 67% of the")
    print("  weight - and the output leans toward V's first row accordingly.")
    print()


def demo_scaling_matters():
    print("=" * 72)
    print("  WHY THE sqrt(d_k) SCALING EXISTS")
    print("=" * 72)
    print()

    rng = np.random.default_rng(3)

    print(f"  {'d_k':>6}{'max |score| unscaled':>24}{'max weight unscaled':>22}{'scaled':>10}")
    print("  " + "-" * 62)

    for d_k in [4, 64, 512]:
        Q = rng.normal(size=(1, d_k))
        K = rng.normal(size=(6, d_k))

        unscaled = Q @ K.T
        scaled = unscaled / np.sqrt(d_k)

        max_unscaled_weight = softmax(unscaled).max()
        max_scaled_weight = softmax(scaled).max()

        print(f"  {d_k:>6}{np.abs(unscaled).max():>24.2f}"
              f"{max_unscaled_weight:>22.4f}{max_scaled_weight:>10.4f}")

    print()
    print("  As d_k grows, unscaled scores grow with it and softmax saturates -")
    print("  the max weight heads toward 1.0, so the 'blend' becomes a hard pick")
    print("  and gradients vanish. The scaled column stays reasonable. That is")
    print("  the entire job of dividing by sqrt(d_k).")
    print()


def demo_transformer_block():
    print("=" * 72)
    print("  A COMPLETE TRANSFORMER BLOCK")
    print("=" * 72)
    print()

    n_tokens, d_model, n_heads = 5, 16, 4
    d_head = d_model // n_heads
    d_ffn = d_model * 4          # the usual 4x expansion

    rng = np.random.default_rng(11)
    X = rng.normal(size=(n_tokens, d_model))

    attention_weights = (
        [rng.normal(size=(d_model, d_head)) * 0.1 for _ in range(n_heads)],
        [rng.normal(size=(d_model, d_head)) * 0.1 for _ in range(n_heads)],
        [rng.normal(size=(d_model, d_head)) * 0.1 for _ in range(n_heads)],
        rng.normal(size=(n_heads * d_head, d_model)) * 0.1,
    )
    ffn_weights = (
        rng.normal(size=(d_model, d_ffn)) * 0.1,
        np.zeros(d_ffn),
        rng.normal(size=(d_ffn, d_model)) * 0.1,
        np.zeros(d_model),
    )

    print(f"  Input shape : {X.shape}")

    # Stack blocks - the output shape is identical to the input shape,
    # which is exactly why they can be stacked at all.
    H = X
    for layer in range(1, 4):
        H = transformer_block(H, attention_weights, ffn_weights,
                              mask=causal_mask(n_tokens))
        drift = float(np.linalg.norm(H - X))
        print(f"  After block {layer}: {H.shape}   cumulative change from input = {drift:.3f}")

    print()
    print(f"  d_model={d_model}, {n_heads} heads -> {d_head} dims each.")
    print("  Heads SPLIT the dimensions rather than adding to them, which is why")
    print("  multi-head attention costs about the same as single-head.")
    print()
    print("  Note the shape never changes. That invariant is what lets you stack")
    print("  12 blocks (GPT-2) or 100+ (frontier models) without any plumbing.")
    print()


if __name__ == "__main__":
    demo_attention_basics()
    demo_scaling_matters()
    demo_transformer_block()
