# Rotary Positional Embeddings (RoPE)

## 1. The problem RoPE solves

A transformer's attention mechanism is permutation-equivariant: if you shuffle the tokens in the input, the output shuffles the same way. The model has no built-in notion of "position." To do language modeling we need to inject positional information somehow.

Two earlier approaches:

- **Sinusoidal absolute embeddings** (original Transformer): add a position-dependent vector to each token's embedding. The model has to learn to disentangle content from position, and it never naturally encodes "tokens 5 and 6 are close."
- **Learned absolute embeddings** (GPT-2): same problem, plus a hard cap at the maximum sequence length seen during training.

What we actually want is for the **attention score** between a query at position `m` and a key at position `n` to depend only on `m − n` — the relative distance — not on `m` and `n` individually. RoPE achieves exactly this, with no learned parameters.

## 2. The 2D foundation

Take a 2D vector `x = (x₁, x₂)` and rotate it by angle `mθ`:

```
R_m · x = ( cos(mθ)  -sin(mθ) ) ( x₁ )
          ( sin(mθ)   cos(mθ) ) ( x₂ )
```

Now the magic. Suppose query `q` lives at position `m` and key `k` at position `n`. We rotate them by their respective positional angles:

```
q' = R_m · q
k' = R_n · k
```

The attention score (a dot product) becomes:

```
q'ᵀ · k' = (R_m q)ᵀ (R_n k) = qᵀ R_mᵀ R_n k
```

Two facts about 2D rotations:

1. **Orthogonality:** `R_mᵀ = R_{-m}` (rotating backward by `m` undoes a forward rotation by `m`).
2. **Additivity:** `R_a · R_b = R_{a+b}` (composing rotations adds angles).

Combining: `R_mᵀ R_n = R_{-m} R_n = R_{n-m}`. So:

```
q'ᵀ · k' = qᵀ R_{n-m} k
```

The score depends **only on `n − m`**. The absolute positions have vanished. This is the entire point of RoPE.

Note that `R_{n-m}` is a rotation, so it preserves the norm of `k`. This is why we need an **orthogonal** transformation specifically: a non-orthogonal one (e.g. an arbitrary linear map) would scale vectors differently at different positions and break the relative-distance property.

## 3. Generalizing to `d` dimensions

A query/key vector in a transformer head has dimension `d = head_dim`, typically 64 or 128 — not 2. RoPE handles this by **splitting the vector into `d/2` independent 2D pairs and rotating each pair by its own frequency**:

```
pair i:  (x_{2i}, x_{2i+1})    rotated by angle  m · θ_i
```

The frequencies form a geometric progression:

```
θ_i = base^(-2i / d),    i ∈ {0, 1, ..., d/2 - 1}
```

with `base` typically 10,000 (original RoPE) or 100,000+ (modern long-context models — including nanochat). Low-`i` channels get high frequencies (rotate fast → sensitive to short-range position); high-`i` channels get low frequencies (rotate slowly → sensitive to long-range position). The geometric spacing covers many timescales of "distance" simultaneously.

The full operation can be written as a block-diagonal rotation `R_m = diag(R_{m,θ₀}, R_{m,θ₁}, ..., R_{m,θ_{d/2-1}})`, but in practice we never materialize this matrix — we use an elementwise trick (Section 6).

The relative-distance property still holds: each 2D slice independently satisfies `R_mᵀ R_n = R_{n-m}`, so the full dot product is a sum of `d/2` independent relative-distance terms.

## 4. Why attention decays with distance

Splitting the dot product across slices:

```
qᵀ R_{n-m} k  =  Σᵢ ‖qᵢ‖ ‖kᵢ‖ · cos(γᵢ + (m-n) θᵢ)
```

where `γᵢ` is the natural content angle between `qᵢ` and `kᵢ` in slice `i` (independent of position), and `(m-n)θᵢ` is the positional phase shift in that slice.

Now interpret the sum:

- **Small `|m-n|` (close tokens):** the term `(m-n)θᵢ` is small for *every* `i`. All slices contribute coherently — the cosine values are close to `cos(γᵢ)`, so content similarity dominates. The attention score is roughly `Σᵢ ‖qᵢ‖‖kᵢ‖ cos(γᵢ)` — the un-rotated dot product.

- **Large `|m-n|` (far tokens):** different slices have wildly different `θᵢ`, so `(m-n)θᵢ` lands at very different angles across `i`. Some terms are positive, some negative — they **destructively interfere**. The sum's magnitude shrinks. This is the "decay with distance" property of RoPE attention scores.

Crucially, this decay is a **consequence of the geometric frequency spacing**, not of rotation per se. If you used a single frequency for all slices, the dot product would oscillate periodically with distance instead of decaying. The spectrum of frequencies is what produces washout at long range.

## 5. What RoPE does and doesn't give you

**Does:**
- Inject position information without learned parameters.
- Make attention scores depend only on relative distance.
- Produce natural decay with distance through frequency interference.

**Does not:**
- Magically extrapolate beyond training length. The math is well-defined for any `m`, but the model has only seen specific ranges of `(m-n)θᵢ` during training. Larger relative distances put the rotational phase into regions the model never observed, and quality degrades. Extending context length requires explicit techniques (linear scaling, NTK-aware scaling, YaRN) that adjust `θᵢ` so the same dot-product distribution covers a longer range.
- Move vectors "spatially close" in any geometric sense. RoPE preserves vector norms (rotations are isometries). What it does is align the *phases* of the 2D slices for nearby positions, which makes the dot product large when content is similar.

## 6. The implementation: rotate-half

We never build the block-diagonal matrix. Two reasons: it's mostly zeros, and elementwise ops are far faster on GPUs.

If we precompute two arrays of shape `(seq_len, d/2)`:

```
cos[m, i] = cos(m · θᵢ)
sin[m, i] = sin(m · θᵢ)
```

then the rotation of a vector `x` of shape `(..., d)` at position `m` becomes:

```
x_rotated = x * cos_expanded + rotate_half(x) * sin_expanded
```

where `rotate_half(x) = concat(-x[d/2:], x[:d/2])` (negate and swap the two halves). This implements the same per-slice 2D rotation, but as three pointwise tensor ops — no matmul, no Python loop over slices.

There are two equivalent conventions for which channels form a "pair":

- **Interleaved**: pair channel `2i` with `2i+1`. Used in the original RoPE paper.
- **Halved**: pair channel `i` with `i + d/2`. Used in most modern code (Llama, nanochat) because the `rotate_half` trick maps cleanly to `(x₁, x₂) = (x[:d/2], x[d/2:])`.

Both are mathematically equivalent — just a relabeling of channels. nanochat uses the halved convention, which is why `apply_rotary_emb` in `gpt.py` does:

```python
d = x.shape[3] // 2
x1, x2 = x[..., :d], x[..., d:]
y1 = x1 * cos + x2 * sin
y2 = -x1 * sin + x2 * cos
return torch.cat([y1, y2], 3)
```

This is exactly the 2D rotation `(x₁, x₂) → (x₁ cos + x₂ sin, -x₁ sin + x₂ cos)` applied in parallel across all `d/2` slices and all heads.

## 7. Mapping to nanochat's code

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))   # θᵢ for i = 0..d/2-1
    t = torch.arange(seq_len, dtype=torch.float32, device=device)  # positions m
    freqs = torch.outer(t, inv_freq)                         # freqs[m, i] = m · θᵢ
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, T, 1, d/2)
    return cos, sin
```

| Code | Math |
|------|------|
| `channel_range` | the indices `2i` for `i = 0..d/2-1` |
| `inv_freq` | the frequency vector `θᵢ = base^(-2i/d)` |
| `t` | the position vector `m ∈ {0,...,seq_len-1}` |
| `freqs[m, i]` | the angle `m · θᵢ` |
| `cos`, `sin` | the precomputed `cos(m · θᵢ)` and `sin(m · θᵢ)` |
| `[None, :, None, :]` | broadcasting dims for batch and heads |

The shape `(1, seq_len, 1, d/2)` means at forward time it broadcasts against query/key tensors of shape `(B, T, H, d)` — same rotation applied to every batch element and every head, with a different rotation per position and per slice.

`base = 100000` (rather than the original 10000) gives slower rotation in the high-`i` channels, which extends the effective context range. nanochat further over-allocates `seq_len * 10` worth of cached positions so longer-than-trained-on inference doesn't blow past the cache.

## 8. One-paragraph mental model

RoPE views each query/key vector as a stack of `d/2` independent 2D vectors. At position `m`, each pair is rotated by its own position-dependent angle `m·θᵢ`, where the `θᵢ` form a geometric spectrum. Because rotations are orthogonal and additive, the dot product of a position-`m` query with a position-`n` key reduces to the same query-key dot product evaluated under a single rotation `R_{n-m}` — depending only on relative distance. The geometric spread of frequencies means short-range interactions stay coherent across slices while long-range interactions have their phases scatter and cancel, producing a natural distance decay in attention scores. No parameters, no max-length wall in the math (though the model only generalizes to distances near those it trained on), and a fast elementwise implementation via the rotate-half trick.
