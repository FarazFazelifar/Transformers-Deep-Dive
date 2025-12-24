# Attention

The mechanism that allows the model to weigh the importance of different tokens relative to each other.

## Scaled Dot-Product Attention

**Formula:**
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}} + M\right)V$$

where $Q$ (queries), $K$ (keys), and $V$ (values) are matrices computed from input tokens, $d_k$ is the dimensionality of the key vectors, and $M$ is an optional mask (e.g., for causal masking).

**Intuition:** Each query compares itself to every key (similarity via dot product). The softmax turns similarities into a distribution over values; the output is a weighted sum of values.

## Self-Attention

In self-attention, queries, keys, and values are all linear projections of the same input sequence. This allows every token to gather contextual information from the entire sequence.

## Multi-Headed Attention

Instead of a single attention computation, multi-head attention performs $h$ parallel attention operations with smaller dimensions, then concatenates the results and projects back to the model dimension. This enables the model to attend to information from different representation subspaces.

**Formula (multi-head):**
$$\mathrm{MultiHead}(X)=\mathrm{Concat}(\mathrm{head}_1,...,\mathrm{head}_h)W^O$$
$$\mathrm{head}_i=\mathrm{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

## Masked (Causal) Attention

For autoregressive generation, causal masks prevent tokens from attending to future positions. This is usually implemented by adding a large negative number (e.g., -1e9) to the logits for disallowed key positions before softmax.

## Cross-Attention

In encoder-decoder models, cross-attention uses queries from the decoder and keys/values from the encoder, enabling the decoder to condition on encoder outputs.


## References

- **Attention is All You Need** — [Vaswani et al., 2017 (foundational paper)](https://arxiv.org/abs/1706.03762)
- **The Annotated Transformer** — [step-by-step annotated implementation and explanation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- **The Illustrated Transformer** — visual walkthrough by Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- **Hugging Face Course — Attention** — [concise course material and examples](https://huggingface.co/course/chapter2/3)
