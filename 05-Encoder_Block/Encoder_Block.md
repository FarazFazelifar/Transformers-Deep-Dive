# Encoder Block

The encoder block is the fundamental repeating unit of a transformer encoder. It combines self-attention and feed-forward processing with residual connections and normalization to build deep, bidirectional context understanding.

## Architecture

An encoder block typically follows this pattern:

1. **Layer Normalization** — normalize input
2. **Self-Attention** — compute attention over all positions (bidirectional)
3. **Residual Connection** — add back the original input
4. **Layer Normalization** — normalize again
5. **Feed-Forward Network** — position-wise MLP
6. **Residual Connection** — add back the attention output

**Formula:**
$$y_1 = x + \mathrm{MultiHeadAttention}(\mathrm{LayerNorm}(x))$$
$$y = y_1 + \mathrm{FFN}(\mathrm{LayerNorm}(y_1))$$

This is called "Pre-Norm" architecture (normalize before the sublayer).

## Key Properties

- **Bidirectional Context:** Unlike decoders, encoders can attend to future tokens. Every token sees the entire sequence.
- **Stacking:** Multiple encoder blocks are stacked (typically 12 for BERT, 24 for BERT-large) to build deeper representations.
- **Shared Parameters:** All blocks use independent parameters (no weight sharing across layers, though some modern architectures experiment with this).

## Training & Inference

- **Training:** All positions computed in parallel; efficient.
- **Inference:** Given full input sequence, encoder computes once to produce contextualized embeddings for downstream tasks.

## References

- **Attention is All You Need** (original Transformer): https://arxiv.org/abs/1706.03762
- **BERT: Pre-training of Deep Bidirectional Transformers** — how encoder blocks are used: https://arxiv.org/abs/1810.04805
- **LayerNorm and Residuals** — architectural building blocks: https://arxiv.org/abs/1607.06450
