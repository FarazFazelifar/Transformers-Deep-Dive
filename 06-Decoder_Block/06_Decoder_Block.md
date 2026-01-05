# Section 06: The Decoder Block

In previous sections, we built the **Encoder**, a bidirectional machine designed to "digest" a whole sequence and extract context. But a model that only understands is only half the battle. Now, we enter the world of **The Decoder Block**—the component responsible for *creation*.

The Decoder's primary job is auto-regression: generating the next token in a sequence based on what has come before it. This requires a fundamental shift in architecture: we must enforce **Causality**.

---

## 1. The Causal Bottleneck: Masked Self-Attention

In the Encoder, every token can see every other token. In the Decoder, this is "cheating." If we are predicting the 4th word in a sentence, we cannot allow the model to look at the 5th word.

To prevent this, we use **Masked Multi-Head Attention**. We apply a "look-ahead mask" to the attention scores before the softmax operation.

### Mathematical Formulation

Given queries $Q$, keys $K$, and values $V$, the masked attention is defined as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

Where $M$ is the causal mask matrix:
- $M_{ij} = 0$ if $i \geq j$ (allowed connection)
- $M_{ij} = -\infty$ if $i < j$ (future connection, blocked)

When we add $-\infty$ to the scores, the `softmax` function maps them to $0$, effectively killing the attention flow from future tokens.

---

## 2. Bridging the Gap: Encoder-Decoder Cross-Attention

The most unique feature of the original Transformer Decoder is the **Cross-Attention** layer. This is where the Decoder "listens" to the Encoder's summary of the input sequence.

Unlike Self-Attention, where $Q, K, V$ all come from the same source:
- **Queries ($Q$)**: Come from the Decoder's masked self-attention output (what we've generated so far).
- **Keys ($K$) and Values ($V$)**: Come from the final output of the Encoder stack (the "context" of the input).

This allows every position in the Decoder to attend to all positions in the input sequence, perfectly bridging the gap between understanding and generation.

---

## 3. Structural Assembly: Pre-Norm Standard

In this repository, we strictly follow the **Pre-Norm** architecture for modern stability. The data flow through a single Decoder Block looks like this:

1.  **Masked Self-Attention Path**: `Norm -> Masked MHA -> Add (Residual)`
2.  **Cross-Attention Path**: `Norm -> Cross MHA -> Add (Residual)`
3.  **Feed-Forward Path**: `Norm -> FFN -> Add (Residual)`

![Transformer Decoder Block: High-fidelity diagram comparing Post-Norm and Pre-Norm architectures.](../images/decoder_prenorm_diagram.png)
*Figure 1: Architectural comparison sourced from Sebastian Raschka. Our implementation follows the Pre-Norm path (right) for enhanced training stability.*

---

## 4. Why the "Shifted Right" Target?

When training a Decoder, we feed it the *target* sequence, but shifted right by one position. 
- **Input**: `<SOS> I love deep`
- **Output**: `I love deep learning`

This "teacher forcing" ensures that at every step $t$, the model uses the ground-truth sequence up to $t-1$ to predict the token at $t$. The causal mask ensures it doesn't just copy the next word from the input.

---

## References

- **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**: The foundational paper for the Transformer architecture.
- **[On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745)**: Rationale for the Pre-Norm architecture used in this implementation.
- **[Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)**: A visual guide to the data flow between Encoder and Decoder.
