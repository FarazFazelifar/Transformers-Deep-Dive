# Input Processing

Transform raw text into processed vectors that the model can interpret.

## Tokenization (BPE, WordPiece)

**Intuition:** Breaking text into manageable subword units. Rather than treating each word as atomic (which creates huge vocabularies) or each character separately (which creates long sequences), subword tokenization finds a middle ground.

**Byte Pair Encoding (BPE):**
- Starts with characters, iteratively merges the most frequent adjacent pairs
- Creates a vocabulary of frequent subwords
- Used in GPT models

**WordPiece:**
- Similar to BPE but merges pairs based on likelihood probability instead of frequency
- Marks subword continuations with `##` prefix (e.g., "transformer" → "transform" + "##er")
- Used in BERT

**Example:**
```
Input: "tokenization"
WordPiece: ["token", "##ization"]
BPE: ["token", "ization"]
```

## Embeddings

**Intuition:** Each token is converted to a dense vector of learned numbers that capture semantic meaning. Similar words end up with similar vectors in this learned space.

**Mathematical Definition:**
$$E = \text{Embedding Matrix} \in \mathbb{R}^{|V| \times d}$$

where $|V|$ is vocabulary size and $d$ is embedding dimension (e.g., 512 or 768).

For token $t$ with index $i$:
$$\mathbf{e}_t = E[i] \in \mathbb{R}^{d}$$

**Key Properties:**
- Learned during training via backpropagation
- Captures semantic relationships: $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$
- Higher dimensional embeddings can capture more nuance but require more computation

## Positional Encodings (Sinusoidal, RoPE, ALiBi)

Transformers lack built-in sequence order awareness (unlike RNNs). Positional encodings inject position information.

### Sinusoidal Positional Encoding

**Formula:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is position in sequence, $i$ is dimension index, $d$ is embedding dimension.

**Intuition:** Different frequencies allow the model to learn relative positions. Lower frequencies capture global structure, higher frequencies capture fine-grained positions.

**Final Representation:**
$$\mathbf{x}_{pos} = \mathbf{e}_t + PE_{pos}$$

### Rotary Position Embedding (RoPE)

**Intuition:** Encodes position as a rotation in the embedding space. Relative positions become angle differences, naturally capturing distance.

**Formula:** Apply rotation matrix to query and key vectors:
$$\mathbf{q}_{m} = R_{\Theta,m} \mathbf{q}$$
$$\mathbf{k}_{n} = R_{\Theta,n} \mathbf{k}$$

where $R_{\Theta,m}$ is rotation matrix for position $m$ with frequencies $\Theta$.

**Advantage:** Enables extrapolation to longer sequences than training length.

### Attention with Linear Biases (ALiBi)

**Intuition:** Instead of adding positional embeddings, directly bias the attention scores based on token distance. Simpler and more efficient.

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{bias}_{rel\text{-}dist}\right)V$$

where the bias matrix penalizes attention between distant tokens:
$$\text{bias}_{ij} = -\alpha |i - j|$$

$\alpha$ is a learned penalty coefficient per attention head.

**Advantages:**
- No position embeddings needed (smaller model)
- Trains 11% faster and uses 11% less memory than sinusoidal
- Naturally extrapolates to longer sequences

## References

* **Tokenization Overview:** [Hugging Face Course - Tokenizers](https://huggingface.co/course/en/chapter2/4)
* **Word2Vec (Embeddings):** [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) - Foundational work on dense embeddings
* **Attention is All You Need:** [Transformer Paper](https://arxiv.org/abs/1706.03762) - Introduces sinusoidal positional encoding
* **RoPE:** [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Rotation-based position encoding
* **ALiBi:** [Train Short, Test Long with ALiBi](https://arxiv.org/abs/2108.12409) - Linear bias position method enabling extrapolation
