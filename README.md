# 🌌 Transformers Deep Dive

This repository is a  deep dive into the Transformer architecture. Each section is built from the ground up with a focus on the mathematics behind transformers and newer approaches in the field.

---

## 🛠 Project Principles

- **Math-First**: We start with the LaTeX formulation before touching a single line of code.
- **From Scratch**: Logic is implemented in pure PyTorch/NumPy without high-level library abstractions.
---

## 🗺 Roadmap

### Phase 1: The Building Blocks
- **[01. Introduction & Prerequisites](01-Introduction/01_Introduction.md)**: Math foundations and project glossary.
- **[02. Input Processing](02-Input_Processing/02_Input_Processing.md)**: Tokenization (BPE, WordPiece) and Positional Encodings (RoPE).
- **[03. Attention Mechanisms](03-Attention/03_Attention.md)**: Scaled Dot-Product, Multi-Head, and Causal Masking.
- **[04. FFN & Normalization](04-FFN_and_Norms/04_FFN_and_Norms.md)**: Feed-Forward Networks, LayerNorm, RMSNorm, and SwiGLU.

### Phase 2: The Architecture
- **[05. The Encoder Block](05-Encoder_Block)**: Bi-directional context and structure.
- **[06. The Decoder Block](06-Decoder_Block)**: Auto-regressive generation and causal flow.
- **[07. The Full Transformer](07-Full_Transformer)**: Recreating the "Attention Is All You Need" paper.

### Phase 3: The Family Tree
- **[08. Encoder-Only Models](08-Encoder_Only_Models)**: Understanding BERT and classification.
- **[09. Decoder-Only Models](09-Decoder_Only_Models)**: Exploring GPT, Llama, and modern LLMs.
- **[10. Encoder-Decoder Models](10-Encoder_Decoder_Models)**: T5, Whisper, and hybrid systems.

---

## 📚 Sources & References

- **Understanding LLMs Survey**: [arXiv:2401.02038](https://arxiv.org/pdf/2401.02038v2)
- **Machine Learning Mastery**: [High-quality implementation tutorials.](https://machinelearningmastery.com)

---