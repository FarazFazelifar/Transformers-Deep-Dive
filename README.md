# Transformers-Deep-Dive

## Phase 1: The Building Blocks

**01. Introduction & Prerequisites**

Establish the necessary mathematical and conceptual foundations before diving into the architecture.
* Glossary & Notation
* Math Refreshers (Linear Algebra, Probability)

**02. Input Processing**

Transform raw text into processed vectors that the model can interpret.
* Tokenization (BPE, WordPiece)
* Embeddings
* Positional Encodings (Sinusoidal, RoPE, ALiBi)

**03. Attention**

The mechanism that allows the model to weigh the importance of different words in relation to each other.
* Scaled Dot-Product Attention
* Self-Attention
* Multi-Headed Attention
* Masked Attention
* Cross-Attention

**04. FFN & Norms**

Feed-forward and normalization layers that process information after attention is applied.
* Feed Forward Networks
* Activation Functions (ReLU, GeLU, SwiGLU)
* Normalization (LayerNorm, RMSNorm)
* Residual Connections

---

## Phase 2: The Architecture

**05. The Encoder Block**
Bi-directional component designed to build deep understanding of context.
* Self-Attention + FFN Structure
* Bi-directional Context

**06. The Decoder Block**
Uni-directional component optimized for generating sequences one token at a time.
* Masked Self-Attention + FFN Structure
* Causal / Auto-regressive Generation

**07. The Full Transformer**
Combine encoder and decoder stacks to recreate the original architecture proposed in 2017.
* Encoder-Decoder Connection
* Sequence-to-Sequence Data Flow

---

## Phase 3: The Family Tree

**08. Encoder-Only Models**
Architectures like BERT that specialize in understanding and classification tasks.
* Masked Language Modeling
* Downstream Tasks (NER, Sentiment)

**09. Decoder-Only Models**
Architectures like GPT and Llama that dominate modern text generation and reasoning.
* Causal Language Modeling
* Zero-shot & Few-shot capabilities

**10. Encoder-Decoder Models**
Hybrid architectures ideal for sequence-to-sequence tasks like translation.
* T5 & Whisper
* Translation & Summarization

---

## Phase 4: Life Cycle of a Model

**11. Training Dynamics**
Optimization processes, loss functions, and schedules used to teach the model.
* Cross Entropy Loss
* Optimizers (AdamW)
* Learning Rate Schedulers (Warmup, Cosine Decay)

**12. Inference Mechanics**
How trained models generate output efficiently using techniques like KV caching.
* The Generation Bottleneck
* KV Caching
* Sampling Strategies (Temperature, Top-k, Nucleus)

---

## Phase 5: Modern Frontier

**13. Modern Optimizations**
Current techniques for making models faster, smaller, and easier to fine-tune.
* PEFT & LoRA
* Quantization (FP16, INT8)
* Flash Attention

**14. Vision Transformers (ViT)**
Extending the Transformer architecture beyond text to process visual data.
* Image Patching
* Multimodal Integration