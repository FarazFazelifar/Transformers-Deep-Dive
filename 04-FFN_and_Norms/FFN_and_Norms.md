# FFN & Norms

Feed-forward networks (FFN) and normalization layers process token representations after attention. They are crucial for transforming and stabilizing signals inside transformer layers.

## Feed-Forward Network (Position-wise)

**Structure:** A two-layer MLP applied independently at each sequence position:
$$\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2$$

where $W_1\in\mathbb{R}^{d_{ff}\times d}$, $W_2\in\mathbb{R}^{d\times d_{ff}}$, and $\sigma$ is a nonlinear activation (GeLU, ReLU, SwiGLU).

**Intuition:** The FFN expands the representation into a higher-dimensional space where nonlinear transformations can mix features, then projects back to the model dimension.

## Activation Functions

- **ReLU:** $\mathrm{ReLU}(x)=\max(0,x)$ — simple and fast.
- **GeLU:** Gaussian Error Linear Unit, smooth alternative often used in transformers:
$$\mathrm{GeLU}(x)=x\cdot\Phi(x)\approx x\cdot\frac{1}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right)$$
- **SwiGLU:** A gated linear unit variant used for improved parameter efficiency:
$$\mathrm{SwiGLU}(x)= (W_a x)\odot\mathrm{SiLU}(W_b x)$$

## Normalization

Normalization stabilizes training and controls variance across features.

### Layer Normalization
Normalizes across the feature dimension for each token:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\mathrm{LayerNorm}(x) = \gamma \odot \hat{x} + \beta$$

where $\mu$ and $\sigma^2$ are mean and variance computed per token across features; $\gamma,\beta$ are learned scale and shift.

### RMSNorm
Root-Mean-Square normalization scales by RMS instead of centering by mean. Simpler and sometimes faster:
$$\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)} \odot g$$
where $\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$ and $g$ is a learned gain.

## Residual Connections

Common pattern: Post-attention and post-FFN blocks include residual (skip) connections:
$$y = x + \mathrm{Block}(x)$$
This helps gradients flow and stabilizes deep networks.

## References

- GeLU paper / formulation: https://arxiv.org/abs/1606.08415
- SwiGLU / Gated activations: https://arxiv.org/abs/2002.05202
- LayerNorm original: https://arxiv.org/abs/1607.06450
- RMSNorm discussion: https://arxiv.org/abs/1910.07467
