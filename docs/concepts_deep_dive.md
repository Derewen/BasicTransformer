# Transformer Arithmetic: Conceptual Deep Dive

This document provides a detailed conceptual and mathematical deep dive into the four pillars of modern Transformer architectures: **Multi-Head Attention**, **Layer Normalization**, **Non-linear Activations**, and **Adaptive Optimizers**.

---

## 1. Causal Multi-Head Attention (MHA)

Multi-Head Attention is the mechanism that allows the model to map dynamic, contextual relationships across the sequence.

### A. The Math of $Q, K, V$ Projections
Why do we project the same input coordinate vector $X$ into three separate representations?
If we did not project them separately, $Q$ and $K$ would be identical, resulting in a symmetric score matrix where token $A$ looks at token $B$ with the exact same strength that token $B$ looks at token $A$. This is incorrect for language. 
* In language and math, relationships are **directional** (asymmetric). The equals sign `=` needs to search for digits (strong query-to-key match), but digits do not need to query the equals sign.
* Having separate projection weights $W_Q, W_K, W_V$ allows the network to learn this directional asymmetry.

### B. Dot-Product Similarity ($Q K^T$)
The dot product of two vectors is a measure of their alignment (similarity). 
* If two vectors point in the same direction in our 128-dimensional space, their dot product is a large positive number.
* If they are orthogonal (90 degrees), their dot product is 0.
* By multiplying the Query matrix $Q$ by the Key matrix $K^T$, we calculate a pairwise dot-product score for every possible pair of tokens in our sequence.

### C. Scaling by $\frac{1}{\sqrt{d_k}}$
As the dimension size of the vectors ($d_k = 32$) grows, the average magnitude of the raw dot products increases. 
* If raw scores are very large (e.g. $+50$ or $-50$), the **Softmax function** (which exponentiates the scores) will output a probability distribution that is extremely polarized: `1.0` for the largest score, and `0.0` for all others.
* This is called **gradient saturation**. When the output is a flat `1.0` or `0.0`, the mathematical derivatives become `0.0`. This stops training completely because the weights cannot update.
* We divide by $\sqrt{32} \approx 5.65$ to scale the scores back down, ensuring the softmax outputs smooth, trainable percentages.

### D. Causal Masking & Softmax
To enforce causality (preventing looking at future answers), we apply a lower-triangular mask where the upper triangle is set to $-\infty$ (`-1e9`).

The Softmax function for an element $x_i$ in a row is defined as:
$$P(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

If we pass $-\infty$ into this equation:
$$e^{-\infty} = 0$$
Thus, the probability $P(x_{\text{future}})$ becomes exactly `0.0`, rendering future tokens completely invisible to the attention mechanism.

---

## 2. Layer Normalization (LayerNorm)

As coordinates travel through deep layers, their values can accumulate and grow out of control. LayerNorm prevents this.

### A. The Mathematics of LayerNorm
For a given token vector $x \in \mathbb{R}^{d}$ (our 128 coordinates), LayerNorm calculates the mean ($\mu$) and variance ($\sigma^2$) **across its 128 features**:
$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

It then normalizes the vector and applies trainable scaling ($\gamma$) and shifting ($\beta$) parameters:
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
* $\epsilon$ is a tiny constant (`1e-5`) to prevent division by zero.
* $\gamma$ and $\beta$ allow the model to learn the optimal scale and offset for coordinates.

### B. Pre-LN vs. Post-LN Stability
* **Post-LN (Old GPT-2 Style):** LayerNorm is placed on the residual path *after* addition. This forces the gradients to travel through a division scale factor at every single block during backpropagation. For deep networks, the gradients at the early layers shrink to zero (vanishing gradients), making the model untrainable without highly complex learning rate warmups.
* **Pre-LN (Our Style / Modern LLMs):** LayerNorm is placed on the active branches *before* attention and MLP. The main residual path ($x_{i+1} = x_i + \text{Layer}(x_i)$) remains a clean addition highway. Gradients can flow backwards through this highway completely unimpeded, allowing stable training of extremely deep networks.

---

## 3. Multi-Layer Perceptrons & Non-linear Activations

Why do we need the MLP block and GELU?

### A. The Necessity of Non-Linearity
If our neural network only consisted of linear projections (`nn.Dense` matrix multiplications), the entire model would collapse mathematically into a single linear equation:
$$Y = W_{\text{final}} \cdot X + B_{\text{final}}$$
No matter how many blocks or layers we stacked, the model would only be able to solve simple linear relationships. It would be completely unable to solve complex arithmetic logic (like carry-overs or checks).

We must inject a **non-linear activation function** between our linear layers to bend the coordinate space.

### B. GELU (Gaussian Error Linear Unit)
Our MLP uses the **GELU** activation function.

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x), \quad \text{where } X \sim \mathcal{N}(0, 1)$$

Approximate formula used in computation:
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)$$

```text
GELU Curve vs. ReLU Curve:
      ReLU (Sharp Corner)                GELU (Smooth Curvature)
             ▲                                      ▲
             │  /                                   │  /
             │ /                                    │ /
             │/                                     │/
  ───────────┼────────►                  ───────────┼────────►
             │                                     /│
             │                                    \_│
```

* **Why GELU is superior to ReLU:** ReLU sets all negative values to exactly `0.0`. This causes "dead neurons"—if a node's output falls below zero, it stops outputting gradients, and it can never recover or learn again.
* GELU has a smooth, curved shape. It allows a tiny negative gradient to leak through for small negative inputs, keeping neurons alive and allowing smoother optimization.

---

## 4. Adaptive Optimization (The Adam Optimizer)

Traditional SGD (Stochastic Gradient Descent) updates all weights using a single global learning rate: $\theta = \theta - \eta \cdot g$. 

In a Transformer, this is highly inefficient. Some features (like the `<eos>` token or rare numbers) appear rarely, so their weights receive very small, infrequent updates. Other features (like digits or `+`) appear constantly and receive huge updates.

We use the **Adam (Adaptive Moment Estimation)** optimizer, which calculates a **unique, dynamic learning rate for every single weight in the model**.

Adam tracks two moving statistics for each weight:
1. **First Moment (Momentum):** The rolling average of the gradients, which keeps the model moving in a consistent direction even if a single batch is noisy.
2. **Second Moment (Adaptive Scaling):** The rolling average of the squared gradients. If a weight receives highly erratic, massive updates, Adam divides its step size by the square root of this variance. 

This guarantees that quiet weights get larger, sensitive steps when they receive gradients, and loud, chaotic weights are scaled down to prevent training from exploding.
