---
layout: post
title: "Representer theorem"
subtitle: "How general can we state the theorem?"
background: 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

A powerful set of methods in machine learning arise from the use of kernels. One of the main results from these include the so-called *representer theorem* which allows us make tracktacle the inference of a desired approximation function. In this post we will discuss the math behind the proofs of the representer theorem and its generalizations.

## Classical representer theorem

Let $$K$$ be a reproducing kernel, and denote by $$\mathcal{H}_K$$ its [RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space). Suppose $$\mathcal{X}$$ be a non-empty set and that we are given:

- Data set: $$\left\{(x_i,y_i)\right\}_{i=1}^{N}\subset \mathcal{X} \times \mathbb{R}$$ where $$d\geq 1$$ and $$N > 1$$. 

- Cost function: $$L :(\mathbb{R}^d \times \mathbb{R} \times \mathbb{R})^N \to \mathbb{R} \cup \{\infty\}$$.

- Strictly monotonically increasing function: $$g : [0,\infty) \to \mathbb{R}$$.

then any element from the set

$$
\underset{f\in\mathcal{H}_K}{\text{argmin }} L\left(\left(x_i,y_i,f(x_i)\right)_{i=1}^N\right) + g(\|f\|_{\mathcal{H}_K})  
$$

is of the form 

$$f(\cdot) = \sum_{i=1}^N \beta_i K(\cdot,x_i) $$ 

where $$\beta \in \mathbb{R}^N$$.

*Proof:* Since the set $$H_0 = \text{span}\{K(\cdot,x_1),\dots,K(\cdot,x_N)\}$$ is isomorphic to $$\mathbb{R}^N$$ (which is complete), then it is a closed subspace of $$\mathcal{H}_K$$ so 

$$
\mathcal{H}_K = H_0 \oplus H_0^\perp.
$$

Thus, if $$f\in \mathcal{H}_K$$ then there exists $$\beta \in \mathbb{R}^N$$ and $$w \in H_0^\perp$$ such that

$$
f(\cdot) = \sum_{i=1}^N \beta_i K(\cdot,x_i) + w(\cdot).
$$

Notice that for any $$j\in\{1,\dots,N\}$$ we have

$$

f(x_j) = \sum_{i=1}^N \beta_i K(x_j,x_i) + w(x_j) = \sum_{i=1}^N \beta_i K(x_j,x_i)  + \langle w,K(x_j,\cdot) \rangle = \sum_{i=1}^N \beta_i K(x_j,x_i) 
$$

which means that when we evaluate $$f$$ at any of the training points we only care about the first part, i.e., $$f(x_j)$$ does not depend on evaluating $$w$$ at $x_j$, and we conclude that $$L\left(\left(x_i,y_i,f(x_i)\right)_{i=1}^N\right)$$ does not depend on $$w(x_j)$$. On the other hand, notice that

$$
\begin{align*}
g(\|f\|_{\mathcal{H}_K}) &= g(\|\sum_{i=1}^N \beta_i K(\cdot,x_i) + w(\cdot)\|_{\mathcal{H}_K}) \\
&= g((\|\sum_{i=1}^N \beta_i K(\cdot,x_i) + w(\cdot)\|_{\mathcal{H}_K}^2)^{1/2}) \\
&= g((\|\sum_{i=1}^N \beta_i K(\cdot,x_i)\|_{\mathcal{H}_K}^2 + \|w(\cdot)\|_{\mathcal{H}_K}^2 + 2 \langle \sum_{i=1}^N \beta_i K(\cdot,x_i), w(\cdot) \rangle)^{1/2}) \\
&= g((\|\sum_{i=1}^N \beta_i K(\cdot,x_i)\|_{\mathcal{H}_K}^2 + \|w(\cdot)\|_{\mathcal{H}_K}^2 )^{1/2}) \\
&\geq g((\|\sum_{i=1}^N \beta_i K(\cdot,x_i)\|_{\mathcal{H}_K}^2 )^{1/2})\\
&= g(\|\sum_{i=1}^N \beta_i K(\cdot,x_i)\|_{\mathcal{H}_K} ) 
\end{align*}
$$

where we have use the properties of $$g$$ being monotonically strictly increasing and $$v \in H_0^\perp$$.

Thus, we just saw that if we choose $$w \equiv 0$$ then we always solve for the variational problem meaning that we must have that any minimizer is of the form 

$$f(\cdot) = \sum_{i=1}^N \beta_i K(\cdot,x_i) $$ 

where $$\beta \in \mathbb{R}^N$$.






## References:

- For the classical result we refer to 

> G. S. Kimeldorf and G. Wahba. Some results on Tchebycheffian spline functions.
J. Math. Anal. Applic., 33:82–95, 1971.

> Schölkopf, Bernhard; Herbrich, Ralf; Smola, Alex J. (2001). "A Generalized Representer Theorem". In Helmbold, David; Williamson, Bob (eds.). Computational Learning Theory. Lecture Notes in Computer Science. Vol. 2111. Berlin, Heidelberg: Springer. pp. 416–426



