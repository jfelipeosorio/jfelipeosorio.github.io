---
layout: post
title: "Representer theorems"
subtitle: "Different setups"
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

which means that when we evaluate $$f$$ at any of the training points we only care about the first part, i.e., $$f(x_j)$$ does not depend on evaluating $$w$$ at $$x_j$$, and we conclude that $$L\left(\left(x_i,y_i,f(x_i)\right)_{i=1}^N\right)$$ does not depend on $$w(x_j)$$. On the other hand, notice that

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

where we have use the properties of $$g$$ being monotonically strictly increasing and $$w \in H_0^\perp$$.

Thus, we just saw that if we choose $$w \equiv 0$$ then we always solve for the variational problem meaning that we must have that any minimizer is of the form 

$$f(\cdot) = \sum_{i=1}^N \beta_i K(\cdot,x_i) $$ 

where $$\beta \in \mathbb{R}^N$$.

## Interpolation version (discrete)

We usually assume that our pair of observations $$\{(x_i,y_i)\}_{i=1}^N$$ have the following functional dependency 

$$
y_i = f^\dagger(x_i) \quad \text{ for all } \quad i \in \{1,\dots,N\}
$$

so that we would like to find an approximation to $$f^\dagger$$ by solving

$$
\underset{f\in\mathcal{H}_K}{\text{argmin }} \|f\|_{\mathcal{H}_K} \quad \text{ s.t. } \quad f(x_i) = f^\dagger(x_i) \quad \forall i\in\{1,\dots,N\}
$$

whose solution according to the classical result is of the form

$$
\hat f(x) = \sum_{i=1}^N K(x,x_i) \beta_i
$$

for some $$\beta \in \mathbb{R}^N$$ that satisfies the linear system

$$
\sum_{i=1}^N K(x_i,x_j) \beta_i = y_i \quad \text{for all}\quad j \in \{1,\dots,N\}.
$$

To see the last statement about $$\beta$$, we just need to remember that $$\hat f$$ satisfies the interpolation constraints.

## Interpolation version (continuous)

We have access to $$\{(x,y):x\in \Omega \text{ and } y \in \mathbb{R}\} \subset \mathcal{X} \times \mathbb{R}$$ and we assume they follow the functional dependency 

$$
y = f^\dagger(x) \quad \text{ for all } \quad x \in \Omega
$$

so that we would like to find an approximation to $$f^\dagger$$ by solving

$$
\underset{f\in\mathcal{H}_K}{\text{argmin }} \|f\|_{\mathcal{H}_K} \quad \text{ s.t. } \quad f(x) = f^\dagger(x)\quad \forall x\in\Omega
$$

whose solution (to be proven) is of the form

$$
\hat f(x) = \int_{\Omega} K(x,y) \beta(y) dy
$$

for some function $$\beta \in \mathbb{R}^\Omega$$ that satisfies the integral equation

$$
\int_{\Omega} K(x,y) \beta(y) dy = f^\dagger(x) \quad \text{for all}\quad x \in \Omega.
$$

*Proof*: Since the set

$$H_0 = \left\{h \in \mathcal{H}_K| \exists \beta \in \mathbb{R}^{\Omega} : h(\cdot) = \int_\Omega K(\cdot,x)\beta(x)dx\right\}$$

is a closed (because of weak convergence) subspace (since $$K(\cdot,x) \in \mathcal{H}_K$$) of $$\mathcal{H}_K$$ so 

$$
\mathcal{H}_K = H_0 \oplus H_0^\perp.
$$

Thus, if $$f\in \mathcal{H}_K$$ then there exists $$\beta(y) \in \mathbb{R}^\Omega$$ and $$w \in H_0^\perp$$ such that

$$
f(\cdot) = \int_\Omega K(\cdot,x)\beta(x)dx + w(\cdot).
$$

Notice that for any $$y \in \Omega$$ we have

$$

f(y) = \int_\Omega K(y,x)\beta(x)dx + w(y) = \int_\Omega K(y,x)\beta(x)dx  + \langle w,K(y,\cdot) \rangle = \int_\Omega K(y,x)\beta(x)dx 
$$

which means that when we evaluate $$f$$ at any of the training points we only care about the first part, i.e., $$f(y)$$ does not depend on evaluating $$w$$ at $$y$$.

On the other hand if we assume that $$\beta$$ is the Radon-Nikodym derivative of an induced measure $$\mu$$ with respect to some base measure(could be Lebesgue for continuous random variables ?), notice that

$$
\begin{align*}
\|f\|_{\mathcal{H}_K} &= \|\int_\Omega K(\cdot,x)\beta(x)dx + w(\cdot)\|_{\mathcal{H}_K} \\
&= (\|\int_\Omega K(\cdot,x)\beta(x)dx + w(\cdot)\|_{\mathcal{H}_K}^2)^{1/2} \\
&= (\|\int_\Omega K(\cdot,x)\beta(x)dx\|_{\mathcal{H}_K}^2 + \|w(\cdot)\|_{\mathcal{H}_K}^2 + 2 \langle \int_\Omega K(\cdot,x)\beta(x)dx, w(\cdot) \rangle )^{1/2} \\
&= (\|\int_\Omega K(\cdot,x)\beta(x)dx\|_{\mathcal{H}_K}^2 + \|w(\cdot)\|_{\mathcal{H}_K}^2  \\
&\geq (\|\int_\Omega K(\cdot,x)\beta(x)dx\|_{\mathcal{H}_K}^2 )^{1/2}\\
&= \|\int_\Omega K(\cdot,x)\beta(x)dx\|_{\mathcal{H}_K} 
\end{align*}
$$

where we have use the properties of $$g$$ being monotonically strictly increasing and $$w \in H_0^\perp$$.

Thus, we just saw that if we choose $$w \equiv 0$$ then we always solve for the variational problem meaning that we must have that any minimizer is of the form 

$$f(\cdot) = \int_\Omega K(\cdot,x)\beta(x)dx = \int_\Omega K(\cdot,x)d\mu(x)$$ 

where $$\beta \in \mathbb{R}^\Omega$$ is the density of a measure $$\mu$$. $$\quad \square$$.

Note: Another approch is studying the approximating power of the set

$$
K(Z):=\overline{\operatorname{span}}(k(x, \cdot), x \in Z)
$$

to the set of continuous functions $$C(Z)$$ with the sup norm as it is done [here](https://thomaszh3.github.io/writeups/RKHS.pdf) for universal kernels.

##  Linear PDE constrained problem (discrete)

Let $$\mathcal{L}$$ be a linear differential operator. We are given pair of observations $$\{(x_i,g(x_i))\}_{i=1}^N$$ having the following functional dependency 

$$
\mathcal{L}f^\dagger(x_i) = g(x_i)\quad \text{ for all } \quad i \in \{1,\dots,N\}
$$

so that we would like to find an approximation to $$f^\dagger$$ by solving

$$
\underset{f\in\mathcal{H}_K}{\text{argmin }} \|f\|_{\mathcal{H}_K} \quad \text{ s.t. } \quad \mathcal{L}f(x_i) = g(x_i) \quad \forall i\in\{1,\dots,N\}
$$

whose solution is of the form

$$
\hat f(x) = \sum_{i=1}^N \mathcal{L}_y K(x,y)\Big|_{y=x_i} \beta_i
$$

for some $$\beta \in \mathbb{R}^N$$ that satisfies the linear system

$$
\mathcal{L}_x \left(\sum_{i=1}^N \mathcal{L}_y K(x,y)\Big|_{y=x_i} \beta_i\right)\Big|_{x=x_j} = g(x_j) \quad \text{for all}\quad j \in \{1,\dots,N\}.
$$

which reduces to

$$
\sum_{i=1}^N \mathcal{L}_x\mathcal{L}_y K(x,y)\Big|_{x=x_j,y=x_i} \beta_i = g(x_j) \quad \text{for all}\quad j \in \{1,\dots,N\}.
$$

##  Linear PDE constrained problem (continuous)

Let $$\mathcal{L}$$ be a linear differential operator. We are given pair of observations $$\{(x,g(x)):x\in \Omega\} \subset \mathcal{X} \times \mathbb{R}$$ having the functional dependency  

$$
\mathcal{L}f^\dagger(x) = g(x)\quad \text{ for all } x \in \Omega
$$

so that we would like to find an approximation to $$f^\dagger$$ by solving

$$
\underset{f\in\mathcal{H}_K}{\text{argmin }} \|f\|_{\mathcal{H}_K} \quad \text{ s.t. } \quad \mathcal{L}f(x) = g(x) \quad \forall x\in \Omega
$$

whose solution is of the form

$$
\hat f(x) = \int_\Omega \mathcal{L}_y K(x,y) \beta(y) dy
$$

for some $$\beta \in \mathbb{R}^\Omega$$ that satisfies the integral equation

$$
\mathcal{L}_x \left(\int_\Omega \mathcal{L}_y K(x,y) \beta(y) dy\right)= g(x) \quad \text{for all}\quad x \in \Omega.
$$

which reduces to

$$
\int_\Omega \mathcal{L}_x\mathcal{L}_y K(x,y) \beta(y) dy = g(x) \quad \text{for all}\quad x \in \Omega.
$$

## References:

- For the classical result we refer to 

> G. S. Kimeldorf and G. Wahba. Some results on Tchebycheffian spline functions.
J. Math. Anal. Applic., 33:82–95, 1971.

> Schölkopf, Bernhard; Herbrich, Ralf; Smola, Alex J. (2001). "A Generalized Representer Theorem". In Helmbold, David; Williamson, Bob (eds.). Computational Learning Theory. Lecture Notes in Computer Science. Vol. 2111. Berlin, Heidelberg: Springer. pp. 416–426



