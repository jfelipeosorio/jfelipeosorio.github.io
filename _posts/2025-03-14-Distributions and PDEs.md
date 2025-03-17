---
layout: post
title: "Distributions, weak derivatives and PDEs"
subtitle: "A conceptual introduction"
background: 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Here we want to revisit some useful concepts such as the weak derivative, a distribution and how they are usually used to construct solutions to PDEs. We will avoid to say that spaces are actually equivalence classes since the sets are partitioned with the relation of being equal almost everywhere and the reader should be careful when equality of functions is written or any identification of a function is made.

## Introduction

Let $$\Omega \subset \mathbb{R}^N$$ be open. We will consider a large space of functions on $$\Omega$$ called **locally integrable** functions

$$
L_{\rm loc}^1(\Omega) := \{v:\Omega \to \mathbb{R} | v \text{ measurable, } \forall K \subset \Omega \text{ compact } v|_K \in L^1(K) \}
$$ 

and by large here we mean that $$L_{\rm loc}^1(\Omega)$$ contains for example $$L^p(\Omega)$$ for all $$1 \leq p \leq \infty$$ or $$C^k(\Omega)$$ for any $$k\geq 1$$. This space of (possibly rough) functions will provide enough generality to define a weaker version of derivatives for any of its elements later.

Then we define the space of **test** functions

$$
D(\Omega) := \{\varphi:\Omega \to \mathbb{R} | \varphi \in C^\infty(\Omega),\operatorname{supp}(\varphi) \text{ compact subset of } \Omega \},
$$ 

which will provide a space of nicely behaved functions where we can take derivatives of any order we might need. For later purposes, $$D(\Omega)$$ is endowed with an inductive limit topology which makes it a locally convex [TVS](https://en.wikipedia.org/wiki/Locally_convex_topological_vector_space). In this topology, a sequence $$\left(\varphi_k\right)_{k=1}^\infty \subset D(\Omega)$$ converges to $$\varphi \in D(\Omega)$$ if and only if

$$
\exists K \subset \Omega \text{ compact}, \forall k\geq 1 \operatorname{supp}(\varphi_k) \subset K, \operatorname{supp}(\varphi) \subset K, \forall \alpha \text{ with } \vert \alpha \vert \geq 0 :
$$

$$
\sup _{x \in K}\left|\partial^\alpha \varphi_k(x)-\partial^\alpha \varphi(x)\right| \rightarrow 0 \text { as } k \rightarrow \infty.
$$

For intuition, locally integrable functions play the role of functions whose classical (strong) derivatives might not exist for every element, so we will first extend the notion to a more general notion of derivative called the weak derivative. To do this recall on the following property that usual differentiable functions satisfy:

**Proposition 1**. Let $$k\geq 1$$ and $$v \in C^k(\Omega)$$. Then for any [multi-index](https://en.wikipedia.org/wiki/Multi-index_notation)  $$\alpha$$ with order $$ \vert \alpha \vert \leq k $$,

$$
\int_{\Omega}\left(\partial^\alpha v\right) \varphi \mathrm{d} x=(-1)^{|\alpha|} \int_{\Omega} v \partial^\alpha \varphi \mathrm{d} x, \qquad \forall \varphi \in D(\Omega).
$$

This can be seen as an integration by parts formula without boundary terms due to the fact that $$\varphi \in D(\Omega)$$. 

## Weak derivative

We could then suggest $$v^\alpha$$, a function that satisfies  

$$
\int_{\Omega} v^\alpha \varphi \mathrm{d} x=(-1)^{|\alpha|} \int_{\Omega} v \partial^\alpha \varphi \mathrm{d} x, \qquad \forall \varphi \in D(\Omega),
$$

to be a **weak derivative** of order $$\vert \alpha \vert$$ for a given $$v \in L_{\rm loc}^1(\Omega)$$. But wait, hang on:

- Does this notion of weak derivative coincides with the classical notion of derivative if $$v \in C^k(\Omega)$$ ?
- Is it actually unique for a given $$v \in L_{\rm loc}^1(\Omega)$$?

Fortunally, the answer to these questions is yes. This can be proven by using an important property of functions in $$v \in L_{\rm loc}^1(\Omega)$$:

**Fundamental lemma of the calculus of variations**. Let $$v \in L_{\rm loc}^1(\Omega)$$ be such that

$$
\int_{\Omega} v \varphi \mathrm{~d} x=0, \qquad \forall \varphi \in D(\Omega).
$$

Then $$v  = 0$$.


Now we are sure about two things: first, there is no ambiguity of refering to **the** weak derivative of a function in $$ v \in L_{\rm loc}^1(\Omega)$$ and this notion indeed generalizes the strong derivative in the case where $$ v \in C^k(\Omega) \subset L_{\rm loc}^1(\Omega)$$. Thus, notation-wise we are safe to write the symbol

$$
\partial^\alpha v
$$

to denote classical and weak derivatives for $$v \in L_{\rm loc}^1(\Omega)$$.


## Distributions

Here we introduce a new set of objects called **distributions** which will serve as a generalization of usual functions, for the sake of concretness functions in $$L_{\rm loc}^1(\Omega)$$; ergo distributions are also referred to as *generalized functions*.

**Definition 1.** The set of distributions is defined as the dual space of $$D(\Omega)$$, i.e.,

$$
D'(\Omega) := \{T: D(\Omega) \to \mathbb{R} |  T \text{ is linear and continuous} \}.
$$ 

Here "continuity" is understood with respect to the inductive limit topology we previously defined for $$D(\Omega)$$. We naturally equip $$D'(\Omega)$$ with a weak$$^*$$-like topology. In this topology, a sequence $$\left(T^k\right)_{k=1}^\infty \subset D'(\Omega)$$ converges to $$T \in D'(\Omega)$$ if and only if

$$
T^k(\varphi) \to T(\varphi) \quad \text{ as } \quad k \to \infty, \qquad \forall \varphi \in D(\Omega).
$$

In this case, the sequence $$\left(T^k\right)_{k=1}^\infty$$ is said to converge to $$T$$ in the sense of distributions, denoted by

$$
T^k \to T \quad \text{ as } \quad k \to \infty,  \text{ in } D'(\Omega).
$$

An equivalent **definition** of a distribution is to be a linear functional $$T : D(\Omega) \to \mathbb{R}$$ such that

$$
\forall K\subset \Omega \text{ compact}, \exists C(K), \exists m(K) \in \mathbb{N},\text{ s.t. } \forall \varphi \in D{\Omega} \text{ with } \operatorname{supp}(\varphi) \subset K:
$$

$$
|T(\varphi)| \leq C(K) \sup\{\left|\partial^\alpha \varphi(x)\right|: |\alpha| \leq m(K), x \in K\} 
$$

### Why distributions are a generalization of functions ?

Let $$v \in L_{\rm loc}^1(\Omega)$$ then it can be proved (using the equivalent definition of a distribution) that the associated linear functional

$$
T_v : \varphi \in D(\Omega) \to T_v(\varphi) := \int_{\Omega} v \varphi dx
$$

defines a distribution on $$\Omega$$. The distribution $$T_v$$ is called the distribution associated with the function (locally integrable) $$v$$. Any distribution defined by a locally integrable function is said to be a **regular** distribution.

So far we have identified functions with functionals in $$D'(\Omega)$$. What about the converse? For any given distribution $$T \in D'(\Omega)$$ can we find $$v \in L_{\rm loc}^1(\Omega)$$ such that $$T = T_v$$ ? 

No, and a famous example of this fact, is given by the **Dirac distribution** $$\delta_a$$ (can be proved it is indeed a distribution, and usuallly $$\delta := \delta_0$$) for $$a \in \Omega$$ defined by

$$
\delta_a: \varphi \in D(\Omega) \to \delta_a(\varphi):= \varphi(a).
$$

Thus, in general we can uniquely identify functions with elements in $$D'(\Omega)$$ and also there are elements in $$D'(\Omega)$$ which cannot be identified with a function. Giving us at least an injection of usual functions in a set which has more elements. 

One may ask now, is the vector space structure of the addition and scalar multiplication in $$L_{\rm loc}^1(\Omega)$$ still compatible with the vector space structure in $$D'(\Omega)$$?

The answer is positive. Let's investigate how this is done for the addition. By definition, addition for any $$ \{T,S\} \subset D'(\Omega)$$ is given by

$$
(T + S) (\varphi) : \varphi \in D(\Omega) \to (T + S) (\varphi) := T (\varphi) + S (\varphi).
$$

Notice that if $$\{v,w\}\subset L_{\rm loc}^1(\Omega)$$ then 

$$
\begin{align*}
(T_v + T_w) (\varphi) &= T_v (\varphi) + T_w (\varphi) \\
&= \int_\Omega v \varphi dx + \int_\Omega w \varphi dx \\
&= \int_\Omega (v+w) \varphi dx \\
&=: T_{v+w}(\varphi)
\end{align*}
$$

for any $$\varphi \in D(\Omega)$$. Similarly it can be proven for the scalar multiplication that 

$$
(\lambda T_v) (\varphi) = T_{\lambda v} (\varphi)
$$

for all scalars $$\lambda$$ where the reader might be aware on the diferent scalar multiplications $$\lambda T_v$$ and $$\lambda v$$.

Can we additionally, ask for compatibility of derivatives as well ? Yes.

## Derivative in the sense of distributions
Let $$T$$ be a distribution on $$\Omega$$ (of course $$T \in D'(\Omega)$$, but intentionally, we stated it in that way since for the case where $$T = T_v$$ we match the way we refer to a function: *let $$v$$ be a function on $$\Omega$$*), and let $$\alpha$$ be a multi-index with $$\vert \alpha \vert \geq 1$$. Then the linear functional defined by 

$$
\partial^\alpha T: \varphi \in D(\Omega) \to \partial^\alpha T (\varphi) := (-1)^{\vert\alpha\vert} T(\partial^\alpha \varphi)
$$

is again a distribution on $$\Omega$$, called the **partial derivative of order $$\alpha$$ of $$T$$ in the sense of distributions**.

Is this definition compatible with the derivative of any $$v \in L_{\rm loc}^1(\Omega)$$ ? In other words, is it true that $$\partial^\alpha T_v = T_{\partial^\alpha v}$$ ? Yes, let's look at this argument: For any $$\varphi \in D(\Omega)$$

$$
\begin{align*}
\partial^\alpha T_v (\varphi) &:= (-1)^{\vert\alpha\vert} T_v(\partial^\alpha \varphi)\\
&= (-1)^{\vert\alpha\vert} \int_\Omega v\partial^\alpha \varphi dx\\
&= \int_\Omega \partial^\alpha v \varphi dx\\
&=: T_{\partial^\alpha v}.
\end{align*}
$$

This shows that the weak derivatives are a special cases of derivatives in the sense of distributions. Or in other words, one can see the distributional derivative as a generalization of the weak derivative.

## Linear PDE with constant coefficients

Given a set of multi-indices $$A$$ and coefficients $$a_\alpha \in \mathbb{R}$$ where $$\alpha \in A$$, the linear functional 

$$
\mathcal{L}T: \varphi \in D(\Omega) \to \mathcal{L}T(\varphi):= \sum_{\alpha \in A} (-1)^{\vert \alpha \vert} a_\alpha T(\partial^\alpha \varphi),\qquad \forall \varphi \in D(\Omega)
$$

defines a **linear partial differential operator in the sense of distributions**,

$$
\mathcal{L}: T \in D'(\Omega) \to \mathcal{L}T \in D'(\Omega).
$$

Given any distribution $$f \in D'(\Omega)$$, one may ask if there exists a distribution $$T \in D'(\Omega)$$ that satisfies

$$
\mathcal{L} T = f \quad \text{ in } D'(\Omega).
$$

If such $$T$$ exists then $$T$$ is called a solution of $$\mathcal{L}T=f$$ *in the sense of distributions*. If $$f = \delta_0$$ then $$T$$ is called the fundamental solution of the PDE and the reason for that will be clear once we know that $$$$

Note: Here we are equating distributions and let's just recall that two distributions $$T,S \in D'(\Omega)$$ are equal if and only if 

$$
T(\varphi) = S(\varphi)\quad \forall \varphi\in D(\Omega).
$$

**Example 1.** Let $$\Omega \subset \mathbb{R}^N$$ be open and contains the origin. Let $$\omega_N$$ be the volume of the unit ball in $$\mathbb{R}^N$$. Consider the locally integrable function

$$
v(x) = 
\begin{cases}
\frac{1}{2\omega_2} \log \vert x \vert & \text{ if } x \neq 0, N = 2\\
\frac{1}{N(2-N)\omega_N} \vert x \vert^{2-N} & \text{ if } x \neq 0, N \geq 3\\
\end{cases}
$$

then one can show that 

$$
\Delta T_v = \delta_0 \quad \text{ in } D'(\Omega),
$$

equivalently,

$$
\Delta T_v (\varphi) = \delta_0(\varphi) \quad \forall \varphi \in D(\Omega),
$$

equivalently,

$$
T_{\Delta v} (\varphi) = \varphi(0) \quad \forall \varphi \in D(\Omega),
$$

equivalently,

$$
T_{v} (\Delta \varphi) = \varphi(0) \quad \forall \varphi \in D(\Omega),
$$

equivalently,

$$
\int_\Omega v (\Delta \varphi) dx = \varphi(0) \quad \forall \varphi \in D(\Omega).
$$

Note: The chain of equivalent definitions above has the goal of telling that whenever one reads something like 

$$
\Delta v = \delta_0
$$

where it seems like functions and distributions are being equated and $$\delta_0$$ is being treated as a "function" (**which is not**), it is the case that the functions in the expression, for this case only $$v$$, should be thought instead as its associated regular distribution $$T_v$$. 

## Solutions of PDEs

Consider now the problem,

$$
\Delta u = f
$$

where $$f \in D(\Omega)$$ we can use the fundamental solution of the Poisson equation $$T_v$$ to solve for this problem by proposing the distribution

$$
u = f * T_v
$$ 

where $$*$$ denotes the convolution between the function $$f$$ and the distribution $$T_v$$ defined via the map

$$ 
C : (\varphi, T) \in D(\Omega) \times D'(\Omega) \to C(\varphi, T):=(\varphi * T) \in D'(\Omega)
$$ 

with $$(\varphi * T)(\phi):= T\underbrace{(\tilde\varphi * \phi)}_{\text{usual convolution}}$$ for any $$\phi \in D(\Omega)$$ and $$\tilde \varphi(x) := \varphi(-x)$$.

Before we actually test $$u$$ is a solution, we need to check two key properties of the convolution as defined above:

- For any linear PDE operator $$\mathcal L$$ we have that 

$$
\mathcal{L}(f * T) = \mathcal{L} f * T = f *(\mathcal{L} * T).
$$

- $$(f * \delta) = T_f$$.

Note that the second property is nice in the sense that the delta distribution $$\delta$$ acts like a unitary element under this multiplication defined (not fully in the space of distributions but close enough) as the convolution.

Thus,

$$

\begin{align*}
\Delta u &= f * T_v\\
&= f * \Delta T_v\\
&= f * \delta\\
&= T_f.
\end{align*}
$$


## Notation for distributions

Motivated from the usual dual space elements $$T \in D'(\Omega)$$ being evaluated at a points of its domain $$\varphi \in D(\Omega)$$, one can also find in literature the following notations:

$$
T(\varphi) = [T, \varphi] = \langle T, \varphi  \rangle = _{D'(\Omega)}\langle T,\varphi  \rangle_{D(\Omega)}.
$$

## References:

This summary is made out from these wonderful sources:

> [1] Ciarlet, P. G. (2013). Linear and nonlinear functional analysis with applications. Society for Industrial and Applied Mathematics.

> [2] Distribution theory [series](https://www.youtube.com/watch?v=gwVEEUg8PBY&list=PLBh2i93oe2qsbptdcvFlowCl51EX_a3nB) by [The Bright Side of Mathematics](https://www.youtube.com/@brightsideofmaths).

> [3] Gelfand, I.M., Shilov G.E. (1964). Generalized Functions, Volume I. Properties and Operations. See English translation [here](https://bookstore.ams.org/view?ProductCode=CHELGELFSET).



