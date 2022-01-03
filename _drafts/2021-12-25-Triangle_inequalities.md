---
layout: post
title: "Triangle Inequality and its reverse"
subtitle: "Useful bounds for the rest of complex analysis."
background: 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Let's talk about two useful inequalities in Complex Analysis an their proofs.

## Triangle inequality
Let $$\{z,w\}\subseteq\mathbb{C}$$ then 
$$
|z+w| \leq |z| + |w|.
$$

Proof: 

Let $$\{z,w\}\subseteq\mathbb{C}$$, then it is enough to prove that 
$$ 
|z+w|^2 \leq ( |z| + |w| )^2 
$$
since it is just a matter of taking the square root on both sides and the inequality will hold since $$f(x) = \sqrt{x}$$ is an increasing function. 

Thus,

$$
\begin{aligned} 
|z+w|^2 & = (z+w)\overline{(z+w)} \hspace{4cm} &\text{(1)}\\ 
        & = (z+w)(\overline{z}+\overline{w}) \hspace{4cm} &\text{(2)}\\ 
        & = z\bar{z} + z\bar{w} + w\bar{z} + w\bar{w} \hspace{4cm} &\text{(3)}\\ 
        & = |z|^2 + z\bar{w} + \overline{z\bar{w}} + |w|^2 \hspace{4cm} &\text{(4)}\\
        & = |z|^2 + 2Re(z\bar{w}) + |w|^2 \hspace{4cm} &\text{(5)}\\
        & \leq |z|^2 + 2|z\bar{w}| + |w|^2 \hspace{4cm} &\text{(6)}\\
        & = |z|^2 + 2|z||w| + |w|^2 \hspace{4cm} &\text{(7)}\\
        & = (|z|+|w|)^2 \hspace{4cm} &\text{(8)}.\\
\end{aligned} 
$$

Finally, $$ 
|z+w| \leq  |z| + |w|.
$$

Note: In addition, if we replace $$w$$ by $$-w$$ then we have also that $$ 
|z-w| \leq  |z| + |w|.
$$

## Reverse Triangle Inequality

Let $$\{z,w\}\subseteq\mathbb{C}$$ then 
$$
|z+w| \geq ||z| - |w||.
$$

Proof:

Notice that we need to prove that 
- $$i.$$ $$
|z+w| \leq |z| - |w|.
$$