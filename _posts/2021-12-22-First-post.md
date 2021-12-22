---
layout: post
title: "Kernel Density Estimation using Density Matrices"
subtitle: "Quantum mechanics math help us to calculate faster density estimates."
background: '/img/posts/First-post/alldensities_pot3.jpeg'
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Density estimation methods can be used to solve a variety of statistical and machine learning challenges. They can be used to tackle a variety of problems, including anomaly detection, generative models, semi-supervised learning, compression, and text-to-speech.  A popular technique to find density estimates for new samples in a non parametric set up is Kernel Density Estimation (KDE), a method which suffers from costly evaluations especially for large data sets and higher dimensions. In this post we will discuss on the math behind an efficient way to calculate density estimates using density matrices (a concept from quantum mechanics).

## Introduction
One of the principal methodologies for analyzing data is assuming its random nature and modelling its probability behaviour. In many applications we have a finite
set of data and we would like to know what probability distribution generated the
data. From statistical inference this problem has played a central role in research
and has inspired many methods which rely on the use of the density function such
as non parametric regression when non linear patterns are observed. Also in machine learning many approaches to anomaly detection make use of the probability
density function. 

The parametric approach to density estimation given some data $$\mathbf{x}_1,\dots,\mathbf{x}_N$$ assumes that each $$\mathbf{x}_i$$ is sampled independently from a random vector $$\mathbf{X}\sim f(\mathbf{x};\mathbf{\theta})$$ and the theory is developed around building an estimator $$\hat{\mathbf{\theta}}$$ with good statistical properties such us unbiasdness, consistency, efficiency and sufficiency. and finally the density of a new sample $$\mathbf{x}$$ is given by:
\\[
x^2
\\]
## Kernel Density Estimation
If we are given a realization of a random sample $$\{x_1,\dots,x_n\}\subseteq\mathbb{R}$$ and then we want to estimate the probability density (in a non-parametric fashion, i.e., not assuming any underlying distribution for each of the random variables that generated the given points) at a new given point $$x^*\in\mathbb{R}$$ then we can use the *kernel density estimation* at that point which is given by:


## Text
We are gonna have so much fun.

## Math
Centered math
\\[ x = {-b \pm \sqrt{b^2-4ac} \over 2a} \\]
Inline math $$x^2$$.

## Images
![imagen](/img/posts/First-post/alldensities_pot3.jpeg)

# Code
```
a = input()

```



