---
layout: post
title: "Fourier series and transform"
subtitle: "Tool for solving PDEs and computation in general"
background: '' 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

In this post we give some definitions and points out important facts about one of the most fundamental topics in mathematics, fourier analysis. 

## Fourier series

Consider the operator

$$
\begin{align*}
  \mathcal{F} \colon L^1(\mathbb{T}^n) &\to l^\infty\left(\mathbb{Z}^n\right) \\
  f &\mapsto \mathcal{F} f := \frac{1}{2 \pi} \int_{\mathbb{T}^n} f(\theta) e^{-i k \cdot \theta} d\theta.
\end{align*}
$$


$$\frac{\partial \rho}{\partial t}+ \nabla\cdot\left(\rho \mathbf{u}\right)  = 0$$

$$\frac{ \partial \rho \mathbf{u}}{\partial t} + \nabla\cdot\left(\rho \mathbf{u}\mathbf{u}^T\right) =  \nabla\cdot \overline{\overline \sigma} + \rho\mathbf{F} $$

$$\frac{\partial \rho E}{\partial t} + \nabla\cdot\left(\rho\mathbf{u}E\right) = -\nabla\cdot\left(p\mathbf{u}\right) + \nabla\cdot\left(\overline{\overline\tau}\cdot\mathbf{u}\right) + \nabla\cdot\left(K \nabla T\right) + \rho \dot{q} + \rho \mathbf{F}\cdot\mathbf{u} $$

where the following two relations hold 

$$p = \rho r T$$

$$e = e(p,T)$$

The first three equations refer to the conservation of **mass**, **momentum** and **energy** of a fluid element moving in space and time respectively. And the two remaining equations are related to the **equation of state of a perfect gas** and the **thermodynamics relationship** that relates internal energy with pressure and temperature.

Note that we end up with a system of $$7$$ partial differential equations and $$7$$ dependent variables (on $$\mathbf{x} = x\mathbf{e}_1+y\mathbf{e}_2+z\mathbf{e}_3$$ and $$t$$ since we are using a cartesian set of coordinates) which are:

| ----------- | ----------- |
| Density:      | $$\rho = \rho(\mathbf{x},t)$$       |
| Velocity:   | $$\mathbf{u} = \mathbf{u}(\mathbf{x},t) = u\mathbf{e_1} + v\mathbf{e_2} + w\mathbf{e_3}= u_1\mathbf{e_1} + u_2\mathbf{e_2} + u_3\mathbf{e_3}$$|
| Pressure:   | $$p = p(\mathbf{x},t)$$|
| Internal energy:| $$e = e(\mathbf{x},t)$$ where $$E = e + \frac{\mathbf{u}^2}{2} = e + \frac{\|\mathbf{u}\|^2}{2}$$|
|Temperature:|$$T= T(\mathbf{x},t)$$|

Finally, the following quantities are assumed to be known:

| ----------- | ----------- |
| $$\overline{\overline\sigma}:=$$ | Stress tensor where $$\overline{\overline\sigma} = -p\overline{\overline I} + \overline{\overline\tau}$$ and $$\overline{\overline I}$$ is the unit tensor       |
| $$\overline{\overline\tau}:=$$   | Viscous stress tensor|
| $$\mathbf{F}:=$$| External volume forces where $$\mathbf{F} = F_x\mathbf{e_1} + F_y\mathbf{e_2} + F_z\mathbf{e_3}$$|
|$$K:=$$|Thermal conductivity|
|$$\dot{q}:=$$|Volumetric heat sources|
|$$r:=$$|Specific gas constant where $$r = k/M$$, $$k=8314\frac{J}{Kg.mol.K}$$, and $$M=$$ molar mass of gas|


## Terminology

*Compressibility* shows up in the equations as an explicit dependence of density on space and time, and real effects of this might appear depending on the Mach number [regimes](https://en.wikipedia.org/wiki/Mach_number). *Unsteadiness* is the time dependence of the unknowns in the equations. And viscosity refers in the equations to the viscous stress tensor $$\tau$$ term. 

## Derivation 

The 3D Navier-Stokes equations follow from expressing the conservation of [**mass**](https://en.wikipedia.org/wiki/Conservation_of_mass), [**momentum**](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/newton-second-law#:~:text=Newton's%20second%20law%20states%20that%20the%20rate%20of%20change%20of,is%20equated%20to%20the%20forces.),and [**energy**](https://en.wikipedia.org/wiki/Conservation_of_energy) in a differential form for a fluid element that moves in space and time. Normally, you end up from this process with an integral form of the equations and finally you apply [Reynolds transport theorem](https://en.wikipedia.org/wiki/Reynolds_transport_theorem) to express them in the differential [conservative](https://en.wikipedia.org/wiki/Conservation_form) form expressed in this post.


## Further reading

- Chapters 1 and 2 of Hirsch's [book](https://www.amazon.com/Numerical-Computation-Internal-External-Flows/dp/0750665947) are a superb explanation for the derivation and context of these set of equations, as well as the computational aspects and numerical schemes used nowdays to simulate fluids. This was the main source used to write this post.

- NASA has a beautiful summary on physics concepts related to [Aerodynamics](https://www.grc.nasa.gov/www/k-12/airplane/short.html) including the equations here studied.

- A classical book that covers topics on fluid dynamics is Anderson's [book](https://www.amazon.com/Modern-Compressible-Flow-Historical-Perspective/dp/0072424435).