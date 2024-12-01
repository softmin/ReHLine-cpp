# ReHLine <a href="https://github.com/softmin/ReHLine"><img src="https://raw.githubusercontent.com/softmin/ReHLine/main/images/logo.png" align="right" height="138" /></a>

**ReHLine** is designed to be a computationally efficient and practically useful software package for large-scale empirical risk minimization (ERM) problems.

The **ReHLine** solver has four appealing
"linear properties":

- It applies to any convex piecewise linear-quadratic loss function, including the hinge loss, the check loss, the Huber loss, etc.
- In addition, it supports linear equality and inequality constraints on the parameter vector.
- The optimization algorithm has a provable linear convergence rate.
- The per-iteration computational complexity is linear in the sample size.

This repository, **ReHLine-cpp**, provides efficient C++ code that implements the core algorithm of the **ReHLine** solver.
It is also the foundation of the [Python](https://github.com/softmin/ReHLine-python)
and [R](https://github.com/softmin/ReHLine-r) interfaces to **ReHLine**.

**ReHLine-cpp** is a tiny and header-only library aiming to be fast and easy to use. The whole library is a single
header file [rehline.h](rehline.h), and its only dependency is the
[Eigen](https://eigen.tuxfamily.org) library, which is also header-only.

## 📝 Formulation

**ReHLine** is designed to address the empirical regularized ReLU-ReHU minimization problem, named *ReHLine optimization*, of the following form:

$$
\min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_ i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}}( s_{hi} \mathbf{x}_ i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \Vert \mathbf{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},
$$

where $\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}$ and $\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}$ are the ReLU-ReHU loss parameters, and $(\mathbf{A},\mathbf{b})$ are the constraint parameters.
The ReLU and ReHU functions are defined as $\mathrm{ReLU}(z)=\max(z,0)$ and

$$
\mathrm{ReHU}_\tau(z) =
  \begin{cases}
  \ 0,                     & z \leq 0 \\
  \ z^2/2,                 & 0 < z \leq \tau \\
  \ \tau( z - \tau/2 ),   & z > \tau
  \end{cases}.
$$

This formulation has a wide range of applications spanning various fields, including statistics, machine learning, computational biology, and social studies. Some popular examples include SVMs with fairness constraints (FairSVM), elastic net regularized quantile regression (ElasticQR), and ridge regularized Huber minimization (RidgeHuber).

![](https://raw.githubusercontent.com/softmin/ReHLine/main/images/tab.png)

## ⚖️ License

**ReHLine-cpp** is open source under the MIT license.
