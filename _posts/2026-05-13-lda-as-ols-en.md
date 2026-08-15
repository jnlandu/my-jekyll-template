<!-- ---
layout: post
title: "Deriving Linear Discriminant Analysis as a Least Squares Problem"
toc: true
author: Jeremie Mabiala
author_profile: "./assets/static/logo.jpeg"
summary: >-
  We show that Fisher's Linear Discriminant Analysis and Ordinary Least Squares
  regression share the same optimal projection direction when labels are encoded
  appropriately. This equivalence opens the door to iterative solvers — such as
  Conjugate Gradients and LSQR — as drop-in alternatives to the classical
  direct formula w = S_W^{-1}(m_1 - m_2).
tags: [machine-learning, linear-algebra, classification, optimization]
---

## Introduction

Linear Discriminant Analysis (LDA) and Ordinary Least Squares (OLS) regression
look like completely different tools. LDA is a classifier: it finds a projection
that maximally separates two (or more) classes. OLS is a regressor: it fits a
linear function to continuous targets by minimizing squared residuals.

Yet underneath, they solve the same linear system. This post gives a self-contained
derivation of the fact that, for binary classification, the OLS solution with
appropriately encoded class labels yields **exactly the same projection direction**
as Fisher's LDA. Along the way we will see how scatter matrices appear naturally
from the regression normal equations, and why this equivalence matters in practice.



## Fisher's Linear Discriminant Analysis

### Setup

Let $$\{\mathbf{x}_n, c_n\}_{n=1}^{N}$# be a labelled dataset in $\mathbb{R}^d$ with
binary classes $C_1$ and $C_2$, containing $N_1$ and $N_2$ samples respectively
($N_1 + N_2 = N$). Define class means

$$\mathbf{m}_k = \frac{1}{N_k}\sum_{n \in C_k} \mathbf{x}_n, \qquad k = 1, 2,$$

and the overall mean $\mathbf{m} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n = \frac{N_1}{N}\mathbf{m}_1 + \frac{N_2}{N}\mathbf{m}_2$.

### The Fisher Criterion

We seek a direction $\mathbf{w} \in \mathbb{R}^d$ such that the scalar projections
$y_n = \mathbf{w}^{\top}\mathbf{x}_n$ are well-separated between classes.
Fisher formalises this as maximising the ratio

$$J(\mathbf{w}) = \frac{\mathbf{w}^{\top} S_B\, \mathbf{w}}{\mathbf{w}^{\top} S_W\, \mathbf{w}},$$

where the **between-class scatter matrix** is

$$S_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^{\top},$$

and the **within-class scatter matrix** is

$$S_W = \sum_{k=1}^{2}\sum_{n \in C_k}(\mathbf{x}_n - \mathbf{m}_k)(\mathbf{x}_n - \mathbf{m}_k)^{\top}.$$

### The LDA Solution

Using the method of Lagrange multipliers (or by differentiating $J$ and setting
the gradient to zero), the optimal direction satisfies the generalised eigenvalue
problem

$$S_B\,\mathbf{w} = \lambda\, S_W\,\mathbf{w}.$$

Because $S_B\,\mathbf{w}$ always points in the direction of $(\mathbf{m}_1 - \mathbf{m}_2)$,
the solution reduces to

$$\boxed{\mathbf{w}^* \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2).}$$

The classifier projects each point onto $\mathbf{w}^*$ and assigns it to the
class whose projected mean is closest.



## Ordinary Least Squares: a Regression Perspective

### Encoding Class Labels

The key insight is to replace the discrete labels $c_n \in \{C_1, C_2\}$ with
real-valued targets

$$t_n = \begin{cases} \dfrac{N}{N_1} & \text{if } \mathbf{x}_n \in C_1, \\[6pt] -\dfrac{N}{N_2} & \text{if } \mathbf{x}_n \in C_2. \end{cases}$$

This encoding is not arbitrary. A key property is that the targets are **zero-mean**:

$$\sum_{n=1}^{N} t_n = N_1 \cdot \frac{N}{N_1} + N_2 \cdot \left(-\frac{N}{N_2}\right) = N - N = 0.$$

### The OLS Objective

We fit an affine model $\hat{y}_n = \mathbf{w}^{\top}\mathbf{x}_n + w_0$ by
minimising the sum of squared residuals:

$$E(\mathbf{w}, w_0) = \frac{1}{2}\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right)^2.$$



## Deriving the Normal Equations

### Solving for the Bias $w_0$

Setting $\partial E / \partial w_0 = 0$:

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right) = 0.$$

Rearranging and using $\sum_n t_n = 0$ and $\sum_n \mathbf{x}_n = N\mathbf{m}$:

$$N w_0 + \mathbf{w}^{\top}(N\mathbf{m}) = 0 \implies \boxed{w_0 = -\mathbf{w}^{\top}\mathbf{m}.}$$

### Solving for the Weight Vector $\mathbf{w}$

Setting $\partial E / \partial \mathbf{w} = 0$:

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right)\mathbf{x}_n = \mathbf{0}.$$

Substituting $w_0 = -\mathbf{w}^{\top}\mathbf{m}$:

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}(\mathbf{x}_n - \mathbf{m}) - t_n\right)\mathbf{x}_n = \mathbf{0}.$$

Expanding:

$$\left(\sum_{n=1}^{N}\mathbf{x}_n\mathbf{x}_n^{\top}\right)\mathbf{w}
  - N\mathbf{m}\mathbf{m}^{\top}\mathbf{w}
  = \sum_{n=1}^{N} t_n \mathbf{x}_n.$$

The matrix on the left is the **total scatter matrix**:

$$S_T = \sum_{n=1}^{N}(\mathbf{x}_n - \mathbf{m})(\mathbf{x}_n - \mathbf{m})^{\top}
     = \sum_{n=1}^{N}\mathbf{x}_n\mathbf{x}_n^{\top} - N\mathbf{m}\mathbf{m}^{\top}.$$

On the right, computing $\sum_n t_n \mathbf{x}_n$:

$$\sum_{n=1}^{N} t_n \mathbf{x}_n
  = \frac{N}{N_1}\sum_{n \in C_1}\mathbf{x}_n - \frac{N}{N_2}\sum_{n \in C_2}\mathbf{x}_n
  = \frac{N}{N_1}\cdot N_1\mathbf{m}_1 - \frac{N}{N_2}\cdot N_2\mathbf{m}_2
  = N(\mathbf{m}_1 - \mathbf{m}_2).$$

So the normal equation becomes:

$$\boxed{S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2).}$$



## Connecting $S_T$ to $S_W$ and $S_B$

The total scatter matrix decomposes as

$$S_T = S_W + S_B^{(\text{full})},$$

where the **full between-class scatter** is

$$S_B^{(\text{full})} = \sum_{k=1}^{2} N_k(\mathbf{m}_k - \mathbf{m})(\mathbf{m}_k - \mathbf{m})^{\top}.$$

We now show that $S_B^{(\text{full})}\,\mathbf{w}$ is proportional to
$(\mathbf{m}_1 - \mathbf{m}_2)$ and hence does not change the *direction* of $\mathbf{w}$.

Note that $\mathbf{m} = \frac{N_1}{N}\mathbf{m}_1 + \frac{N_2}{N}\mathbf{m}_2$, so

$$\mathbf{m}_1 - \mathbf{m} = \frac{N_2}{N}(\mathbf{m}_1 - \mathbf{m}_2),
  \qquad \mathbf{m}_2 - \mathbf{m} = -\frac{N_1}{N}(\mathbf{m}_1 - \mathbf{m}_2).$$

Therefore:

$$S_B^{(\text{full})}\,\mathbf{w}
  = N_1\frac{N_2^2}{N^2}\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right]
  + N_2\frac{N_1^2}{N^2}\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right].$$

Collecting terms:

$$S_B^{(\text{full})}\,\mathbf{w}
  = \frac{N_1 N_2}{N^2}(N_1 + N_2)\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right]
  = \frac{N_1 N_2}{N}\,S_B\,\mathbf{w}.$$

So $S_B^{(\text{full})}\,\mathbf{w}$ is indeed in the direction of
$(\mathbf{m}_1 - \mathbf{m}_2)$ (or $S_B\,\mathbf{w}$). Let $\alpha$ denote the
scalar $\frac{N_1 N_2}{N}(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}$. Then

$$S_T\,\mathbf{w} = S_W\,\mathbf{w} + \alpha\,(\mathbf{m}_1 - \mathbf{m}_2).$$

Substituting back into the normal equation:

$$S_W\,\mathbf{w} + \alpha\,(\mathbf{m}_1 - \mathbf{m}_2) = N(\mathbf{m}_1 - \mathbf{m}_2).$$

Rearranging:

$$S_W\,\mathbf{w} = (N - \alpha)(\mathbf{m}_1 - \mathbf{m}_2).$$

The scalar $(N - \alpha)$ is just a positive constant that does not affect the
direction of $\mathbf{w}$, so:

$$\mathbf{w} \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2).$$

This is precisely the Fisher LDA solution.



## The Equivalence Theorem

> **Theorem.** For binary classification with $N_1$ samples in $C_1$ and $N_2$
> samples in $C_2$, let target values be $t_n = N/N_1$ for $C_1$ and
> $t_n = -N/N_2$ for $C_2$. The weight vector $\mathbf{w}^*$ that minimises
> $E(\mathbf{w}, w_0) = \frac{1}{2}\|\mathbf{X}\mathbf{w} + w_0\mathbf{1} - \mathbf{t}\|^2$
> satisfies $\mathbf{w}^* \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$,
> which is also the Fisher LDA direction.

The proof is the derivation in Sections 3 and 4. Note that the two methods agree
on the **projection direction** but not on the intercept or the scale of
$\mathbf{w}$: OLS gives a specific affine model while LDA gives a direction for
a nearest-mean classifier.



## Matrix Form and Iterative Solvers

### Compact Normal Equation

Writing $\mathbf{X} \in \mathbb{R}^{N \times d}$ for the data matrix (rows are
samples) and $\mathbf{t} \in \mathbb{R}^N$ for the target vector, the OLS
problem is

$$\min_{\mathbf{w}, w_0}\; \frac{1}{2}\|\tilde{\mathbf{X}}\boldsymbol{\beta} - \mathbf{t}\|^2,
\qquad \tilde{\mathbf{X}} = [\mathbf{X} \;\; \mathbf{1}],
\quad \boldsymbol{\beta} = \begin{pmatrix}\mathbf{w} \\ w_0\end{pmatrix}.$$

The normal equation is

$$\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}\,\boldsymbol{\beta} = \tilde{\mathbf{X}}^{\top}\mathbf{t}.$$

### Why This Matters: the Door to Iterative Methods

The LDA formula $\mathbf{w} = S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$ is usually
solved by factorising $S_W$ (e.g., Cholesky decomposition), which costs
$O(d^3)$. In high dimensions, this is expensive and sometimes numerically
fragile.

The OLS formulation restates the same problem as a standard least-squares
system, which can be solved with:

| Method | Cost per iteration | Best suited for |
|--------|-------------------|-----------------|
| Cholesky (direct) | $O(d^3)$ one-time | small $d$ |
| **Conjugate Gradients (CG)** | $O(Nd)$ | large sparse $S_W$ |
| **LSQR** | $O(Nd)$ | rectangular systems, numerical stability |
| **Randomised SVD** | $O(Nd\,k)$ | approximate low-rank cases |

For large-scale problems where $d$ (feature dimension) is in the thousands, an
iterative solver like **LSQR** or **CG applied to $S_T$** can converge in far
fewer than $d$ iterations when the scatter matrix has rapidly decaying
eigenvalues — a common situation in practice.

### The CG Approach Explicitly

Applying CG to the system $S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2)$:

1. Initialise $\mathbf{w}_0 = \mathbf{0}$, residual $\mathbf{r}_0 = N(\mathbf{m}_1 - \mathbf{m}_2)$, direction $\mathbf{p}_0 = \mathbf{r}_0$.
2. For $k = 0, 1, 2, \ldots$ until convergence:

$$\alpha_k = \frac{\mathbf{r}_k^{\top}\mathbf{r}_k}{\mathbf{p}_k^{\top} S_T\,\mathbf{p}_k},
\qquad \mathbf{w}_{k+1} = \mathbf{w}_k + \alpha_k\,\mathbf{p}_k,$$

$$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k\, S_T\,\mathbf{p}_k,
\qquad \beta_k = \frac{\mathbf{r}_{k+1}^{\top}\mathbf{r}_{k+1}}{\mathbf{r}_k^{\top}\mathbf{r}_k},
\qquad \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k\,\mathbf{p}_k.$$

Each matrix-vector product $S_T\,\mathbf{p}$ can be computed implicitly from
the data as $S_T\,\mathbf{p} = \mathbf{X}_c^{\top}(\mathbf{X}_c\,\mathbf{p})$,
where $\mathbf{X}_c$ is the mean-centred data matrix. This keeps the memory
footprint at $O(Nd)$ rather than $O(d^2)$.



## Numerical Example (Sketch)

Suppose $d = 2$, $N_1 = N_2 = 50$, and

$$S_W = \begin{pmatrix}2 & 1\\ 1 & 3\end{pmatrix},
\qquad \mathbf{m}_1 - \mathbf{m}_2 = \begin{pmatrix}1 \\ 0\end{pmatrix}.$$

**LDA (direct):**

$$\mathbf{w}^* = S_W^{-1}\begin{pmatrix}1\\0\end{pmatrix}
  = \frac{1}{5}\begin{pmatrix}3 & -1\\ -1 & 2\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix}
  = \frac{1}{5}\begin{pmatrix}3\\ -1\end{pmatrix}.$$

**OLS (via normal equation):**  With balanced classes ($N_1 = N_2$), the target
encoding simplifies to $t_n = +1$ for $C_1$ and $t_n = -1$ for $C_2$. Solving
$S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2)$ and using $S_T = S_W + S_B$
(both computable from the data) recovers the same direction $\propto (3, -1)^{\top}$.



## Summary

The equivalence established here can be stated cleanly:

> **LDA and OLS share the same optimal projection direction.**

Specifically, if we encode class memberships as $t_n = N/N_k$ (positive for one
class, negative for the other), the weight vector from OLS regression satisfies
exactly the same linear system as Fisher's discriminant. The derivation passes
through the decomposition $S_T = S_W + S_B$ and the observation that
$S_B\,\mathbf{w}$ is always collinear with the class-mean difference
$(\mathbf{m}_1 - \mathbf{m}_2)$.

This is more than a mathematical curiosity:

- It gives a **new algorithm** for LDA: solve the OLS normal equation with any
  method you like — direct or iterative.
- It enables **regularised LDA**: add a ridge term $\lambda I$ to $S_T$, which
  corresponds to $\ell_2$-regularised regression, avoiding singularity issues
  when $d > N$.
- It paves the way for **kernel LDA** by kernelising the regression problem
  instead of kernelising Fisher's criterion directly.



## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 4.1.5. Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Section 4.3. Springer.
3. Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification*, Chapter 3. Wiley.
4. Paige, C. C. & Saunders, M. A. (1982). LSQR: An algorithm for sparse linear equations and sparse least squares. *ACM Transactions on Mathematical Software*, 8(1), 43–71. -->
