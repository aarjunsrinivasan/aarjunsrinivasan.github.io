---
title: "Is PCA Just SVD? A Geometric and Optimization View from ML to Deep Learning"
date: 2026-01-01
draft: false
tags: ["Machine Learning", "Deep Learning", "Linear Algebra", "PCA", "SVD"]
categories: ["Machine Learning & Linear Algebra"]
readingTime: 15
description: "Exploring the mathematical foundations of PCA and SVD, understanding why they coincide for centered data, and how this relationship plays out from classical machine learning to modern deep learning applications."
math: true
---

Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) are two fundamental techniques in machine learning and data science. The question "Is PCA just SVD?" is often asked, and the answer reveals an important distinction about their nature and relationship.

> **The Answer**
> 
> **Short answer: No.**
> 
> **Long answer:** PCA is a statistical objective; SVD is a numerical linear algebra tool. They meet—but they are not the same thing.

In this post, we'll explore their mathematical foundations, understand why they coincide for centered data, and see how this relationship plays out from classical machine learning to modern deep learning applications.

## 1. Singular Value Decomposition (SVD)

### 1.1 Definition

Singular Value Decomposition is a fundamental matrix factorization technique from numerical linear algebra. For any matrix $A \in \mathbb{R}^{m \times n}$, SVD decomposes it as:

$$A = U \Sigma V^\top$$

where:

- $U \in \mathbb{R}^{m \times m}$ — orthonormal columns (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ — diagonal matrix with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$
- $V \in \mathbb{R}^{n \times n}$ — orthonormal columns (right singular vectors)

The singular values $\sigma_i$ are the square roots of the eigenvalues of both $A^T A$ and $A A^T$.

SVD is a **general matrix decomposition**—it exists for any matrix (real or complex, square or rectangular) which can be proved based on spectral theorem. It's a pure linear algebra tool with no statistical assumptions.

## 2. What Is PCA Really?

### 2.1 PCA Is a Statistical Problem

Unlike SVD, Principal Component Analysis is fundamentally a **statistical technique**. PCA is defined as:

> **PCA Objective:** Find orthogonal directions that maximize the variance of projected data.

This is a statistical optimization problem, not just a matrix factorization. The goal is to find directions in the feature space along which the data exhibits maximum variability.

### 2.2 Data Matrix Setup

Let's set up the problem formally. Consider a data matrix:

$$X \in \mathbb{R}^{m \times n}$$

where:

- **Rows = samples** ($m$ data points)
- **Columns = features** ($n$ features per sample)

Each row $\mathbf{x}_i^T$ represents one data sample, and each column represents one feature across all samples.

### 2.3 Why Center the Data?

This is a crucial step that reveals the statistical nature of PCA. We center the data:

$$\tilde{X} = X - \boldsymbol{\mu}$$

where $\boldsymbol{\mu} = \frac{1}{m}\sum_{i=1}^m \mathbf{x}_i$ is the sample mean vector.

**Why is centering necessary?**

- **PCA measures variance:** Variance is a measure of spread around the mean. By definition, variance is computed relative to the mean.
- **Variance is defined relative to the mean:** For a random variable, $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$. Without centering, we're not measuring true variance.
- **Without centering → first component points to the mean:** If we don't center, the first principal component will be biased toward the direction of the mean vector, not the direction of maximum variance. This ensures PCA captures variance, not mean shift.

More formally, let's derive why centering is necessary. Consider we want to find a direction $\mathbf{w}$ (a unit vector) along which to project our data to maximize variance. Even if we start with uncentered data, when we compute variance properly, we must subtract the mean. The variance of the projections $z_i = \mathbf{x}_i^T \mathbf{w}$ along direction $\mathbf{w}$ is:

$$\text{Var}(z) = \frac{1}{m-1} \sum_{i=1}^m (z_i - \bar{z})^2 = \frac{1}{m-1} \sum_{i=1}^m (\mathbf{x}_i^T \mathbf{w} - \boldsymbol{\mu}^T \mathbf{w})^2$$

Notice that even though we started with uncentered data $\mathbf{x}_i$, the variance formula automatically subtracts the mean of the projections ($\bar{z} = \boldsymbol{\mu}^T \mathbf{w}$). This can be rewritten as:

$$\text{Var}(z) = \frac{1}{m-1} \sum_{i=1}^m [(\mathbf{x}_i - \boldsymbol{\mu})^T \mathbf{w}]^2 = \mathbf{w}^T \left[\frac{1}{m-1} \sum_{i=1}^m (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T\right] \mathbf{w} = \mathbf{w}^T \Sigma_x \mathbf{w}$$

This shows that the variance along direction $\mathbf{w}$ depends on the covariance matrix $\Sigma_x$ of **centered** data $(\mathbf{x}_i - \boldsymbol{\mu})$. The key insight is that variance is always computed relative to the mean—this is built into the definition of variance. Therefore, to maximize variance along direction $\mathbf{w}$, we must work with centered data. Without centering, we'd be optimizing a different objective that mixes variance with the mean location, and the first principal component would point toward the mean rather than the direction of maximum variance.

## 3. Objective: What Are We Maximizing?

### 3.1 Projection Viewpoint (Key Intuition)

The key intuition behind PCA comes from the projection viewpoint. We choose a unit direction $\mathbf{w} \in \mathbb{R}^n$ (where $\|\mathbf{w}\| = 1$) and project all data points onto it:

$$z_i = \mathbf{x}_i^\top \mathbf{w}$$

This gives us a 1D representation of the data. Each high-dimensional point $\mathbf{x}_i$ is mapped to a scalar $z_i$ along the direction $\mathbf{w}$.

**PCA asks:** Along which direction does the projected data have maximum variance?

This is the core statistical question. We want to find the direction that preserves the most information about the data's variability.

### 3.2 Variance of the Projection

For centered data $\tilde{X}$, the variance of the projected data is:

$$\text{Var}(\tilde{X}\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (\tilde{\mathbf{x}}_i^\top \mathbf{w})^2$$

Note: For centered data, the mean of projections is zero, so variance simplifies to the mean of squared projections. This is because $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$, and for centered data $\mathbb{E}[X] = 0$, so $\text{Var}(X) = \mathbb{E}[X^2]$.

Rewriting in matrix form:

$$\text{Var}(\tilde{X}\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (\tilde{\mathbf{x}}_i^\top \mathbf{w})^2 = \frac{1}{m} \mathbf{w}^\top \tilde{X}^\top \tilde{X} \mathbf{w}$$

Define the sample covariance matrix:

$$\Sigma_x = \frac{1}{m} \tilde{X}^\top \tilde{X}$$

So the objective becomes:

$$\boxed{ \max_{\mathbf{w}} \quad \mathbf{w}^\top \Sigma_x \mathbf{w} }$$

### 3.3 Constraint: Why Unit Norm?

Without a constraint, the objective is meaningless. We could simply scale $\mathbf{w}$ to make the variance arbitrarily large:

$$\mathbf{w}^\top \Sigma_x \mathbf{w} \rightarrow \infty \quad \text{by scaling } \mathbf{w}$$

So we impose the constraint:

$$\boxed{ \mathbf{w}^\top \mathbf{w} = 1 }$$

**Geometric meaning:** We are choosing a *direction*, not a magnitude. The constraint ensures we're optimizing over the unit sphere, which makes the problem well-defined.

### 3.4 Full Optimization Problem

Combining the objective and constraint, we have:

$$\boxed{ \max_{\mathbf{w}} \quad \mathbf{w}^\top \Sigma_x \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{w} = 1 }$$

## 4. Solution via Lagrange Multipliers

We solve this constrained optimization problem using the method of Lagrange multipliers.

### 4.1 Step 1: Construct the Lagrangian

The Lagrangian function is:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top \Sigma_x \mathbf{w} - \lambda(\mathbf{w}^\top \mathbf{w} - 1)$$

where $\lambda$ is the Lagrange multiplier associated with the constraint $\mathbf{w}^\top \mathbf{w} = 1$.

### 4.2 Step 2: Take Gradient with Respect to $\mathbf{w}$

Taking the gradient of the Lagrangian with respect to $\mathbf{w}$:

$$\nabla_{\mathbf{w}} \mathcal{L} = 2\Sigma_x \mathbf{w} - 2\lambda \mathbf{w}$$

Setting the gradient to zero:

$$2\Sigma_x \mathbf{w} - 2\lambda \mathbf{w} = 0$$

Dividing by 2 and rearranging:

$$\boxed{ \Sigma_x \mathbf{w} = \lambda \mathbf{w} }$$

> **🔑 Key Result:** The PCA directions are eigenvectors of the covariance matrix $\Sigma_x$. The Lagrange multiplier $\lambda$ is the corresponding eigenvalue.

### 4.3 Step 3: Which Eigenvector?

To determine which eigenvector gives the maximum variance, substitute back into the objective. From the eigenvalue equation:

$$\mathbf{w}^\top \Sigma_x \mathbf{w} = \mathbf{w}^\top (\lambda \mathbf{w}) = \lambda \mathbf{w}^\top \mathbf{w} = \lambda$$

So:

> **Maximizing variance ⇔ maximizing eigenvalue**

Therefore:

- **First principal component** = eigenvector with largest eigenvalue
- **Second principal component** = eigenvector with second largest eigenvalue (orthogonal to the first)
- **Remaining components** = next largest eigenvalues (all mutually orthogonal)

## 5. Geometric Interpretation (Very Important)

Understanding what PCA is doing geometrically provides crucial intuition:

### 5.1 What PCA Is Geometrically Doing

Consider a data cloud in $\mathbb{R}^n$. PCA finds orthogonal axes that:

- **Maximize spread:** The first axis aligns with the direction of maximum variance
- **Minimize reconstruction error:** When we project data onto these axes and reconstruct, we minimize the squared error

This is equivalent to:

- **Rotating the coordinate system:** We're finding a new orthonormal basis for the data space
- **Aligning axes with directions of maximal variance:** The new coordinate axes point along the "principal directions" of the data

The geometric interpretation connects the statistical objective (variance maximization) with the linear algebra solution (eigenvalue decomposition).

## 6. The Connection: When Do PCA and SVD Meet?

Now we can understand when and why PCA and SVD coincide:

For **centered data** $\tilde{X}$, the covariance matrix is:

$$\Sigma_x = \frac{1}{m} \tilde{X}^\top \tilde{X}$$

If we apply SVD to the centered data matrix:

$$\tilde{X} = U \Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$ has orthonormal columns, $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values, and $V \in \mathbb{R}^{n \times n}$ has orthonormal columns.

Now, let's compute $\tilde{X}^\top \tilde{X}$ using the SVD decomposition:

$$\tilde{X}^\top \tilde{X} = (U \Sigma V^\top)^\top (U \Sigma V^\top) = (V \Sigma^\top U^\top) (U \Sigma V^\top) = V \Sigma^\top \Sigma V^\top = V \Sigma^2 V^\top$$

where $\Sigma^2$ is a diagonal matrix with entries $\sigma_i^2$ (the squares of singular values).

Now, the covariance matrix becomes:

$$\Sigma_x = \frac{1}{m} \tilde{X}^\top \tilde{X} = \frac{1}{m} V \Sigma^2 V^\top$$

**Key observation:** This is an eigendecomposition!

Multiplying both sides by $V$ from the right and using $V V^\top = I_n$:

$$\Sigma_x V = V \left(\frac{1}{m} \Sigma^2\right)$$

This shows that $\Sigma_x V = V \Lambda$, where $\Lambda = \frac{1}{m} \Sigma^2$ is a diagonal matrix. This is exactly the eigenvalue equation!

The eigenvalue equation $\Sigma_x V = V \Lambda$ means that each column $\mathbf{v}_i$ of $V$ satisfies:

$$\Sigma_x \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

Therefore:

- The **columns of $V$** (right singular vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$) are the **eigenvectors of $\Sigma_x$**, which are exactly the **principal components**
- The **eigenvalues** are $\lambda_i = \frac{\sigma_i^2}{m}$, where $\sigma_i$ are the singular values
- The first principal component is $\mathbf{v}_1$ (corresponding to the largest eigenvalue $\lambda_1$)

### 6.1 Projecting Data onto Principal Directions

Now that we have found the principal directions (the columns of $V$), the next step is to project our data onto these directions. This is where the left singular vectors $U$ come into play.

For a data point $\tilde{\mathbf{x}}_i$ (a row of $\tilde{X}$) and principal component $\mathbf{v}_j$ (a column of $V$), the projection is:

$$z_{ij} = \tilde{\mathbf{x}}_i^\top \mathbf{v}_j$$

This gives the coordinate of data point $i$ along principal component $j$. When we project **all** data points onto principal component $\mathbf{v}_j$, we get:

$$\tilde{X} \mathbf{v}_j = U \Sigma V^\top \mathbf{v}_j = U \Sigma \mathbf{e}_j = \sigma_j \mathbf{u}_j$$

where $\mathbf{u}_j$ is the $j$-th column of $U$. This shows that:

$$\tilde{X} \mathbf{v}_j = \sigma_j \mathbf{u}_j$$

In other words, **projecting all data points onto principal component $\mathbf{v}_j$ gives $\sigma_j \mathbf{u}_j$**, where $\mathbf{u}_j$ is the $j$-th left singular vector (normalized projection vector).

More generally, when we project all data onto all principal components, we get:

$$\tilde{X} V = U \Sigma V^\top V = U \Sigma$$

since $V^\top V = I_n$. The matrix $\tilde{X} V$ contains all projections: the $(i,j)$-th element is the projection of data point $i$ onto principal component $j$. This equals $U \Sigma$.

> **Summary:** For centered data $\tilde{X} = U \Sigma V^\top$:
> - $V$ contains eigenvectors of $\tilde{X}^\top \tilde{X}$ (and of covariance matrix $\Sigma_x$)
> - $U$ contains eigenvectors of $\tilde{X} \tilde{X}^\top$
> - Eigenvalues: $\lambda_i = \frac{\sigma_i^2}{m}$ where $\sigma_i$ are singular values

> **When They Meet:** For centered data, PCA and SVD produce the same result. The principal components from PCA (eigenvectors of the covariance matrix) are exactly the right singular vectors from SVD. However, they are still *different things*—PCA is solving a statistical optimization problem, while SVD is performing a matrix factorization. They just happen to coincide for centered data.

### 6.2 Why Use SVD for PCA Computationally?

Even though PCA and SVD are conceptually different, in practice, SVD is often the preferred computational method for PCA because:

1. **Numerical Stability:** SVD algorithms (like those in LAPACK) are more numerically stable than directly computing eigenvalues of the covariance matrix, especially when $n \gg m$ or $m \gg n$.
2. **Efficiency:** For large matrices, SVD can be more efficient, especially when we only need the top $k$ components (truncated SVD).
3. **Memory Efficiency:** When $n$ is very large, computing $\Sigma_x \in \mathbb{R}^{n \times n}$ may be infeasible, but SVD can work directly on $\tilde{X}$.

This is why libraries like scikit-learn implement PCA using SVD under the hood—it's the best computational approach, even though PCA itself is a statistical technique.

## 7. Practical Considerations

### 7.1 Implementation

In Python, both approaches are available. Here's a complete example showing the relationship:

```python
import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import svd

# Generate sample data
np.random.seed(42)
m, n = 100, 3  # 100 samples, 3 features
X = np.random.randn(m, n)

# Center the data (crucial for PCA)
X_centered = X - X.mean(axis=0)

# Method 1: PCA using sklearn (uses SVD internally)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)
print("PCA components shape:", X_pca.shape)
print("PCA explained variance:", pca.explained_variance_ratio_)

# Method 2: Direct SVD approach
U, s, Vt = svd(X_centered, full_matrices=False)
# Vt contains the principal components (right singular vectors)
# s contains singular values (related to eigenvalues)
# U contains the projections (left singular vectors)

# Project data using top k components
k = 2
X_svd = U[:, :k] @ np.diag(s[:k])
# Or equivalently:
X_svd_alt = X_centered @ Vt[:k].T

# Verify they give the same result (up to sign)
print("\nAre PCA and SVD results equivalent?")
print("Max difference:", np.abs(X_pca - X_svd_alt).max())

# Eigenvalues from covariance matrix
cov_matrix = (1/(m-1)) * X_centered.T @ X_centered
eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
eigenvals = eigenvals[::-1]  # Sort descending
eigenvecs = eigenvecs[:, ::-1]

# Relationship: σ² = (n-1) * λ
print("\nRelationship between singular values and eigenvalues:")
print("Singular values (squared):", s[:k]**2)
print("Eigenvalues (scaled):", eigenvals[:k] * (m-1))
print("They match:", np.allclose(s[:k]**2, eigenvals[:k] * (m-1)))
```

Notice that sklearn's PCA uses SVD internally for numerical stability, but the conceptual framework is still statistical (variance maximization).

### 7.2 Computational Complexity

For a matrix $X \in \mathbb{R}^{n \times d}$:

- **Full SVD:** $O(\min(n^2d, nd^2))$
- **Truncated SVD (top $k$):** $O(knd)$ using iterative methods
- **PCA via Covariance:** $O(nd^2 + d^3)$ (computing $C$ + eigendecomposition)

For large-scale problems, truncated SVD is often preferred.

---

*Note: This post includes interactive visualizations in the original HTML version. The mathematical content and key insights are preserved here in Markdown format.*
