# Eigenvalues and Eigenvectors in AI

Eigenvalues and eigenvectors are foundational concepts in linear algebra with critical applications in artificial intelligence (AI), machine learning, and data science. This README provides an overview of their mathematical definition, significance, and practical use cases in AI systems.

---

## Table of Contents
1. [Definition](#definition)
2. [Importance in AI](#importance-in-ai)
3. [Applications](#applications)
   - Dimensionality Reduction (PCA)
   - Graph-Based Learning (Spectral Clustering)
   - Neural Networks & Deep Learning
4. [Code Example](#code-example)
5. [References](#references)

---

## Definition
For a square matrix \( A \in \mathbb{R}^{n \times n} \), an eigenvector \( v \) and its corresponding eigenvalue \( \lambda \) satisfy:
\[
A v = \lambda v
\]
- **Eigenvalue (\( \lambda \))**: Scalar representing the scaling factor applied to the eigenvector.
- **Eigenvector (\( v \))**: Non-zero vector that only changes by a scalar factor (its direction remains unchanged) under the linear transformation defined by \( A \).

---

## Importance in AI
Eigenvalues and eigenvectors help analyze and simplify high-dimensional data by identifying key patterns and structures. They are used in:
- **Dimensionality Reduction**: Preserve critical information while reducing data complexity.
- **Graph Analysis**: Model relationships in networks (e.g., social networks, molecules).
- **Stability Analysis**: Diagnose numerical stability in neural network training.

---

## Applications

### 1. Dimensionality Reduction: Principal Component Analysis (PCA)
PCA uses eigenvectors of the covariance matrix to identify directions of maximum variance in data:
- Eigenvectors define principal components.
- Eigenvalues quantify the variance explained by each component.
- **Example**: Reducing a 1000-dimensional image dataset to 50 dimensions while retaining 95% of the variance.

### 2. Graph-Based Learning: Spectral Clustering
Spectral clustering leverages eigenvectors of the graph Laplacian matrix to partition data into clusters:
- Eigenvectors reveal connectivity patterns in graphs.
- **Example**: Grouping customers into segments based on purchasing behavior.

### 3. Neural Networks & Deep Learning
Eigenvalue decomposition helps analyze weight matrices in neural networks:
- **Vanishing/Exploding Gradients**: Eigenvalues of weight matrices indicate gradient stability.
- **Singular Value Decomposition (SVD)**: Used in techniques like weight pruning and low-rank approximations.

---

## Code Example: Computing Eigenvalues/Eigenvectors in Python
```python
import numpy as np

# Define a square matrix A
A = np.array([[4, 2], 
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify Av = \u03bbv for the first eigenvector
v = eigenvectors[:, 0]  # First eigenvector
lambda_val = eigenvalues[0]  # Corresponding eigenvalue

# Compute Av and \u03bbv
Av = A @ v
lambda_v = lambda_val * v

# Display results
print("\nA @ v:", np.round(Av, 2))
print("\u03bb * v:", np.round(lambda_v, 2))
```

---

## References
- Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
