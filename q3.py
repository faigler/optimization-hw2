import numpy as np


'------3.a-----------'


def dot_product(u, v):
    """Compute inner product u^T v using primitive code."""
    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result


def vector_norm(v):
    """Compute Euclidean norm ||v||_2 using primitive code."""
    return dot_product(v, v) ** 0.5


def classical_gram_schmidt(A):
    """
    Compute QR factorization using Classical Gram-Schmidt.

    In classical GS, when computing column i, we subtract from a_i
    all projections onto the previous q_j vectors.
    """
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)

    for i in range(n):
        # Start from the original column a_i
        v = A[:, i].copy()

        # Remove projections on previous q_j vectors
        for j in range(i):
            R[j, i] = dot_product(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]

        # Normalize the remaining vector
        R[i, i] = vector_norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R


def modified_gram_schmidt(A):
    """
    Compute QR factorization using Modified Gram-Schmidt.

    In modified GS, when computing column i, we update v after each
    projection, and the next projection is computed using the updated v.
    """
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)

    for i in range(n):
        # Start from the original column a_i
        v = A[:, i].copy()

        # Sequentially remove projections from the current updated v
        for j in range(i):
            R[j, i] = dot_product(Q[:, j], v)
            v = v - R[j, i] * Q[:, j]

        # Normalize the remaining vector
        R[i, i] = vector_norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R

