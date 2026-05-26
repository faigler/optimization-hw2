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


'------3.b-----------'


def create_assignment_matrix(dtype):
    """
    Create the matrix A from section 3(b), using the given dtype.
    dtype can be np.float32 or np.float64.
    """
    n = 50

    t = 2.0 ** np.arange(-8, 10, 2.0)
    t = t.astype(dtype)

    m = len(t)

    A = (
        np.random.rand(n, m).astype(dtype)
        @ np.diag(t)
        @ np.random.rand(m, m).astype(dtype)
    )

    return A

# Build the matrix from the assignment using 32-bit precision.

A_32=create_assignment_matrix(np.float32)

# Compute QR factorizations using our own implementations
Q_gs_32, R_gs_32 = classical_gram_schmidt(A_32)
Q_mgs_32, R_mgs_32 = modified_gram_schmidt(A_32)

print("--- Section 3(b): 32-bit QR factorizations computed ---")

print("Classical GS reconstruction error ||A - QR||_F:")
print(np.linalg.norm(A_32 - Q_gs_32 @ R_gs_32, ord="fro"))

print("\nModified GS reconstruction error ||A - QR||_F:")
print(np.linalg.norm(A_32 - Q_mgs_32 @ R_mgs_32, ord="fro"))


'------3.c-----------'


# Compute the orthogonality error ||Q^T Q - I||_F for both QR factorizations.

I = np.eye(A_32.shape[1], dtype=np.float32)

gs_orthogonality_error = np.linalg.norm(Q_gs_32.T @ Q_gs_32 - I, ord="fro")
mgs_orthogonality_error = np.linalg.norm(Q_mgs_32.T @ Q_mgs_32 - I, ord="fro")

print("\n--- Section 3(c): Orthogonality error ---")

print("Classical Gram-Schmidt ||Q^T Q - I||_F:")
print(gs_orthogonality_error)

print("\nModified Gram-Schmidt ||Q^T Q - I||_F:")
print(mgs_orthogonality_error)

if mgs_orthogonality_error < gs_orthogonality_error:
    print("\nModified Gram-Schmidt produced a better QR factorization.")
else:
    print("\nClassical Gram-Schmidt produced a better QR factorization.")


'------3.d-----------'


A_64 = create_assignment_matrix(np.float64)

Q_gs_64, R_gs_64 = classical_gram_schmidt(A_64)
Q_mgs_64, R_mgs_64 = modified_gram_schmidt(A_64)

m = A_64.shape[1]
I_64 = np.eye(m, dtype=np.float64)

gs_orthogonality_error_64 = np.linalg.norm(Q_gs_64.T @ Q_gs_64 - I_64, ord="fro")
mgs_orthogonality_error_64 = np.linalg.norm(Q_mgs_64.T @ Q_mgs_64 - I_64, ord="fro")

print("\n--- Section 3(d): 64-bit orthogonality error ---")

print("Classical Gram-Schmidt ||Q^T Q - I||_F:")
print(gs_orthogonality_error_64)

print("\nModified Gram-Schmidt ||Q^T Q - I||_F:")
print(mgs_orthogonality_error_64)


'------3.e-----------'


print("\n--- Section 3(e): Compare R[j, i] sequences ---")

num_cols = A_64.shape[1]

for i in range(1, num_cols):
    gs_sequence = R_gs_64[:i, i]
    mgs_sequence = R_mgs_64[:i, i]

    max_difference = np.max(np.abs(gs_sequence - mgs_sequence))
    #checks whether the two sequences are numerically close.
    are_close = np.allclose(gs_sequence, mgs_sequence)

    print(f"\nColumn i = {i + 1}")
    print("GS  sequence:", gs_sequence)
    print("MGS sequence:", mgs_sequence)
    print("Max absolute difference:", max_difference)
    print("Are the sequences close?", are_close)


print("\n--- Section 3(e): Verify ||q_j||_2 = 1 ---")

print("\nGS 64-bit column norms:")
print(np.linalg.norm(Q_gs_64, axis=0))

print("\nMGS 64-bit column norms:")
print(np.linalg.norm(Q_mgs_64, axis=0))



print("\n--- Section 3(e): Classical GS - ||a_i|| stays constant ---")

for i in range(1, A_64.shape[1]):
    ai_norm = np.linalg.norm(A_64[:, i])

    # In GS, for column i, the algorithm always uses the original a_i.
    # Therefore, the norm is the same for every j < i.
    gs_norms = [ai_norm] * i

    print(f"Column {i + 1}:")
    print(gs_norms)


print("\n--- Section 3(e): Modified GS - ||v|| is non-increasing ---")

for i in range(1, A_64.shape[1]):
    v = A_64[:, i].copy()
    mgs_norms = [np.linalg.norm(v)]

    for j in range(i):
        r = dot_product(Q_mgs_64[:, j], v)
        v = v - r * Q_mgs_64[:, j]
        mgs_norms.append(np.linalg.norm(v))

    is_decreasing = all(
        mgs_norms[k + 1] <= mgs_norms[k] + 1e-12
        for k in range(len(mgs_norms) - 1)
    )

    print(f"Column {i + 1}:")
    print(mgs_norms)
    print("Non-increasing:", is_decreasing)