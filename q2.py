import numpy as np


'---------2.a---------'


def fwd_sub(L, b):
    """
    Solve Lx = b where L is a lower triangular matrix.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        sum_val = 0.0

        for j in range(i):
            sum_val += L[i, j] * x[j]

        x[i] = (b[i] - sum_val) / L[i, i]

    return x


def bwd_sub(U, b):
    """
    Solve Ux = b where U is an upper triangular matrix.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_val = 0.0

        for j in range(i + 1, n):
            sum_val += U[i, j] * x[j]

        x[i] = (b[i] - sum_val) / U[i, i]

    return x


print("Both algorithms run in O(n^2) time. This is because for each row i we compute a sum over at most n entries, and there are n rows. More precisely, the total number of arithmetic operations is proportional to 1+2+...+n = O(n^2).")


'------2.b-----------'


A = np.array([
    [2, 1, 2],
    [1, -2, 1],
    [1, 2, 3],
    [1, 1, 1]
], dtype=float)

b = np.array([6, 1, 5, 2], dtype=float)

# Normal equations:
# A^T A x = A^T b
ATA = A.T @ A
ATb = A.T @ b

# Cholesky factorization:
# A^T A = L L^T
L = np.linalg.cholesky(ATA)

# Solve L y = A^T b
y = fwd_sub(L, ATb)

# Solve L^T x = y
x = bwd_sub(L.T, y)

print("A^T A:")
print(ATA)

print("\nA^T b:")
print(ATb)

print("\nCholesky factor L:")
print(L)

print("\nIntermediate solution y from L y = A^T b:")
print(y)

print("\nLeast squares solution x:")
print(x)

print("\nCheck A @ x:")
print(A @ x)

print("\nResidual Ax - b:")
print(A @ x - b)

print("\nResidual norm ||Ax - b||_2:")
print(np.linalg.norm(A @ x - b, 2))


'-------2.c-----------'


# Least Squares using QR factorization
# A = Q R
# The LS problem becomes R x = Q^T b

Q, R = np.linalg.qr(A, mode="reduced")

QTb = Q.T @ b

x_qr = bwd_sub(R, QTb)

print("\n--- Section 2(c)(i): QR factorization ---")
print("Q:")
print(Q)

print("\nR:")
print(R)

print("\nQ^T b:")
print(QTb)

print("\nLeast squares solution using QR:")
print(x_qr)

print("\nResidual norm using QR:")
print(np.linalg.norm(A @ x_qr - b, 2))


# Least Squares using SVD factorization
# A = U Sigma V^T
# x = V Sigma^{-1} U^T b

U, S, Vt = np.linalg.svd(A, full_matrices=False)

UTb = U.T @ b

# Solve Sigma z = U^T b.
# Since Sigma is diagonal, this means z_i = (U^T b)_i / sigma_i.
z = UTb / S

# Then x = V z.
# Since Vt = V^T, we have V = Vt.T.
x_svd = Vt.T @ z

print("\n--- Section 2(c)(ii): SVD factorization ---")
print("Singular values:")
print(S)

print("\nU^T b:")
print(UTb)

print("\nz = Sigma^{-1} U^T b:")
print(z)

print("\nLeast squares solution using SVD:")
print(x_svd)

print("\nResidual norm using SVD:")
print(np.linalg.norm(A @ x_svd - b, 2))


# Compare all methods
print("\n--- Comparison ---")
print("x from Cholesky:")
print(x)

print("\nx from QR:")
print(x_qr)

print("\nx from SVD:")
print(x_svd)

print("\nAre Cholesky and QR close?")
print(np.allclose(x, x_qr))

print("\nAre Cholesky and SVD close?")
print(np.allclose(x, x_svd))