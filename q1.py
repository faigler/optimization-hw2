import numpy as np

# Define the matrix A
A = np.array([
    [1, 2, 3, 4],
    [2, 4, -4, 8],
    [-5, 4, 1, 5],
    [5, 0, -3, -7]
], dtype=float)


'--------------1.b-------------------------'
# Compute the SVD decomposition: A = U @ Sigma @ Vt
U, S, Vt = np.linalg.svd(A)

# The largest singular value is the first one
sigma_max = S[0]

# The vector x that maximizes ||Ax||_2 / ||x||_2
# is the first right singular vector
x = Vt[0, :]

# Compute the ratio
Ax = A @ x
ratio = np.linalg.norm(Ax, 2) / np.linalg.norm(x, 2)

print("Singular values:")
print(S)

print("\nLargest singular value sigma_max:")
print(sigma_max)

print("\nVector x that maximizes the expression:")
print(x)

print("\nA @ x:")
print(Ax)

print("\n||A x||_2 / ||x||_2:")
print(ratio)


'----------1.c--------------------'

# Matrix norms
norm_1 = np.linalg.norm(A, 1)
norm_inf = np.linalg.norm(A, np.inf)
norm_2 = np.linalg.norm(A, 2)

# Spectral radius
eigenvalues = np.linalg.eigvals(A)
spectral_radius = np.max(np.abs(eigenvalues))

print("\nSpectral radius rho(A):")
print(spectral_radius)

print("\nMatrix norms:")
print("||A||_1 =", norm_1)
print("||A||_2 =", norm_2)
print("||A||_inf =", norm_inf)

print("\nComparison:")
print("rho(A) <= ||A||_1:", spectral_radius <= norm_1)
print("rho(A) <= ||A||_2:", spectral_radius <= norm_2)
print("rho(A) <= ||A||_inf:", spectral_radius <= norm_inf)

print("""
This was expected because the spectral radius is always bounded above by any induced matrix norm.
Intuitively, the spectral radius measures the largest scaling factor of A along an eigenvector direction,
while an induced norm measures the maximal stretching of A over all possible directions.
Therefore, the maximal stretching must be at least as large as the stretching in any eigenvector direction.
Hence, for all induced norms:
    rho(A) <= ||A||
Therefore, it makes sense that ||A||_1, ||A||_2, and ||A||_inf are all larger than rho(A).
""")