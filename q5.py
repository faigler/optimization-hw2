import numpy as np


'------5.a-----------'

A = np.array([
    [5, 4, 0],
    [4, 5, 0],
    [0, 0, 2],
    [0, 0, 1]
], dtype=float)

B = np.array([
    [15, 14, 0],
    [14, 15, 0],
    [0, 0, 2],
    [0, 0, 1]
], dtype=float)


def best_rank_1_approximation(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    sigma_1 = S[0]
    u_1 = U[:, 0]
    v_1 = Vt[0, :]

    M_1 = sigma_1 * np.outer(u_1, v_1)

    return M_1


A_1 = best_rank_1_approximation(A)
B_1 = best_rank_1_approximation(B)

print("Best rank-1 approximation of A:")
print(A_1)

print("\nBest rank-1 approximation of B:")
print(B_1)


'------5.b-----------'

def approximation_errors(M, M_1):
    abs_error = np.linalg.norm(M - M_1, ord="fro")
    rel_error = abs_error / np.linalg.norm(M, ord="fro")
    singular_values = np.linalg.svd(M, compute_uv=False)

    return abs_error, rel_error, singular_values


A_abs_error, A_rel_error, A_singular_values = approximation_errors(A, A_1)
B_abs_error, B_rel_error, B_singular_values = approximation_errors(B, B_1)

print("A singular values:")
print(A_singular_values)

print("\nB singular values:")
print(B_singular_values)

print("\n||A - A_1||_F:")
print(A_abs_error)

print("\n||B - B_1||_F:")
print(B_abs_error)

print("\nRelative error for A:")
print(A_rel_error)

print("\nRelative error for B:")
print(B_rel_error)