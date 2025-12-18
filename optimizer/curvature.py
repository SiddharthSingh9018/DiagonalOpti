import numpy as np


def finite_diff_hvp(grad_f, x, v, eps=1e-4):
    """
    Finite-difference Hessian-vector product.
    Approximates H(x) v ≈ [∇f(x + eps v) − ∇f(x)] / eps
    """
    return (grad_f(x + eps * v) - grad_f(x)) / eps


def randomized_hessian_sketch(grad_f, x, k, oversample=4, eps=1e-4):
    """
    Compute a low-rank randomized sketch of the Hessian at x.

    Returns:
        Q : orthonormal basis (n × L)
        B : projected Hessian (L × L)
    """
    n = x.shape[0]
    L = min(n, k + oversample)

    Omega = np.random.randn(n, L)
    Y = np.zeros((n, L))

    g = grad_f(x)
    for j in range(L):
        Y[:, j] = (grad_f(x + eps * Omega[:, j]) - g) / eps

    Q, _ = np.linalg.qr(Y)

    HQ = np.zeros((n, L))
    for j in range(L):
        HQ[:, j] = (grad_f(x + eps * Q[:, j]) - g) / eps

    B = Q.T @ HQ
    return Q, B


def symmetric_antisymmetric_split(B):
    """
    Decompose matrix B into symmetric and antisymmetric parts.
    """
    S = 0.5 * (B + B.T)
    A = 0.5 * (B - B.T)
    return S, A


def antisymmetry_ratio(S, A, eps=1e-12):
    """
    Compute ||A||_F / ||S||_F as a noise / non-conservativity indicator.
    """
    return np.linalg.norm(A, "fro") / (np.linalg.norm(S, "fro") + eps)


def damp_eigenvalues(evals, noise_level, c_noise=1.0, lambda_min=1e-3):
    """
    Noise-aware eigenvalue damping.
    """
    floor = max(c_noise * noise_level, lambda_min)
    return np.maximum(evals, floor)


def subspace_drift(U_prev, U_curr):
    """
    Frobenius norm of difference between projection matrices.
    """
    if U_prev is None:
        return 0.0
    P_prev = U_prev @ U_prev.T
    P_curr = U_curr @ U_curr.T
    return np.linalg.norm(P_curr - P_prev, "fro")
