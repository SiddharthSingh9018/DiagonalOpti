import numpy as np


def sa_rsvd_tr_optimizer(
    f,
    grad_f,
    x0,
    k=3,                 # target rank
    oversample=4,        # RSVD oversampling
    lr=1.0,
    R0=1.0,              # initial trust radius
    R_max=5.0,           # maximum trust radius
    gamma_trust=1.0,     # antisymmetry-based shrink strength
    beta_S=0.2,          # EMA factor for symmetric curvature
    drift_thresh=0.5,    # subspace drift threshold
    c_noise=1.0,         # noise-aware damping factor
    lambda_min=1e-3,     # minimum eigenvalue floor
    alpha_residual=0.1,  # residual gradient weight
    antisymm_gate=2.0,   # fallback threshold
    alpha_fallback=0.2,  # gradient fallback step size
    tol=1e-6,
    max_iter=2000,
):
    """
    S/A–RSVD–TR Optimizer

    Hybrid optimizer combining:
    - Randomized low-rank Hessian approximation
    - Symmetric / antisymmetric curvature decomposition
    - Noise-aware eigenvalue damping
    - Drift-controlled subspace tracking
    - Trust-region step control
    """

    x = x0.copy()
    n = x.shape[0]
    losses = []

    S_ema = None
    U_prev = None
    R_tr = R0

    for it in range(1, max_iter + 1):
        f_old = f(x)
        losses.append(f_old)

        g = grad_f(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        # --------------------------------------------------
        # Randomized Hessian approximation (finite-diff HVP)
        # --------------------------------------------------
        L = min(n, k + oversample)
        Omega = np.random.randn(n, L)
        eps_fd = 1e-4

        Y = np.zeros((n, L))
        for j in range(L):
            Y[:, j] = (grad_f(x + eps_fd * Omega[:, j]) - g) / eps_fd

        Q, _ = np.linalg.qr(Y)

        HQ = np.zeros((n, L))
        for j in range(L):
            HQ[:, j] = (grad_f(x + eps_fd * Q[:, j]) - g) / eps_fd

        B = Q.T @ HQ

        # --------------------------------------------------
        # Symmetric / antisymmetric split
        # --------------------------------------------------
        S = 0.5 * (B + B.T)
        A = 0.5 * (B - B.T)

        S_norm = np.linalg.norm(S, "fro") + 1e-12
        A_norm = np.linalg.norm(A, "fro")
        rho_antisymm = A_norm / S_norm

        # --------------------------------------------------
        # EMA smoothing of symmetric curvature
        # --------------------------------------------------
        if S_ema is None:
            S_ema = S
        else:
            S_ema = (1.0 - beta_S) * S_ema + beta_S * S

        # --------------------------------------------------
        # Eigen-decomposition
        # --------------------------------------------------
        evals, W = np.linalg.eigh(S_ema)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        W = W[:, idx]

        k_eff = min(k, L)
        evals_k = evals[:k_eff]
        W_k = W[:, :k_eff]

        U = Q @ W_k

        # --------------------------------------------------
        # Subspace drift control
        # --------------------------------------------------
        if U_prev is not None:
            P_prev = U_prev @ U_prev.T
            P_curr = U @ U.T
            drift = np.linalg.norm(P_curr - P_prev, "fro")

            if drift > drift_thresh:
                alpha_mix = drift_thresh / (drift + 1e-12)
                U_mix = (1.0 - alpha_mix) * U_prev + alpha_mix * U
                U, _ = np.linalg.qr(U_mix)

        U_prev = U.copy()

        # --------------------------------------------------
        # Noise-aware eigenvalue damping
        # --------------------------------------------------
        damp_floor = max(c_noise * A_norm, lambda_min)
        evals_clamped = np.maximum(evals_k, damp_floor)

        # --------------------------------------------------
        # Step direction
        # --------------------------------------------------
        if rho_antisymm > antisymm_gate:
            p = -alpha_fallback * g
        else:
            z = U.T @ g
            newton_dir = -U @ (z / evals_clamped)
            p = newton_dir - alpha_residual * g

        # --------------------------------------------------
        # Trust-region projection
        # --------------------------------------------------
        p_norm = np.linalg.norm(p)
        R_eff = R_tr / (1.0 + gamma_trust * rho_antisymm)
        R_eff = max(R_eff, 1e-12)

        if p_norm > R_eff:
            p *= R_eff / (p_norm + 1e-12)
            p_norm = R_eff

        step = lr * p
        x_trial = x + step
        f_new = f(x_trial)

        # --------------------------------------------------
        # Quadratic model ratio
        # --------------------------------------------------
        H_step = U @ (evals_k * (U.T @ step))
        m_pred = f_old + g @ step + 0.5 * (step @ H_step)

        denom = f_old - m_pred
        rho_model = 1.0 if abs(denom) < 1e-12 else (f_old - f_new) / denom

        # --------------------------------------------------
        # Trust-region update
        # --------------------------------------------------
        if rho_model < 0.25:
            R_tr *= 0.5
        elif rho_model > 0.75 and p_norm > 0.8 * R_eff:
            R_tr = min(1.5 * R_tr, R_max)

        if rho_model > 0.0:
            x = x_trial

    return x, it, losses
