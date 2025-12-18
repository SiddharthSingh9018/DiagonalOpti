import numpy as np
from optimizer.utils import project_to_ball
from optimizer.curvature import symmetric_antisymmetric_split


def sa_rsvd_tr_optimizer(
    f, grad_f, x0,
    k=3,
    oversample=4,
    lr=1.0,
    R0=1.0,
    R_max=5.0,
    gamma_trust=1.0,
    beta_S=0.2,
    drift_thresh=0.5,
    c_noise=1.0,
    lambda_min=1e-3,
    alpha_residual=0.1,
    antisymm_gate=2.0,
    alpha_fallback=0.2,
    tol=1e-6,
    max_iter=5000,
    warmup_iters=50,
    max_step_norm=0.05
):
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
        if np.linalg.norm(g) < tol:
            break

        # ==============================
        # Gradient-only warmup (critical)
        # ==============================
        if it <= warmup_iters:
            step = -0.01 * g
            step_norm = np.linalg.norm(step)
            if step_norm > max_step_norm:
                step *= max_step_norm / (step_norm + 1e-12)
            x = x + step
            continue

        # ==============================
        # RSVD Hessian approximation
        # ==============================
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

        # ==============================
        # Symmetric / antisymmetric split
        # ==============================
        S, A = symmetric_antisymmetric_split(B)
        S_norm = np.linalg.norm(S, "fro") + 1e-12
        A_norm = np.linalg.norm(A, "fro")
        rho_antisymm = A_norm / S_norm

        # ==============================
        # EMA smoothing
        # ==============================
        if S_ema is None:
            S_ema = S
        else:
            S_ema = (1 - beta_S) * S_ema + beta_S * S

        evals, W = np.linalg.eigh(S_ema)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        W = W[:, idx]

        k_eff = min(k, L)
        evals_k = evals[:k_eff]
        W_k = W[:, :k_eff]

        U = Q @ W_k

        # ==============================
        # Subspace drift control
        # ==============================
        if U_prev is not None:
            drift = np.linalg.norm(U @ U.T - U_prev @ U_prev.T, "fro")
            if drift > drift_thresh:
                alpha = drift_thresh / (drift + 1e-12)
                U, _ = np.linalg.qr((1 - alpha) * U_prev + alpha * U)

        U_prev = U.copy()

        # ==============================
        # Noise-aware damping
        # ==============================
        damp_floor = max(c_noise * A_norm, lambda_min)
        evals_clamped = np.maximum(evals_k, damp_floor)

        # ==============================
        # Direction
        # ==============================
        if rho_antisymm > antisymm_gate:
            p = -alpha_fallback * g
        else:
            z = U.T @ g
            p = -U @ (z / evals_clamped) - alpha_residual * g

        # ==============================
        # Trust region + hard step cap
        # ==============================
        R_eff = max(R_tr / (1 + gamma_trust * rho_antisymm), 1e-12)
        step = lr * p
        step = project_to_ball(step, R_eff)

        step_norm = np.linalg.norm(step)
        if step_norm > max_step_norm:
            step *= max_step_norm / (step_norm + 1e-12)

        x_trial = x + step
        f_new = f(x_trial)

        # ==============================
        # Quadratic model
        # ==============================
        H_step = U @ (evals_k * (U.T @ step))
        m_pred = f_old + g @ step + 0.5 * (step @ H_step)

        denom = f_old - m_pred
        rho = 1.0 if abs(denom) < 1e-12 else (f_old - f_new) / denom

        # ==============================
        # Trust-region update
        # ==============================
        if rho < 0.25:
            R_tr *= 0.5
        elif rho > 0.75:
            R_tr = min(1.5 * R_tr, R_max)

        if rho > 0.0:
            x = x_trial

    return x, it, losses
