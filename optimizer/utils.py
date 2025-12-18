import numpy as np


def safe_norm(x, eps=1e-12):
    """
    Compute Euclidean norm with numerical safety.
    """
    return np.sqrt(np.sum(x * x) + eps)


def project_to_ball(x, radius):
    """
    Project vector x onto an L2 ball of given radius.
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= radius:
        return x
    return x * (radius / (norm_x + 1e-12))


def clip_by_norm(x, max_norm):
    """
    Clip vector by L2 norm.
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= max_norm:
        return x
    return x * (max_norm / (norm_x + 1e-12))


def quadratic_model_value(f0, g, step, H_step):
    """
    Evaluate local quadratic model:
        m(p) = f0 + g^T p + 0.5 p^T H p
    """
    return f0 + g @ step + 0.5 * (step @ H_step)


def acceptance_ratio(f_old, f_new, m_pred, eps=1e-12):
    """
    Compute trust-region acceptance ratio.
    """
    denom = f_old - m_pred
    if abs(denom) < eps:
        return 1.0
    return (f_old - f_new) / denom


def ensure_column_vector(x):
    """
    Ensure vector has shape (n, 1).
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x
