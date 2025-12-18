import numpy as np

def adam(
    f,
    grad_f,
    x0,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    tol=1e-6,
    max_iter=10000,
):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []

    for t in range(1, max_iter + 1):
        loss = f(x)
        losses.append(loss)

        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return x, t, losses
