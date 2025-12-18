import numpy as np

def sgd_momentum(
    f,
    grad_f,
    x0,
    lr=0.01,
    beta=0.9,
    tol=1e-6,
    max_iter=10000,
):
    x = x0.copy()
    v = np.zeros_like(x)
    losses = []

    for it in range(max_iter):
        loss = f(x)
        losses.append(loss)

        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        v = beta * v + (1 - beta) * g
        x = x - lr * v

    return x, it + 1, losses
