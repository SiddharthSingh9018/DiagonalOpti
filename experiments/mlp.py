import numpy as np

from optimizer.optimizer import sa_rsvd_tr_optimizer


# ============================================================
# Synthetic regression data
# ============================================================

def make_synthetic_data(n=200, d=10, noise=0.1):
    X = np.random.randn(n, d)
    true_w = np.random.randn(d, 1)
    y = X @ true_w + noise * np.random.randn(n, 1)
    return X, y


# ============================================================
# Simple 1-hidden-layer MLP
# ============================================================

def mlp_forward(X, params):
    W1, b1, W2, b2 = params
    h = np.tanh(X @ W1 + b1)
    out = h @ W2 + b2
    return out


def mlp_loss(params, X, y):
    preds = mlp_forward(X, params)
    return 0.5 * np.mean((preds - y) ** 2)


def mlp_grad(params, X, y):
    W1, b1, W2, b2 = params
    n = X.shape[0]

    # Forward
    h = np.tanh(X @ W1 + b1)
    preds = h @ W2 + b2
    err = (preds - y) / n   # (n, 1)

    # Backward
    dW2 = h.T @ err                     # (32, 1)
    db2 = np.sum(err, axis=0, keepdims=True)

    dh = err @ W2.T                    # (n, 32)
    dz = dh * (1.0 - h ** 2)

    dW1 = X.T @ dz                     # (10, 32)
    db1 = np.sum(dz, axis=0, keepdims=True)

    # Flatten gradients
    return np.concatenate([
        dW1.ravel(),
        db1.ravel(),
        dW2.ravel(),
        db2.ravel()
    ])


# ============================================================
# Parameter packing helpers
# ============================================================

def pack_params(params):
    return np.concatenate([p.ravel() for p in params])


def unpack_params(theta, shapes):
    params = []
    idx = 0
    for s in shapes:
        size = int(np.prod(s))
        params.append(theta[idx:idx + size].reshape(s))
        idx += size
    return params


# ============================================================
# Run experiment
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_synthetic_data()

    shapes = [
        (10, 32),   # W1
        (1, 32),    # b1
        (32, 1),    # W2
        (1, 1),     # b2
    ]

    params0 = [
        0.1 * np.random.randn(*shapes[0]),
        np.zeros(shapes[1]),
        0.1 * np.random.randn(*shapes[2]),
        np.zeros(shapes[3]),
    ]

    theta0 = pack_params(params0)

    def f(theta):
        return mlp_loss(unpack_params(theta, shapes), X, y)

    def grad_f(theta):
        return mlp_grad(unpack_params(theta, shapes), X, y)

    theta_opt, iters, losses = sa_rsvd_tr_optimizer(
        f=f,
        grad_f=grad_f,
        x0=theta0,
        k=5,
        lr=1.0,
        max_iter=500,
        tol=1e-6,
    )

    print("MLP converged in iterations:", iters)
    print("Final loss:", losses[-1])
