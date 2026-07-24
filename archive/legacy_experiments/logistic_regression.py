import numpy as np
import matplotlib.pyplot as plt

from optimizer.optimizer import sa_rsvd_tr_optimizer


# ============================================================
# Logistic regression (binary, full batch)
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w, X, y, reg=1e-3):
    z = X @ w
    p = sigmoid(z)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    loss += 0.5 * reg * np.dot(w, w)
    return loss


def logistic_grad(w, X, y, reg=1e-3):
    z = X @ w
    p = sigmoid(z)
    grad = X.T @ (p - y) / X.shape[0]
    grad += reg * w
    return grad


# ============================================================
# Data generation
# ============================================================

def make_dataset(n=500, d=10):
    np.random.seed(0)
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    logits = X @ true_w
    y = (logits > 0).astype(float)
    return X, y


# ============================================================
# Run experiment
# ============================================================

if __name__ == "__main__":
    X, y = make_dataset(n=500, d=10)
    d = X.shape[1]
    w0 = np.zeros(d)

    def f(w):
        return logistic_loss(w, X, y)

    def grad_f(w):
        return logistic_grad(w, X, y)

    w_opt, iters, losses = sa_rsvd_tr_optimizer(
        f,
        grad_f,
        w0,
        k=5,
        lr=1.0,
        max_iter=2000,
        warmup_iters=30,
        max_step_norm=0.1,
        tol=1e-6,
    )

    print(f"Converged in {iters} iterations")
    print(f"Final loss: {losses[-1]:.6f}")

    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Logistic Regression – S/A–RSVD–TR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
