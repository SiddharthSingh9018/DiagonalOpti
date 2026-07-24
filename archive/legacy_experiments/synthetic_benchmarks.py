import numpy as np
import matplotlib.pyplot as plt

from optimizer.optimizer import sa_rsvd_tr_optimizer


# ============================================================
# Helpers: Quadratic & Rosenbrock
# ============================================================

def generate_quadratic(dim=10, cond_number=10):
    V, _ = np.linalg.qr(np.random.randn(dim, dim))
    eigvals = np.linspace(1.0, float(cond_number), dim)
    Q = V @ np.diag(eigvals) @ V.T
    return Q


def quadratic_f(x, Q):
    return 0.5 * x.T @ Q @ x


def quadratic_grad(x, Q):
    return Q @ x


def rosenbrock_f(x):
    return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    dx = -2.0 * (1 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    dy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


# ============================================================
# Adam baseline (minimal, stable)
# ============================================================

def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, tol=1e-6, max_iter=20000):

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


# ============================================================
# Run & plot helper
# ============================================================

def run_experiment(name, f, grad_f, x0, optimizers, logy=True):
    print(f"\n=== {name} ===")
    plt.figure(figsize=(7, 5))

    for label, opt in optimizers.items():
        x_opt, iters, losses = opt()
        print(f"{label:18s} | iters = {iters:5d} | final loss = {losses[-1]:.2e}")
        plt.plot(losses, label=label)

    if logy:
        plt.yscale("log")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    dim = 10

    # -----------------------------
    # 1) Well-conditioned quadratic
    # -----------------------------
    Q_well = generate_quadratic(dim, cond_number=10)
    x0 = np.random.randn(dim)

    run_experiment(
        "Well-conditioned Quadratic",
        lambda x: quadratic_f(x, Q_well),
        lambda x: quadratic_grad(x, Q_well),
        x0,
        optimizers={
            "Adam": lambda: adam(
                lambda x: quadratic_f(x, Q_well),
                lambda x: quadratic_grad(x, Q_well),
                x0,
                lr=0.01,
                max_iter=8000,
            ),
            "S/A–RSVD–TR": lambda: sa_rsvd_tr_optimizer(
                lambda x: quadratic_f(x, Q_well),
                lambda x: quadratic_grad(x, Q_well),
                x0,
                k=3,
                max_iter=2000,
                warmup_iters=20,   # 🔑 critical
            ),
        },
    )

    # -----------------------------
    # 2) Ill-conditioned quadratic
    # -----------------------------
    Q_ill = generate_quadratic(dim, cond_number=100)
    x0 = np.random.randn(dim)

    run_experiment(
        "Ill-conditioned Quadratic",
        lambda x: quadratic_f(x, Q_ill),
        lambda x: quadratic_grad(x, Q_ill),
        x0,
        optimizers={
            "Adam": lambda: adam(
                lambda x: quadratic_f(x, Q_ill),
                lambda x: quadratic_grad(x, Q_ill),
                x0,
                lr=0.005,
                max_iter=12000,
            ),
            "S/A–RSVD–TR": lambda: sa_rsvd_tr_optimizer(
                lambda x: quadratic_f(x, Q_ill),
                lambda x: quadratic_grad(x, Q_ill),
                x0,
                k=3,
                max_iter=3000,
                warmup_iters=20,   # 🔑 critical
            ),
        },
    )

    # -----------------------------
    # 3) Rosenbrock
    # -----------------------------
    x0 = np.array([-1.2, 1.0])

    run_experiment(
        "Rosenbrock Function",
        rosenbrock_f,
        rosenbrock_grad,
        x0,
        optimizers={
            "Adam": lambda: adam(
                rosenbrock_f,
                rosenbrock_grad,
                x0,
                lr=0.002,
                max_iter=20000,
            ),
            "S/A–RSVD–TR": lambda: sa_rsvd_tr_optimizer(
                rosenbrock_f,
                rosenbrock_grad,
                x0,
                k=2,
                lr=0.01,
                max_iter=5000,
                warmup_iters=30,   # 🔑 THIS is why Rosenbrock works
            ),
        },
    )
