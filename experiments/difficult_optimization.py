import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.adam import adam
from baselines.sgd import sgd_momentum
from optimizer.optimizer import sa_rsvd_tr_optimizer


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_quadratic(dim, condition_number, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigvals = np.geomspace(1.0, condition_number, dim)
    hessian = q @ np.diag(eigvals) @ q.T
    return hessian


def quadratic_problem(dim, condition_number, seed):
    hessian = make_quadratic(dim, condition_number, seed)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=dim)

    def f(x):
        return 0.5 * float(x.T @ hessian @ x)

    def grad(x):
        return hessian @ x

    return f, grad, x0


def make_deep_mlp_problem(seed, n=256, d_in=16, hidden=32, depth=4, d_out=1, noise=0.05):
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, 100.0, d_in)
    x = rng.normal(size=(n, d_in)) / scales
    true_w = rng.normal(size=(d_in, d_out))
    y = np.tanh(x @ true_w) + noise * rng.normal(size=(n, d_out))
    shapes = []
    dims = [d_in] + [hidden] * depth + [d_out]
    for left, right in zip(dims[:-1], dims[1:]):
        shapes.append((left, right))
        shapes.append((right,))
    params = []
    for shape in shapes:
        if len(shape) == 2:
            params.append(0.1 * rng.normal(size=shape))
        else:
            params.append(np.zeros(shape))
    theta0 = pack(params)

    def unpack(theta):
        out = []
        index = 0
        for shape in shapes:
            size = int(np.prod(shape))
            out.append(theta[index:index + size].reshape(shape))
            index += size
        return out

    def forward(theta):
        params = unpack(theta)
        h = x
        for i in range(0, len(params) - 2, 2):
            h = np.tanh(h @ params[i] + params[i + 1])
        return h @ params[-2] + params[-1]

    def f(theta):
        pred = forward(theta)
        return 0.5 * float(np.mean((pred - y) ** 2))

    def grad(theta):
        params = unpack(theta)
        activations = [x]
        preacts = []
        h = x
        for i in range(0, len(params) - 2, 2):
            z = h @ params[i] + params[i + 1]
            preacts.append(z)
            h = np.tanh(z)
            activations.append(h)
        pred = h @ params[-2] + params[-1]
        delta = (pred - y) / n
        grads = [None] * len(params)
        grads[-2] = activations[-1].T @ delta
        grads[-1] = np.sum(delta, axis=0)
        delta = delta @ params[-2].T
        for layer in range(depth - 1, -1, -1):
            dz = delta * (1.0 - np.tanh(preacts[layer]) ** 2)
            wi = 2 * layer
            grads[wi] = activations[layer].T @ dz
            grads[wi + 1] = np.sum(dz, axis=0)
            if layer > 0:
                delta = dz @ params[wi].T
        return pack(grads)

    return f, grad, theta0


def pack(params):
    return np.concatenate([p.ravel() for p in params])


def run_optimizer(name, f, grad, x0, max_iter):
    start = time.perf_counter()
    if name == "Adam":
        _, iters, losses = adam(f, grad, x0, lr=0.002, max_iter=max_iter)
    elif name == "SGD":
        _, iters, losses = sgd_momentum(f, grad, x0, lr=0.001, beta=0.9, max_iter=max_iter)
    elif name == "DiagonalOptimiser":
        _, iters, losses = sa_rsvd_tr_optimizer(
            f,
            grad,
            x0,
            k=min(5, x0.size),
            oversample=3,
            lr=1.0,
            max_iter=max_iter,
            warmup_iters=min(30, max_iter // 10),
            max_step_norm=0.05,
        )
    else:
        raise ValueError(name)
    elapsed = time.perf_counter() - start
    return iters, losses, elapsed


def summarize_losses(losses, elapsed, target_losses):
    best = min(losses)
    out = {
        "final_loss": float(losses[-1]),
        "best_loss": float(best),
        "wall_time_sec": elapsed,
        "stable": bool(all(math.isfinite(float(v)) for v in losses)),
        "loss_auc_steps": float(np.trapezoid(losses, dx=1.0)) if len(losses) > 1 else 0.0,
    }
    for target in target_losses:
        hit = next((i + 1 for i, loss in enumerate(losses) if loss <= target), None)
        out[f"steps_to_loss_{target}"] = hit
    return out


def run_suite(output_dir, seeds, max_iter):
    output_dir = Path(output_dir)
    metrics = []
    summaries = []
    optimizers = ["Adam", "SGD", "DiagonalOptimiser"]
    quad_targets = [1e-2, 1e-4, 1e-6]
    mlp_targets = [0.05, 0.02, 0.01]

    for seed in seeds:
        for cond in [10, 100, 1000, 10000]:
            f, grad, x0 = quadratic_problem(dim=20, condition_number=cond, seed=seed)
            for opt in optimizers:
                iters, losses, elapsed = run_optimizer(opt, f, grad, x0.copy(), max_iter)
                run_id = f"quadratic_cond{cond}_{opt}_seed{seed}"
                for step, loss in enumerate(losses, start=1):
                    metrics.append({"run_id": run_id, "problem": "quadratic", "condition_number": cond, "optimizer": opt, "seed": seed, "step": step, "loss": float(loss)})
                summary = {"run_id": run_id, "problem": "quadratic", "condition_number": cond, "optimizer": opt, "seed": seed, "iters": iters}
                summary.update(summarize_losses(losses, elapsed, quad_targets))
                summaries.append(summary)

        f, grad, x0 = make_deep_mlp_problem(seed)
        for opt in optimizers:
            iters, losses, elapsed = run_optimizer(opt, f, grad, x0.copy(), max_iter)
            run_id = f"deep_mlp_{opt}_seed{seed}"
            for step, loss in enumerate(losses, start=1):
                metrics.append({"run_id": run_id, "problem": "deep_mlp", "condition_number": "", "optimizer": opt, "seed": seed, "step": step, "loss": float(loss)})
            summary = {"run_id": run_id, "problem": "deep_mlp", "condition_number": "", "optimizer": opt, "seed": seed, "iters": iters}
            summary.update(summarize_losses(losses, elapsed, mlp_targets))
            summaries.append(summary)

    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "summaries.csv", summaries)
    aggregate_rows = aggregate(summaries)
    write_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_json(output_dir / "aggregate.json", aggregate_rows)
    plot_difficult(metrics, summaries, output_dir / "plots")
    return aggregate_rows


def aggregate(summaries):
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row["problem"], row["condition_number"], row["optimizer"])].append(row)
    rows = []
    keys = ["final_loss", "best_loss", "wall_time_sec", "loss_auc_steps"]
    target_keys = sorted({k for row in summaries for k in row if k.startswith("steps_to_loss_")})
    for (problem, cond, opt), group in sorted(grouped.items()):
        out = {"problem": problem, "condition_number": cond, "optimizer": opt, "runs": len(group), "stable_runs": sum(r["stable"] for r in group)}
        for key in keys + target_keys:
            vals = [r[key] for r in group if r.get(key) is not None]
            vals = [float(v) for v in vals]
            out[f"{key}_mean"] = mean(vals) if vals else None
            out[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0 if vals else None
        rows.append(out)
    return rows


def plot_difficult(metrics, summaries, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for problem in sorted(set(row["problem"] for row in metrics)):
        plt.figure(figsize=(8, 5))
        for opt in sorted(set(row["optimizer"] for row in metrics)):
            rows = [r for r in metrics if r["problem"] == problem and r["optimizer"] == opt and (problem != "quadratic" or str(r["condition_number"]) == "1000")]
            by_step = defaultdict(list)
            for row in rows:
                by_step[int(row["step"])].append(float(row["loss"]))
            xs = sorted(by_step)
            ys = [mean(by_step[x]) for x in xs]
            if xs:
                plt.plot(xs, ys, label=opt)
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title(f"{problem} convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{problem}_loss_vs_steps.png", dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Difficult optimization benchmarks for DiagonalOptimiser.")
    parser.add_argument("--output-dir", default="results/difficult_optimization")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    aggregate_rows = run_suite(args.output_dir, seeds, args.max_iter)
    for row in aggregate_rows:
        print(row)


if __name__ == "__main__":
    main()
