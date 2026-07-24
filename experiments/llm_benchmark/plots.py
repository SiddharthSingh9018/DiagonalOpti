from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def _to_float(row, key):
    return float(row[key])


def _group_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _mean_curve(rows, x_key, y_key):
    by_x = defaultdict(list)
    for row in rows:
        by_x[_to_float(row, x_key)].append(_to_float(row, y_key))
    xs = sorted(by_x)
    ys = [mean(by_x[x]) for x in xs]
    sds = [stdev(by_x[x]) if len(by_x[x]) > 1 else 0.0 for x in xs]
    return xs, ys, sds


def _plot_curve(rows, x_key, y_key, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    for (optimizer,), group in sorted(_group_rows(rows, ["optimizer"]).items()):
        xs, ys, sds = _mean_curve(group, x_key, y_key)
        plt.plot(xs, ys, label=optimizer)
        if any(sd > 0 for sd in sds):
            lower = [y - sd for y, sd in zip(ys, sds)]
            upper = [y + sd for y, sd in zip(ys, sds)]
            plt.fill_between(xs, lower, upper, alpha=0.15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_all(metrics_rows, summary_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_duration = _group_rows(metrics_rows, ["duration_multiplier"])
    for (duration,), rows in sorted(by_duration.items(), key=lambda item: int(item[0][0])):
        prefix = output_dir / f"duration_{duration}x"
        _plot_curve(
            rows,
            "step",
            "val_loss",
            f"Validation loss vs steps ({duration}x)",
            "training step",
            "validation loss",
            prefix.with_name(prefix.name + "_loss_vs_steps.png"),
        )
        _plot_curve(
            rows,
            "wall_time_sec",
            "val_loss",
            f"Validation loss vs wall-clock time ({duration}x)",
            "seconds",
            "validation loss",
            prefix.with_name(prefix.name + "_loss_vs_time.png"),
        )
        _plot_curve(
            rows,
            "step",
            "grad_norm",
            f"Gradient norm stability ({duration}x)",
            "training step",
            "gradient norm",
            prefix.with_name(prefix.name + "_grad_norm.png"),
        )

    plt.figure(figsize=(9, 5))
    grouped = _group_rows(summary_rows, ["duration_multiplier", "optimizer"])
    labels = []
    values = []
    errors = []
    for (duration, optimizer), rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        labels.append(f"{optimizer}\\n{duration}x")
        vals = [_to_float(row, "best_val_loss") for row in rows]
        values.append(mean(vals))
        errors.append(stdev(vals) if len(vals) > 1 else 0.0)
    plt.errorbar(range(len(values)), values, yerr=errors, fmt="o", capsize=3)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("best validation loss")
    plt.title("Seed variance")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "seed_variance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for (optimizer,), rows in sorted(_group_rows(summary_rows, ["optimizer"]).items()):
        xs = [int(row["duration_multiplier"]) for row in rows]
        ys = [_to_float(row, "best_val_loss") for row in rows]
        by_x = defaultdict(list)
        for x, y in zip(xs, ys):
            by_x[x].append(y)
        xvals = sorted(by_x)
        yvals = [mean(by_x[x]) for x in xvals]
        plt.plot(xvals, yvals, marker="o", label=optimizer)
    plt.xlabel("training duration multiplier")
    plt.ylabel("best validation loss")
    plt.title("Convergence under longer training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_comparison.png", dpi=180)
    plt.close()
