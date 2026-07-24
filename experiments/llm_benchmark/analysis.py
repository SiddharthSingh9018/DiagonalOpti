import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return math.nan
    return float(value)


def steps_to_targets(rows: list[dict[str, Any]], targets: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    sorted_rows = sorted(rows, key=lambda row: int(row["step"]))
    for target in targets:
        hit = next((row for row in sorted_rows if _float(row, "val_loss") <= target), None)
        out[f"steps_to_loss_{target}"] = int(hit["step"]) if hit is not None else None
    return out


def loss_at_wall_times(rows: list[dict[str, Any]], wall_times: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    sorted_rows = sorted(rows, key=lambda row: _float(row, "wall_time_sec"))
    for wall_time in wall_times:
        candidates = [row for row in sorted_rows if _float(row, "wall_time_sec") <= wall_time]
        out[f"loss_at_{wall_time:g}s"] = _float(candidates[-1], "val_loss") if candidates else None
    return out


def area_under_curve(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    sorted_rows = sorted(rows, key=lambda row: _float(row, x_key))
    if len(sorted_rows) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(sorted_rows[:-1], sorted_rows[1:]):
        x0 = _float(left, x_key)
        x1 = _float(right, x_key)
        y0 = _float(left, y_key)
        y1 = _float(right, y_key)
        if math.isfinite(x0) and math.isfinite(x1) and math.isfinite(y0) and math.isfinite(y1):
            area += 0.5 * (y0 + y1) * max(x1 - x0, 0.0)
    return area


def summarize_run(rows: list[dict[str, Any]], targets: list[float], wall_times: list[float]) -> dict[str, Any]:
    last = max(rows, key=lambda row: int(row["step"]))
    best = min(rows, key=lambda row: _float(row, "val_loss"))
    summary: dict[str, Any] = {
        "optimizer": last["optimizer"],
        "seed": int(last["seed"]),
        "duration_multiplier": int(last["duration_multiplier"]),
        "final_train_loss": _float(last, "train_loss"),
        "final_val_loss": _float(last, "val_loss"),
        "best_val_loss": _float(best, "val_loss"),
        "best_step": int(best["step"]),
        "wall_time_sec": _float(last, "wall_time_sec"),
        "tokens_per_sec": _float(last, "tokens_per_sec"),
        "peak_memory_mb": _float(last, "peak_memory_mb"),
        "last_grad_norm": _float(last, "grad_norm"),
        "last_lr": _float(last, "learning_rate"),
        "val_loss_auc_steps": area_under_curve(rows, "step", "val_loss"),
        "val_loss_auc_time": area_under_curve(rows, "wall_time_sec", "val_loss"),
        "stable": str(last["stable"]).lower() == "true",
        "curvature_updates": _float(last, "curvature_updates"),
        "hvp_probes": _float(last, "hvp_probes"),
    }
    summary.update(steps_to_targets(rows, targets))
    summary.update(loss_at_wall_times(rows, wall_times))
    return summary


def aggregate_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["optimizer"], int(row["duration_multiplier"]))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    numeric_keys = [
        "final_val_loss",
        "best_val_loss",
        "wall_time_sec",
        "tokens_per_sec",
        "peak_memory_mb",
        "last_grad_norm",
        "last_lr",
        "val_loss_auc_steps",
        "val_loss_auc_time",
        "curvature_updates",
        "hvp_probes",
    ]
    target_keys = sorted([key for key in summaries[0].keys() if key.startswith("steps_to_loss_")]) if summaries else []
    wall_keys = sorted([key for key in summaries[0].keys() if key.startswith("loss_at_")]) if summaries else []
    for (optimizer, multiplier), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "optimizer": optimizer,
            "duration_multiplier": multiplier,
            "runs": len(rows),
            "stable_runs": sum(1 for row in rows if row["stable"]),
        }
        for key in numeric_keys + wall_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
            out[f"{key}_mean"] = mean(values) if values else None
            out[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        for key in target_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            out[f"{key}_mean"] = mean(values) if values else None
            out[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
            out[f"{key}_hits"] = len(values)
        aggregate_rows.append(out)
    return aggregate_rows


def markdown_table(aggregate_rows: list[dict[str, Any]], target_key: str = "steps_to_loss_2.8") -> str:
    lines = [
        "| Optimizer | Duration | Final Loss | Best Loss | Steps to Target | Time | Tokens/sec | Stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        def pm(key: str, precision: int = 3) -> str:
            avg = row.get(f"{key}_mean")
            sd = row.get(f"{key}_std")
            if avg is None:
                return "n/a"
            return f"{avg:.{precision}f} +/- {sd:.{precision}f}"

        steps = pm(target_key, 1)
        lines.append(
            "| {optimizer} | {duration_multiplier}x | {final} | {best} | {steps} | {time} | {tok} | {stable}/{runs} |".format(
                optimizer=row["optimizer"],
                duration_multiplier=row["duration_multiplier"],
                final=pm("final_val_loss"),
                best=pm("best_val_loss"),
                steps=steps,
                time=pm("wall_time_sec", 2),
                tok=pm("tokens_per_sec", 0),
                stable=row["stable_runs"],
                runs=row["runs"],
            )
        )
    return "\n".join(lines)
