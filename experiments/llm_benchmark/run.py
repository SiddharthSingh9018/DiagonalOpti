import argparse
import math
import random
import time
from pathlib import Path

import torch

from experiments.llm_benchmark.analysis import aggregate_summaries, markdown_table, summarize_run, write_csv, write_json
from experiments.llm_benchmark.config import config_to_dict, load_config, save_config
from experiments.llm_benchmark.data import get_batch, load_token_data
from experiments.llm_benchmark.model import TinyGPT, count_parameters
from experiments.llm_benchmark.optimizers import make_optimizer, optimizer_stats
from experiments.llm_benchmark.plots import plot_all


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


@torch.no_grad()
def evaluate(model, val_data, batch_size, block_size, device, eval_batches=8):
    model.eval()
    losses = []
    for idx in range(eval_batches):
        x, y = get_batch(val_data, batch_size, block_size, idx, device)
        _, loss = model(x, y)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def save_checkpoint(path: Path, model, optimizer, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_run(config, optimizer_name: str, seed: int, duration_multiplier: int, run_dir: Path, device: torch.device):
    set_seed(seed)
    train_data, val_data, dataset_used = load_token_data(config.data, config.model.vocab_size, seed)
    model = TinyGPT(**config_to_dict(config)["model"]).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = make_optimizer(
        optimizer_name,
        model,
        config.optimizers.learning_rates,
        config.optimizers.weight_decay,
    )

    max_steps = config.train.base_steps * duration_multiplier
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rows = []
    stable = True
    total_tokens = 0
    start_time = time.perf_counter()

    for step in range(1, max_steps + 1):
        step_start = time.perf_counter()
        x, y = get_batch(train_data, config.train.batch_size, config.model.block_size, step, device)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            _, closure_loss = model(x, y)
            closure_loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            return closure_loss

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        g_norm = grad_norm(model.parameters())
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        if optimizer_name == "DiagonalOptimiser":
            optimizer.step(closure=closure)
        else:
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_time = time.perf_counter() - step_start
        total_tokens += x.numel()
        train_loss = float(loss.detach().cpu())
        if not math.isfinite(train_loss) or not math.isfinite(g_norm):
            stable = False

        should_eval = step == 1 or step % config.train.eval_interval == 0 or step == max_steps or not stable
        if should_eval:
            val_loss = evaluate(model, val_data, config.train.batch_size, config.model.block_size, device)
            elapsed = time.perf_counter() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
            stats = optimizer_stats(optimizer)
            rows.append(
                {
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "duration_multiplier": duration_multiplier,
                    "dataset": dataset_used,
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "wall_time_sec": elapsed,
                    "step_time_sec": step_time,
                    "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
                    "peak_memory_mb": peak_memory,
                    "grad_norm": g_norm,
                    "learning_rate": current_lr(optimizer),
                    "stable": stable,
                    "curvature_updates": stats["curvature_updates"],
                    "hvp_probes": stats["hvp_probes"],
                }
            )
        if step % config.train.checkpoint_interval == 0 or step == max_steps or not stable:
            save_checkpoint(
                run_dir / "checkpoints" / f"step_{step}.pt",
                model,
                optimizer,
                {
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "duration_multiplier": duration_multiplier,
                    "step": step,
                    "stable": stable,
                    "parameter_count": count_parameters(model),
                },
            )
        if not stable:
            break

    return rows, summarize_run(rows, config.train.target_losses, config.train.wall_clock_checkpoints_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible TinyGPT optimizer benchmarks.")
    parser.add_argument("--config", default="configs/llm_benchmark.json", help="Path to JSON config.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional smoke-test limit on optimizer/seed/duration runs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    output_root = Path(config.output_dir) / config.experiment_name
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root / "config.resolved.json")

    all_metrics = []
    summaries = []
    run_count = 0
    print(f"Running {config.experiment_name} on {device}")
    for duration_multiplier in config.train.duration_multipliers:
        for optimizer_name in config.optimizers.names:
            for seed in config.seeds:
                if args.max_runs is not None and run_count >= args.max_runs:
                    break
                run_count += 1
                run_dir = output_root / "runs" / f"{optimizer_name}_seed{seed}_{duration_multiplier}x"
                print(f"[{run_count}] {optimizer_name} seed={seed} duration={duration_multiplier}x")
                rows, summary = train_run(config, optimizer_name, seed, duration_multiplier, run_dir, device)
                write_csv(run_dir / "metrics.csv", rows)
                write_json(run_dir / "summary.json", summary)
                all_metrics.extend(rows)
                summaries.append(summary)
            if args.max_runs is not None and run_count >= args.max_runs:
                break
        if args.max_runs is not None and run_count >= args.max_runs:
            break

    aggregate = aggregate_summaries(summaries)
    write_csv(output_root / "metrics.csv", all_metrics)
    write_csv(output_root / "summaries.csv", summaries)
    write_csv(output_root / "aggregate.csv", aggregate)
    write_json(output_root / "aggregate.json", aggregate)
    table = markdown_table(aggregate, target_key=f"steps_to_loss_{config.train.target_losses[1]}")
    (output_root / "results_table.md").write_text(table + "\n", encoding="utf-8")
    if not args.no_plots:
        plot_all(all_metrics, summaries, Path(config.output_dir) / "plots" / config.experiment_name)
    print(table)
    print(f"Saved results to {output_root}")


if __name__ == "__main__":
    main()
