import argparse
import itertools
import json
from pathlib import Path

from experiments.llm_benchmark.analysis import write_csv, write_json
from experiments.llm_benchmark.config import config_to_dict, load_config
from experiments.llm_benchmark.run import resolve_device, train_run


DEFAULT_LR_GRID = {
    "AdamW": [1e-4, 3e-4, 1e-3],
    "Lion": [3e-5, 1e-4, 3e-4],
    "Sophia": [1e-4, 3e-4, 1e-3],
    "Muon": [1e-4, 2e-4, 5e-4],
    "DiagonalOptimiser": [1e-4, 3e-4, 1e-3],
}


def clone_config_with_lr(config, optimizer_name, lr, tuning_steps, tuning_seeds):
    data = config_to_dict(config)
    data["experiment_name"] = f"{config.experiment_name}_tune_{optimizer_name}_lr{lr:g}".replace(".", "p")
    data["seeds"] = tuning_seeds
    data["train"]["base_steps"] = tuning_steps
    data["train"]["duration_multipliers"] = [1]
    data["train"]["checkpoint_interval"] = tuning_steps
    data["optimizers"]["names"] = [optimizer_name]
    data["optimizers"]["learning_rates"][optimizer_name] = lr
    return data


def config_from_dict(data):
    tmp_path = Path("results") / "_tmp_tune_config.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return load_config(tmp_path)


def run_tuning(config_path, output_dir, tuning_steps, tuning_seeds, optimizers):
    base_config = load_config(config_path)
    device = resolve_device(base_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    best_by_optimizer = {}
    for optimizer_name in optimizers:
        best_row = None
        for lr in DEFAULT_LR_GRID[optimizer_name]:
            trial_config = config_from_dict(clone_config_with_lr(base_config, optimizer_name, lr, tuning_steps, tuning_seeds))
            seed_summaries = []
            for seed in tuning_seeds:
                run_dir = output_dir / "runs" / f"{optimizer_name}_lr{lr:g}_seed{seed}".replace(".", "p")
                _, summary = train_run(trial_config, optimizer_name, seed, 1, run_dir, device)
                seed_summaries.append(summary)
            mean_best = sum(row["best_val_loss"] for row in seed_summaries) / len(seed_summaries)
            mean_final = sum(row["final_val_loss"] for row in seed_summaries) / len(seed_summaries)
            row = {
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "tuning_steps": tuning_steps,
                "seeds": ",".join(str(seed) for seed in tuning_seeds),
                "mean_best_val_loss": mean_best,
                "mean_final_val_loss": mean_final,
            }
            trial_rows.append(row)
            if best_row is None or row["mean_best_val_loss"] < best_row["mean_best_val_loss"]:
                best_row = row
        best_by_optimizer[optimizer_name] = best_row

    tuned_config = config_to_dict(base_config)
    tuned_config["experiment_name"] = f"{base_config.experiment_name}_tuned"
    for optimizer_name, row in best_by_optimizer.items():
        tuned_config["optimizers"]["learning_rates"][optimizer_name] = row["learning_rate"]

    write_csv(output_dir / "trials.csv", trial_rows)
    write_json(output_dir / "best_by_optimizer.json", best_by_optimizer)
    write_json(output_dir / "llm_benchmark_tuned.json", tuned_config)
    return trial_rows, best_by_optimizer


def main():
    parser = argparse.ArgumentParser(description="Tune TinyGPT optimizer learning rates on short validation runs.")
    parser.add_argument("--config", default="configs/llm_benchmark.json")
    parser.add_argument("--output-dir", default="results/tuning/llm_benchmark")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--optimizers", default="AdamW,Lion,Sophia,Muon,DiagonalOptimiser")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    optimizers = [item.strip() for item in args.optimizers.split(",") if item.strip()]
    _, best = run_tuning(args.config, args.output_dir, args.steps, seeds, optimizers)
    for optimizer_name, row in best.items():
        print(f"{optimizer_name}: lr={row['learning_rate']} mean_best_val_loss={row['mean_best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
