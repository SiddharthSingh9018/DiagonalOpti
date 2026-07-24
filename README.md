# DiagonalCurvatureOptimizer

DiagonalOptimiser is a research-oriented curvature-aware optimizer. It uses
randomized low-rank Hessian approximation, spectral curvature directions,
trust-region control, and noise-aware stabilization.

The intended claim is deliberately narrow:

> DiagonalOptimiser is a lightweight curvature-aware optimizer designed to
> improve convergence and robustness on difficult optimization problems while
> remaining competitive on neural network training.

This repository should not be read as claiming that DiagonalOptimiser always
beats AdamW or is always faster.

## Benchmark Philosophy

The final benchmark suite is organized around three research questions.

**RQ1: Does DiagonalOptimiser help on difficult optimization landscapes?**

Use controlled objectives where curvature and conditioning are central:

- ill-conditioned quadratics with condition numbers 10, 100, 1000, 10000
- Rosenbrock-style nonconvex objectives
- optional difficult functions only when they reveal meaningful differences

**RQ2: Does DiagonalOptimiser work on neural network training?**

Use small but real training tasks:

- MLP/CNN on MNIST or FashionMNIST
- TinyGPT on WikiText-2, TinyStories, or a structured built-in text corpus

**RQ3: Is the computational overhead justified?**

Report full cost:

- optimizer step time
- curvature computation time when available
- Hessian-vector product/probe counts
- eigendecomposition/trust-region overhead for the NumPy optimizer path
- total wall-clock training time

## Final Notebook Suite

Use these notebooks for publication-facing analysis:

- `notebooks/01_synthetic_optimization.ipynb`
  - ill-conditioned quadratics
  - Rosenbrock/difficult landscape analysis
  - convergence and stability tables

- `notebooks/02_neural_network_training.ipynb`
  - MNIST/FashionMNIST MLP/CNN benchmark scaffold
  - intentionally refuses to report weak substitute results if dataset tooling is missing

- `notebooks/03_transformer_training.ipynb`
  - TinyGPT training
  - learning verification
  - throughput, memory, and optimizer overhead tables

- `notebooks/04_analysis.ipynb`
  - combined tables
  - combined interpretation
  - limitations and conclusions

Legacy exploratory experiments are kept under `archive/legacy_experiments/`.
They are not part of the final evidence because they are single-seed,
interactive, duplicated, or do not save complete metrics.

## Repository Structure

```text
DiagonalOpti/
├── optimizer/                  # Original NumPy DiagonalOptimiser implementation
├── baselines/                  # NumPy Adam and SGD baselines
├── experiments/
│   ├── difficult_optimization.py
│   └── llm_benchmark/          # TinyGPT benchmark package
├── configs/
│   ├── llm_benchmark.json
│   └── llm_benchmark_smoke.json
├── notebooks/
│   ├── 01_synthetic_optimization.ipynb
│   ├── 02_neural_network_training.ipynb
│   ├── 03_transformer_training.ipynb
│   └── 04_analysis.ipynb
├── archive/legacy_experiments/
├── paper/
├── results/                    # Generated outputs, not tracked
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

For the MNIST/FashionMNIST notebook, install torchvision too:

```bash
pip install torchvision
```

## Reproduce Experiments

Synthetic difficult-landscape benchmark:

```bash
python experiments/difficult_optimization.py --max-iter 500 --seeds 0,1,2,3,4
```

TinyGPT smoke test:

```bash
python -m experiments.llm_benchmark.run --config configs/llm_benchmark_smoke.json
```

TinyGPT full benchmark:

```bash
python -m experiments.llm_benchmark.run --config configs/llm_benchmark.json
```

The full TinyGPT config uses 5 seeds and duration multipliers of 1x, 3x, 5x,
and 10x. Smoke results are only pipeline checks; do not use them as scientific
evidence.

## Hyperparameter Tuning Protocol

Tune before reporting final neural-network or Transformer results, but keep the
procedure fair:

1. Use tuning seeds, for example `0,1`.
2. Sweep a small grid for every optimizer, not only DiagonalOptimiser.
3. Select hyperparameters using validation loss from the tuning runs.
4. Freeze the selected hyperparameters.
5. Report final results on held-out evaluation seeds, for example `2,3,4,5,6`.

TinyGPT learning-rate sweep:

```bash
python -m experiments.llm_benchmark.tune \
  --config configs/llm_benchmark.json \
  --steps 200 \
  --seeds 0,1
```

This writes:

- `results/tuning/llm_benchmark/trials.csv`
- `results/tuning/llm_benchmark/best_by_optimizer.json`
- `results/tuning/llm_benchmark/llm_benchmark_tuned.json`

Use the tuned config for evaluation only after the tuning decision is fixed.
Do not choose the best result from the final evaluation seeds.

## What To Show On GitHub

The repository can include smoke plots as pipeline examples, but label them
clearly as smoke tests. Do not present one-seed or five-step runs as benchmark
evidence.

Recommended GitHub figures after full runs:

- synthetic optimization convergence by condition number
- final objective vs condition number with mean +/- std
- TinyGPT validation loss vs steps and wall-clock time
- optimizer overhead table including DiagonalOptimiser curvature/HVP counts

Current smoke outputs are useful for verifying the benchmark code, not for
scientific claims.

## Outputs

Each benchmark writes CSV/JSON outputs under `results/`.

Common files:

- `metrics.csv`
- `summaries.csv`
- `aggregate.csv`
- `aggregate.json`
- plots under a `plots/` directory

The TinyGPT runner also writes:

- per-run metrics under `results/<experiment>/runs/`
- checkpoints
- `results_table.md`

## Metrics

Every final benchmark should report:

- mean +/- standard deviation across at least 5 seeds
- training or objective loss curves
- validation loss curves where applicable
- steps or iterations to target loss
- final and best loss
- wall-clock time
- throughput for neural benchmarks
- memory usage for neural benchmarks
- stability/failure counts
- DiagonalOptimiser curvature/HVP/probe counts where available

## Limitations

The original DiagonalOptimiser is a NumPy function optimizer over `f(x)` and
`grad_f(x)`. It is evaluated directly in the synthetic difficult-landscape
benchmarks.

The TinyGPT benchmark uses a PyTorch-compatible diagonal-curvature variant for
model training. Treat it as neural-training evidence for the optimizer idea,
not as a replacement for direct evaluation of the original S/A-RSVD-TR NumPy
algorithm.

Small-scale benchmarks do not prove large-scale optimizer superiority. Report
where DiagonalOptimiser helps, where it is competitive, and where its overhead
is not justified.
