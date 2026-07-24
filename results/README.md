# Results Directory

This directory is for generated benchmark outputs. Most result files are not
tracked by git because full runs can produce large CSVs, checkpoints, and plots.

Use this convention:

- `*_smoke/`: quick pipeline checks only. Do not cite as final evidence.
- `tuning/`: hyperparameter selection runs. Do not select from held-out eval seeds.
- final experiment folders: full multi-seed evaluation suitable for README or paper tables.

Recommended final runs:

```bash
python experiments/difficult_optimization.py --max-iter 500 --seeds 0,1,2,3,4
python -m experiments.llm_benchmark.run --config configs/llm_benchmark_colab_tuned.json
```

For publication-quality results, report mean +/- standard deviation across at
least 5 seeds and include all optimizer overhead in timing.
