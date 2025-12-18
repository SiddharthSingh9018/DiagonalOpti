# DiagonalCurvatureOptimizer


The optimizer leverages the fact that performing updates in a diagonalized
curvature basis is significantly cheaper than operating in the full parameter space.The
optimization algorithm that combines low-rank curvature estimation, trust region control, and noiseaware
step-size stabilization to improve robustness under minibatch stochasticity. Unlike Adam or
momentum-based methods, Diagonal((S/A RSVD TR)) explicitly models local curvature through a rank-k eigenspace
approximation of the Hessian, enabling a computationally efficient Newton-like step in that subspace
while preserving adaptive diagonal scaling elsewhere. Diagonal((S/A RSVD TR)) also incorporates a trust-region
mechanism based on the ratio of predicted to actual decrease, and introduces the antisymmetric
curvature floor, a technique to prevent step collapse when Hessian estimates are noisy
The project focuses on understanding how local curvature structure and
non-symmetric effects influence optimization dynamics, especially on
ill-conditioned and nonconvex objectives.

This is a research-oriented implementation, not a production library.

---

## Core Idea

The proposed optimizer (S/A RSVD TR):

- Approximates curvature using randomized low-rank Hessian sketches
- Separates symmetric (useful curvature) and antisymmetric (noise / instability) components
- Applies noise-aware eigenvalue damping
- Uses a trust-region mechanism with hard step control
- Falls back to gradient steps when curvature information becomes unreliable

The goal is stability and interpretability rather than raw speed.

---

## Repository Structure
<pre>
DiagonalOpti/
├── optimizer/
│   ├── optimizer.py      # S/A–RSVD–TR curvature-aware optimizer
│   ├── curvature.py      # symmetric / antisymmetric curvature utilities
│   └── utils.py          # trust-region projection and helper routines
│
├── baselines/
│   ├── sgd.py            # SGD with momentum
│   ├── adam.py           # Adam optimizer
│       
│
├── experiments/
│   ├── synthetic_benchmarks.py   # quadratics (well/ill-conditioned) + Rosenbrock
│   └── logistic_regression.py    # convex ML benchmark
│
├── paper/                # draft notes / write-up
├── results/              # plots and outputs (not tracked)
├── .gitignore
└── README.md

</pre>

## Benchmarks

The optimizer is evaluated against standard baselines on:

- Well-conditioned quadratic objectives
- <img width="738" height="619" alt="image" src="https://github.com/user-attachments/assets/32c6dc7a-7aa9-4b75-a0b7-d718a25aaee7" />

- Ill-conditioned quadratic objectives
- <img width="733" height="615" alt="image" src="https://github.com/user-attachments/assets/7aaba51b-4d19-4b05-89cc-6724914b99e5" />
  
- Rosenbrock function
- <img width="720" height="615" alt="image" src="https://github.com/user-attachments/assets/1af0376d-dad7-408d-b6d9-ef3662f9298f" />

  
- Binary logistic regression (convex ML task)
- <img width="615" height="475" alt="image" src="https://github.com/user-attachments/assets/2d58857f-f0e4-4d8d-a659-5f944f2da86a" />


All experiments use identical initializations and objective definitions
to ensure fair comparison.

---

