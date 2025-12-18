# DiagonalOpti


DiagonalOpti explores a curvature-aware optimization method based on
low-rank randomized Hessian approximation with explicit separation of
symmetric and antisymmetric curvature components.

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

DiagonalOpti/
├── optimizer/
│ ├── optimizer.py # S/A RSVD TR optimizer
│ ├── curvature.py # curvature decomposition utilities
│ ├── utils.py # trust-region and projection helpers
│
├── baselines/
│ ├── sgd.py
│ ├── adam.py
│ └── rmsprop.py
│
├── experiments/
│ ├── synthetic_benchmarks.py # quadratics + Rosenbrock
│ └── logistic_regression.py # convex ML objective
│
├── paper/ # draft notes / writeup
└── results/ # plots (not tracked)

## Benchmarks

The optimizer is evaluated against standard baselines on:

- Well-conditioned quadratic objectives
- Ill-conditioned quadratic objectives
- Rosenbrock function (nonconvex)
- Binary logistic regression (convex ML task)

All experiments use identical initializations and objective definitions
to ensure fair comparison.

---

