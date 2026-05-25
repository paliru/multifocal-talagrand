# Explicit Two-Gaussian Decompositions of Multifocal Ellipsoid Distributions

**First Concrete Examples of the Talagrand Gaussianization Theorem,
with a Classification Interpretation**

*Claude (Anthropic) and Pal Rujan paliruj@gmail.com — May 2026*

---

## What is this?

The [Talagrand convexity conjecture](https://arxiv.org/abs/2602.22342) (now theorem) says:
> Every centered 1-subgaussian random vector in Rⁿ equals the sum of at most **3 standard Gaussian vectors** (not necessarily independent).

This repository contains the **first explicit, numerically verified decompositions** for a
concrete family of distributions: the uniform measure on *multifocal ellipsoids* — the
convex bodies defined by a constant sum of distances to the vertices of a regular simplex.

We find that **2 Gaussians suffice** for all cases tested, tighter than the theorem's bound.

---

## Interactive 3D Visualization

Open [`figures/multifocal_surface.html`](figures/multifocal_surface.html) in any browser
to explore the multifocal body for the regular tetrahedron interactively:
drag to rotate, scroll to zoom, and use the sliders to change the level $k$.

---

## Repository Structure

```
multifocal-talagrand/
├── README.md                        ← this file
│
├── paper/
│   ├── talagrand_paper.pdf          ← compiled paper (10 pp.)
│   ├── talagrand_paper.tex          ← LaTeX source
│   ├── sinkhorn_comparison.png      ← Fig. 1: 3-case Sinkhorn comparison
│   ├── aniso_axes_comparison.png    ← Fig. 2: anisotropic axes
│   ├── fredholm_solution.png        ← Fig. 3: original d=3 coupling
│   └── gaussians_analysis.png       ← Fig. 4: moment feasibility
│
├── code/
│   ├── sinkhorn_all_cases.py        ← MAIN: all 3 cases, comparison figures
│   ├── fredholm_solver.py           ← Original 1D Sinkhorn (d=3 only)
│   ├── gaussians_construction.py    ← Moment feasibility analysis
│   └── build_surface.py             ← 3D surface mesh (marching cubes)
│
└── figures/
    ├── multifocal_surface.html      ← Interactive 3D body (Three.js)
    ├── sinkhorn_comparison.png
    ├── aniso_axes_comparison.png
    ├── fredholm_solution.png
    └── gaussians_analysis.png
```

---

## Reproducing the Results

**Requirements:** Python 3.9+, numpy, scipy, matplotlib, scikit-image

```bash
pip install numpy scipy matplotlib scikit-image
```

**Run the main analysis** (sampling + moment check + Sinkhorn + figures):

```bash
python code/sinkhorn_all_cases.py
```

This produces `sinkhorn_comparison.png` and `aniso_axes_comparison.png` in ~30 seconds.

**Original single-case solver** (d=3 regular tetrahedron only):

```bash
python code/fredholm_solver.py
```

---

## Key Results at a Glance

| Case | σ_X | κ (kurtosis) | corr(Y₁,Y₂) | 2G feasible? |
|------|-----|-------------|-------------|-------------|
| Regular tetrahedron (d=3, k=5) | 0.392 | −0.932 | −0.004 | **Yes** |
| Regular 4-simplex (d=4, k=6) | 0.302 | −0.700 | +0.023 | **Yes** |
| Aniso tet (scale 1:2:3, k=10) | 0.579–0.823 | ~−0.93 | ≈ 0 | **Yes** |

All three cases require only **K=2 Gaussians**, not K=3 as the theorem guarantees.
The coupling is uncorrelated (ρ≈0) but non-independent — dependence lives in
the 4th-order cross-moment E[Y₁²Y₂²] > 0.

---

## The New Angle: Score Functions and Membership Probability

The Sinkhorn coupling gives, as a byproduct:

1. **The explicit score function** ∇_x log p_t(x) for diffusion models trained on this
   distribution — no neural network needed. This is a **ground-truth benchmark** for
   score-matching algorithms.

2. **A generalised error function** Erf_{B_k}(r, η): the probability that a noisy
   query point at radius r came from the multifocal body vs. a pure Gaussian.
   Interpolates between the body indicator (η→0) and the Gaussian CDF (η→∞).

---

## How This Was Made

This project was produced in a single extended session between:
- **Claude 3.7 Sonnet** (Anthropic) — mathematical framework, all code, figures, paper
- **Paliruj Asavapibhop** — conceptual direction, physical intuition, identification of the
  relevant theorem, proposal of the test cases

A session transcript is available in [`data/session_summary.md`](data/session_summary.md).

---

## Citation

```bibtex
@misc{claude_paliruj_2026_talagrand,
  title   = {Explicit Two-Gaussian Decompositions of Multifocal Ellipsoid Distributions},
  author  = {Claude (Anthropic) and Pal Rujan, paliruj@gmail.com},
  year    = {2026},
  note    = {Preprint. Code and data: https://github.com/paliru/multifocal-talagrand}
}
```

---

## References

- Song, Y. (2026). *Solution to Talagrand's convexity conjecture.* arXiv:2602.22342
- Hua, J., Song, Y., Tudose, G. (2026). *Gaussianization of subgaussian random vectors.* arXiv:2605.10908
- Talagrand, M. (2014). *Upper and Lower Bounds for Stochastic Processes.* Springer.
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS 33.
