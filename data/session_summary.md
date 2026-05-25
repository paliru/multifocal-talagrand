# Session Summary: Human–AI Mathematical Exploration

**Date:** 24 May 2026  
**Participants:** Claude 3.7 Sonnet (Anthropic) and Paliruj Asavapibhop  
**Duration:** Single extended Cowork session  

## Conceptual arc

1. **Starting question (human):** How does the classical ellipse (sum of distances to two foci = constant) generalise to 3D and higher dimensions?

2. **Identification of the theorem (human):** Paliruj recognised the connection to Talagrand's convexity conjecture — recently solved by Song (arXiv:2602.22342) and Hua–Song–Tudose (arXiv:2605.10908). The key insight: a multifocal body is exactly the kind of subgaussian distribution the theorem applies to.

3. **3D visualisation (AI):** Interactive Three.js rendering of the multifocal body for the regular tetrahedron, computed via marching cubes in Python. Fixed a JS-side triTable corruption by moving all mesh computation to Python.

4. **Symmetry reduction (AI):** Applied Schur's lemma to the S₄ group action on the regular tetrahedron, showing each Gaussian component must be isotropic N(0, σ²I). This reduced a matrix-valued problem to a scalar problem.

5. **Fredholm integral equation (AI):** Recognised that finding the coupling g(Y₁,Y₂) is a three-marginal optimal transport problem; solved with a three-way Sinkhorn algorithm. Converged to < 10⁻⁵ marginal error.

6. **Extension to d=4 and anisotropic cases (human direction, AI execution):** Paliruj proposed two extensions. AI adapted the hit-and-run MCMC sampler for d=4 (rejection sampling fails in 4D), ran Sinkhorn for all axes, verified isotropy.

7. **Surprise result:** All cases require only K=2 Gaussians, beating the theorem's bound of K=3.

8. **Classification and score functions (human request, AI derivation):** Paliruj asked for a generalisation of the error function. AI derived: (a) the noisy membership probability Π(x̃, η), (b) the explicit score function as a Gaussian mixture from the Sinkhorn grid, (c) the connection to Tweedie's formula and diffusion models.

9. **LaTeX paper (AI):** Full 13-page paper with theorem/proof environments, tables, figures, appendix.

## Key files generated in this session

| File | Description |
|------|-------------|
| `code/sinkhorn_all_cases.py` | Main analysis script (sampling + Sinkhorn + figures) |
| `code/fredholm_solver.py` | Original d=3 solver |
| `figures/multifocal_surface.html` | Interactive 3D body |
| `figures/sinkhorn_comparison.png` | Main result figure (3 cases × 4 panels) |
| `figures/aniso_axes_comparison.png` | Anisotropic tetrahedron, 3 axes |
| `paper/talagrand_paper.pdf` | Final compiled paper |

## Reproducibility note

All random seeds are fixed (`np.random.seed(42)` in the main script). The only non-deterministic element is the hit-and-run MCMC for d=4 with seed=2; results are stable across runs to the precision shown in the paper tables.
