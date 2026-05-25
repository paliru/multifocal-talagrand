"""
Explicit 2-Gaussian Construction via the Fredholm Integral Equation
====================================================================
We want a joint density g(u,v) on R^2 with:
  (A) row marginal:      integral_v g(u,v)dv = phi_s(u)   [Y1 ~ N(0,s^2)]
  (B) column marginal:   integral_u g(u,v)du = phi_s(v)   [Y2 ~ N(0,s^2)]
  (C) anti-diag:         integral_u g(u,x-u)du = f_X1(x)  [Y1+Y2 ~ X_1]

This is a 3-marginal coupling problem. We solve it with the
Generalized Sinkhorn (alternating projections) algorithm.
"""

import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── 1. Sample X from multifocal body and get 1D marginal ──────────────────
s3 = np.sqrt(3)
TET = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float) / s3

def sample_body(k, n=200_000, R=2.2):
    pts = np.random.uniform(-R, R, (n*6, 3))
    pts = pts[np.linalg.norm(pts, axis=1) <= R]
    sod = np.sum(np.linalg.norm(pts[:,None,:] - TET[None,:,:], axis=2), axis=1)
    return pts[sod <= k]

k = 5.0
print(f"Sampling B_k for k={k}...")
X3d = sample_body(k)
X1 = X3d[:, 0]          # 1D projection
sigma2_X = np.var(X1)
sigma_X  = np.sqrt(sigma2_X)
s = sigma_X / np.sqrt(2)  # each Gaussian gets half the variance
print(f"  N={len(X3d)}, sigma_X1={sigma_X:.4f}, s={s:.4f}")

# KDE of the 1D marginal density f_X1
kde_X1 = gaussian_kde(X1, bw_method=0.06)

# ── 2. Discrete grid ───────────────────────────────────────────────────────
L  = 3.0 * s          # grid half-width: covers ±3s comfortably
N  = 180              # grid points per axis
u  = np.linspace(-L, L, N)
du = u[1] - u[0]

# Target densities (as probability masses, i.e., density × du)
phi = norm.pdf(u, 0, s) * du   # Gaussian mass vector, shape (N,)
phi /= phi.sum()                 # exact normalise

# Sum values: u[i]+u[j] ranges over [-2L, 2L] with step du
# Anti-diagonal k means i+j = k, sum = u[0]+u[0]+k*du = -2L + k*du
n_diag = 2*N - 1
sum_vals = u[0]*2 + np.arange(n_diag) * du   # shape (2N-1,)

# Target sum density (evaluated at sum grid, then normalised)
# Guard against KDE evaluation far outside support
f_sum_raw = np.maximum(kde_X1(sum_vals), 0)
# Zero-out where |sum| > support extent of X1
support_edge = np.max(np.abs(X1)) * 1.05
f_sum_raw[np.abs(sum_vals) > support_edge] = 0.0
# Normalise to probability mass
f_sum = f_sum_raw * du
f_sum /= f_sum.sum()

# Pre-compute which (i,j) pairs belong to each anti-diagonal
antidiag = [[] for _ in range(n_diag)]
for i in range(N):
    for j in range(N):
        antidiag[i+j].append((i, j))

print(f"Grid: N={N}, L={L:.3f}, du={du:.4f}, {n_diag} anti-diagonals")

# ── 3. Initialise P as product (independent) coupling ────────────────────
P = np.outer(phi, phi)   # shape (N, N);  sum = 1.0 approximately

# ── 4. Three-way Sinkhorn ─────────────────────────────────────────────────
def sinkhorn_3way(P, phi, phi_col, f_sum, antidiag, n_iter=300, tol=1e-6):
    P = P.copy()
    errs = []
    for iteration in range(n_iter):
        # --- (C) anti-diagonal projection ---
        for k_idx, pairs in enumerate(antidiag):
            if not pairs or f_sum[k_idx] < 1e-15:
                for (i,j) in pairs:
                    P[i,j] = 0.0
                continue
            cur = sum(P[i,j] for (i,j) in pairs)
            if cur < 1e-30:
                continue
            scale = f_sum[k_idx] / cur
            for (i,j) in pairs:
                P[i,j] *= scale

        # --- (A) row projection: make row sums = phi ---
        row_sums = P.sum(axis=1)           # shape (N,)
        row_sums = np.where(row_sums < 1e-30, 1e-30, row_sums)
        P *= (phi / row_sums)[:, None]

        # --- (B) column projection: make col sums = phi ---
        col_sums = P.sum(axis=0)
        col_sums = np.where(col_sums < 1e-30, 1e-30, col_sums)
        P *= (phi_col / col_sums)[None, :]

        # Convergence: max deviation in row and col marginals
        err_row = np.max(np.abs(P.sum(axis=1) - phi))
        err_col = np.max(np.abs(P.sum(axis=0) - phi_col))
        err = max(err_row, err_col)
        errs.append(err)
        if err < tol and iteration > 10:
            print(f"  Converged at iteration {iteration+1}, err={err:.2e}")
            break
        if (iteration+1) % 50 == 0:
            print(f"  iter {iteration+1}: err={err:.2e}")
    return P, errs

print("Running 3-way Sinkhorn...")
P_opt, errs = sinkhorn_3way(P, phi, phi, f_sum, antidiag, n_iter=400, tol=5e-6)
print(f"  Final mass: {P_opt.sum():.6f}")

# ── 5. Verify marginals and sum distribution ──────────────────────────────
row_marginal  = P_opt.sum(axis=1) / du    # recover density
col_marginal  = P_opt.sum(axis=0) / du

# Empirical sum density from P_opt
sum_density = np.zeros(n_diag)
for k_idx, pairs in enumerate(antidiag):
    sum_density[k_idx] = sum(P_opt[i,j] for (i,j) in pairs) / du

# ── 6. Visualise ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (a) Joint density g(u,v) = P_opt / du^2
ax0 = fig.add_subplot(gs[0, 0])
g_joint = P_opt / du**2
im = ax0.imshow(g_joint, origin='lower', extent=[-L,L,-L,L],
                cmap='Blues', aspect='auto',
                vmin=0, vmax=np.percentile(g_joint, 99))
ax0.set_xlabel('Y₁'); ax0.set_ylabel('Y₂')
ax0.set_title('Joint density g(Y₁, Y₂)\n[2-Gaussian coupling]', fontsize=10)
ax0.axhline(0, color='gray', lw=0.5, ls='--')
ax0.axvline(0, color='gray', lw=0.5, ls='--')
# Anti-diagonals = contours of Y1+Y2
for xval in [-1.0, -0.5, 0, 0.5, 1.0]:
    vv = np.array([xval - L, xval + L])
    uu = np.array([-L, L])
    mask = (vv >= -L) & (vv <= L)
    if mask.any():
        ax0.plot(uu[mask], vv[mask], 'r-', lw=0.6, alpha=0.5)
fig.colorbar(im, ax=ax0, shrink=0.7)

# (b) Y1 marginal verification
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(u, row_marginal, 'b-', lw=2, label='Y₁ marginal (Sinkhorn)')
ax1.plot(u, norm.pdf(u, 0, s), 'k--', lw=1.5, label=f'N(0,{s:.3f}²) target')
ax1.fill_between(u, row_marginal, norm.pdf(u, 0, s), alpha=0.3, color='blue')
ax1.set_xlabel('u'); ax1.set_ylabel('density')
ax1.set_title('Y₁ marginal: Gaussian check', fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# (c) Sum (Y1+Y2) verification
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(sum_vals, sum_density, 'b-', lw=2, label='Y₁+Y₂ density (Sinkhorn)')
ax2.plot(sum_vals, kde_X1(sum_vals), 'r--', lw=1.5, label='f_X₁ target (KDE)')
ax2.plot(sum_vals, norm.pdf(sum_vals, 0, sigma_X), 'g:', lw=1.5,
         label='Best Gaussian (reference)')
ax2.set_xlabel('x'); ax2.set_ylabel('density')
ax2.set_title('Y₁+Y₂ sum: matches X₁?', fontsize=10)
ax2.set_xlim(-2, 2)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# (d) Convergence of Sinkhorn
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(errs, 'b-', lw=1.5)
ax3.set_xlabel('Iteration'); ax3.set_ylabel('Max marginal error')
ax3.set_title('Sinkhorn convergence', fontsize=10)
ax3.grid(True, alpha=0.3)

# (e) Conditional distribution E[Y1 | Y1+Y2 = x]  and spread
ax4 = fig.add_subplot(gs[1, 1])
cond_mean  = np.zeros(n_diag)
cond_std   = np.zeros(n_diag)
for k_idx, pairs in enumerate(antidiag):
    if not pairs: continue
    mass = np.array([P_opt[i,j] for (i,j) in pairs])
    tot  = mass.sum()
    if tot < 1e-20: continue
    vals = np.array([u[i] for (i,j) in pairs])
    cond_mean[k_idx] = (vals * mass).sum() / tot
    cond_std[k_idx]  = np.sqrt(((vals - cond_mean[k_idx])**2 * mass).sum() / tot)

valid = f_sum > 1e-8 * f_sum.max()
ax4.plot(sum_vals[valid], cond_mean[valid], 'b-', lw=2,
         label='E[Y₁ | Y₁+Y₂=x]')
ax4.fill_between(sum_vals[valid],
                 cond_mean[valid] - cond_std[valid],
                 cond_mean[valid] + cond_std[valid],
                 alpha=0.3, color='blue', label='±σ(Y₁|Y₁+Y₂=x)')
ax4.plot(sum_vals, sum_vals/2, 'k--', lw=1, label='x/2 (symmetric split)')
ax4.set_xlabel('x = Y₁+Y₂'); ax4.set_ylabel('Y₁')
ax4.set_title('Conditional structure:\nE[Y₁ | Y₁+Y₂=x]', fontsize=10)
ax4.set_xlim(-1.6, 1.6)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# (f) Correlation structure: scatter of (Y1,Y2) samples
ax5 = fig.add_subplot(gs[1, 2])
# Sample from P_opt
flat = P_opt.flatten()
flat = np.maximum(flat, 0)
flat /= flat.sum()
idx_samples = np.random.choice(N*N, size=3000, p=flat)
i_s, j_s = np.divmod(idx_samples, N)
y1_s = u[i_s] + np.random.randn(3000)*du*0.3
y2_s = u[j_s] + np.random.randn(3000)*du*0.3
ax5.scatter(y1_s, y2_s, s=2, alpha=0.3, c=y1_s+y2_s, cmap='RdBu_r')
corr = np.corrcoef(y1_s, y2_s)[0,1]
ax5.set_xlabel('Y₁'); ax5.set_ylabel('Y₂')
ax5.set_title(f'Samples from coupling g(Y₁,Y₂)\ncorr(Y₁,Y₂)={corr:.3f}', fontsize=10)
ax5.axhline(0, color='gray', lw=0.5, ls='--')
ax5.axvline(0, color='gray', lw=0.5, ls='--')
ax5.grid(True, alpha=0.2)

fig.suptitle(
    f'2-Gaussian Coupling for Multifocal Body (k={k})\n'
    r'Find g(Y₁,Y₂) with Gaussian marginals N(0,s²) and Y₁+Y₂ ~$X_1$',
    fontsize=12)

plt.savefig('/sessions/adoring-happy-cerf/mnt/outputs/fredholm_solution.png',
            dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary statistics
print(f"\nSummary:")
print(f"  corr(Y1,Y2) = {corr:.4f}  (0 = independent, -1 = fully anti-correlated)")
print(f"  Row marginal max err  = {np.max(np.abs(P_opt.sum(axis=1)/du - norm.pdf(u,0,s))):.4e}")
print(f"  Col marginal max err  = {np.max(np.abs(P_opt.sum(axis=0)/du - norm.pdf(u,0,s))):.4e}")
# Sum marginal L1 error
sum_l1 = np.sum(np.abs(sum_density - kde_X1(sum_vals))) * du
print(f"  Sum marginal L1 error = {sum_l1:.4f}")
print("Plot saved: fredholm_solution.png")
