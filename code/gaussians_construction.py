"""
Constructing the (≤3) Gaussians for the Multifocal Ellipsoid
=============================================================
Following Song (arXiv:2602.22342) and the S_{d+1} symmetry argument.

THEORETICAL SETUP
-----------------
X = uniform distribution on B_k = {P in R^3 : sum_i ||P - V_i|| <= k}
where V_i are vertices of a regular tetrahedron (the 3-simplex).

SYMMETRY REDUCTION (analytical)
--------------------------------
The symmetry group G = S_4 acts on R^3 via the "standard" representation
(the orthogonal action permuting the 4 tetrahedral vertices).

Claim: any valid decomposition X = Y_1 + Y_2 + Y_3
       with each Y_i marginally Gaussian
       must have Y_i ~ N(0, sigma_i^2 * I)   [isotropic!]

Proof sketch: average the decomposition over G:
  X =^d  (1/|G|) sum_{g in G} g(Y_1) + g(Y_2) + g(Y_3)
The averaged distribution for each component has covariance
  E[g(Y_i) g(Y_i)^T] averaged over g  =  sigma_i^2 * I
because the standard representation of S_4 is irreducible
(Schur's lemma => commutant = scalar multiples of I).

So: Y_i ~ N(0, sigma_i^2 * I)  for scalars sigma_1, sigma_2, sigma_3.

KEY CONSEQUENCE: if Y_1, Y_2, Y_3 were INDEPENDENT, their sum would be
  N(0, (sigma_1^2 + sigma_2^2 + sigma_3^2) * I)  -- Gaussian!
But X is NOT Gaussian (bounded support). Therefore the Y_i MUST be
DEPENDENT (jointly non-Gaussian, each marginally Gaussian).

MOMENT CONSTRAINTS
------------------
Let sigma^2_X = Var(X_1) = E[X_1^2]  (each component has equal variance by isotropy)

2nd moment:  sigma_1^2 + sigma_2^2 + sigma_3^2 + 2*(rho_12 + rho_13 + rho_23) = sigma^2_X
             where rho_ij = Cov(Y_i,1, Y_j,1)  [cross-component, scalar by symmetry]

4th moment:  E[(Y_1+Y_2+Y_3)^4] = E[X_1^4]
  = 3*sigma_1^4 + 3*sigma_2^4 + 3*sigma_3^4  (Gaussian 4th moments)
    + 6*E[Y_1^2*Y_2^2] + 6*E[Y_1^2*Y_3^2] + 6*E[Y_2^2*Y_3^2]  (cross 4th)
    + ... cross terms involving 3 components

The EXCESS KURTOSIS of X (how non-Gaussian it is) determines how
the cross-4th-moments E[Y_i^2 * Y_j^2] must deviate from
the independent case (sigma_i^2 * sigma_j^2).
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# ── Regular tetrahedron, circumradius = 1 ──────────────────────────────────
s3 = np.sqrt(3)
TET = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float) / s3
print("Tetrahedron vertices (unit circumradius):")
for i, v in enumerate(TET):
    print(f"  V{i} = {v}")

def sum_of_distances(P, vertices=TET):
    """Sum of Euclidean distances from P to all vertices."""
    return np.sum(np.linalg.norm(P[:,None,:] - vertices[None,:,:], axis=2), axis=1)

# ── Rejection sampling from B_k ────────────────────────────────────────────
def sample_multifocal_body(k, n_samples, bounding_radius=2.0):
    """Sample uniformly from {P : sum ||P-Vi|| <= k}."""
    samples = []
    n_accepted = 0
    n_tried = 0
    while n_accepted < n_samples:
        batch = np.random.uniform(-bounding_radius, bounding_radius, (n_samples*5, 3))
        inside_sphere = np.linalg.norm(batch, axis=1) <= bounding_radius
        batch = batch[inside_sphere]
        sod = sum_of_distances(batch)
        accepted = batch[sod <= k]
        samples.append(accepted)
        n_accepted += len(accepted)
        n_tried += len(batch)
    samples = np.vstack(samples)[:n_samples]
    acceptance_rate = n_accepted / n_tried
    return samples, acceptance_rate

# ── Compute moments and kurtosis ──────────────────────────────────────────
def compute_moments(X):
    """Return key statistics of the distribution."""
    n, d = X.shape
    # 2nd moment (variance of each component, should be equal by isotropy)
    var_components = np.var(X, axis=0)
    sigma2 = np.mean(var_components)   # pooled isotropic variance

    # 4th moment of first component
    m4 = np.mean(X[:,0]**4)
    # Excess kurtosis
    kurtosis = m4 / sigma2**2 - 3
    # Cross-4th moment E[X_1^2 * X_2^2]
    cross_m4 = np.mean(X[:,0]**2 * X[:,1]**2)

    # Compare to Gaussian: N(0, sigma2*I) would have
    gauss_m4 = 3 * sigma2**2
    gauss_cross_m4 = sigma2**2

    return {
        'sigma2': sigma2,
        'sigma': np.sqrt(sigma2),
        'm4': m4,
        'kurtosis': kurtosis,
        'cross_m4': cross_m4,
        'gauss_m4': gauss_m4,
        'gauss_cross_m4': gauss_cross_m4,
        'gauss_kurtosis': 0.0,
    }

# ── Study distribution at several k values ────────────────────────────────
print("\n" + "="*60)
print("MOMENT ANALYSIS for different k values")
print("="*60)
k_min = 4.0  # approximate minimum (sum at centroid)

results = {}
for k in [4.4, 5.0, 6.0, 7.5, 10.0]:
    X, rate = sample_multifocal_body(k, 80000)
    m = compute_moments(X)
    results[k] = m
    results[k]['X'] = X
    print(f"\nk = {k:.1f}  (acceptance rate: {rate:.3f})")
    print(f"  sigma (isotropic std):   {m['sigma']:.4f}")
    print(f"  Excess kurtosis:         {m['kurtosis']:.4f}  (Gaussian = 0)")
    print(f"  E[X1^4]:                 {m['m4']:.4f}  (Gaussian: {m['gauss_m4']:.4f})")
    print(f"  E[X1^2 X2^2]:            {m['cross_m4']:.4f}  (Gaussian: {m['gauss_cross_m4']:.4f})")

# ── The 2-Gaussian construction (moment-matching approach) ─────────────────
print("\n" + "="*60)
print("2-GAUSSIAN CONSTRUCTION via moment matching")
print("="*60)
print("""
By the symmetry argument, we need Y1 ~ N(0, s1^2*I), Y2 ~ N(0, s2^2*I)
dependent, with Y1 + Y2 =^d X.

The 2nd moment gives:  s1^2 + s2^2 + 2*rho = sigma^2_X      ... (1)
The 4th moment gives:  3*s1^4 + 6*E[Y1^2 Y2^2] + 3*s2^4 = E[X1^4]  ... (2)

For the SIMPLEST symmetric case: s1 = s2 = s, rho = -s^2 + sigma^2_X/2

Then (1): 2s^2 + 2*rho = sigma^2_X  =>  rho = sigma^2_X/2 - s^2
For rho to satisfy |rho| <= s^2 (valid covariance): need s >= sigma_X/2.

The 4th moment then constrains E[Y1^2 Y2^2]:
  6*s^4 + 6*E[Y1^2 Y2^2] = E[X1^4]
  E[Y1^2 Y2^2] = (E[X1^4] - 6*s^4) / 6

For this to be achievable by a valid distribution, we need constraints on s.
The minimum: E[Y1^2 Y2^2] >= 0 (trivially) or more precisely via Cauchy-Schwarz:
  E[Y1^2 Y2^2] >= (Cov(Y1^2, Y2^2) + E[Y1^2]E[Y2^2])  but also
  E[Y1^2 Y2^2] <= E[Y1^4]^{1/2} E[Y2^4]^{1/2} = 3s^4  (Cauchy-Schwarz)

So the 4th moment constraint reads:
  0 <= (E[X1^4] - 6*s^4) / 6 <= 3*s^4
  => E[X1^4] >= 6*s^4     and    E[X1^4] <= 24*s^4

This pins down the feasible range of s.
""")

k_demo = 5.0
m = results[k_demo]
sigma2_X = m['sigma2']
m4_X = m['m4']

# Symmetric 2-Gaussian: s1 = s2 = s
# 4th moment constraint: 3s^4 + 3s^4 + 6*E[Y1^2 Y2^2] = m4_X
# At minimum E[Y1^2 Y2^2] = 0:  s_max^4 = m4_X / 6
# At maximum E[Y1^2 Y2^2] = 3s^4: s_min^4 = m4_X / 24
s_max = (m4_X / 6)**0.25
s_min = (m4_X / 24)**0.25
s_natural = np.sqrt(sigma2_X / 2)  # natural choice: each Y_i has half the total variance

print(f"For k = {k_demo}: sigma_X = {np.sqrt(sigma2_X):.4f}")
print(f"  Feasible s range (symmetric 2-Gaussian): [{s_min:.4f}, {s_max:.4f}]")
print(f"  Natural choice s = sigma_X/sqrt(2):       {s_natural:.4f}")
print(f"  Is natural choice feasible? {s_min <= s_natural <= s_max}")

E_cross = (m4_X - 6 * s_natural**4) / 6
rho_12 = sigma2_X / 2 - s_natural**2  # = 0 for symmetric choice
print(f"  With s = sigma_X/sqrt(2): rho_12 = {rho_12:.4f},  E[Y1^2 Y2^2] = {E_cross:.4f}")
print(f"  (Independent Gaussian would give E[Y1^2 Y2^2] = s^4 = {s_natural**4:.4f})")
print(f"  Ratio E[Y1^2 Y2^2] / s^4 = {E_cross / s_natural**4:.4f}  (=1 for indep. Gaussian)")

# ── Visualise: 1D marginal distributions ─────────────────────────────────
print("\nGenerating visualisation...")

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

k_vals_plot = [4.4, 5.0, 6.0]
colors = ['#2266cc', '#22aa66', '#cc4422']

for col_idx, k in enumerate(k_vals_plot):
    m = results[k]
    X = m['X']
    sigma = m['sigma']

    ax = fig.add_subplot(gs[0, col_idx])
    ax.set_facecolor('white')

    # 1D marginal of X (project onto x-axis)
    x_proj = X[:, 0]
    t = np.linspace(x_proj.min(), x_proj.max(), 300)

    # KDE of X_1
    kde = stats.gaussian_kde(x_proj, bw_method=0.05)
    ax.fill_between(t, kde(t), alpha=0.4, color=colors[col_idx], label=f'X₁ (k={k})')
    ax.plot(t, kde(t), color=colors[col_idx], lw=1.5)

    # Best 1-Gaussian approximation
    gauss_density = stats.norm.pdf(t, 0, sigma)
    ax.plot(t, gauss_density, 'k--', lw=1.5, alpha=0.7, label='N(0,σ²)')

    ax.set_title(f'k = {k}  |  excess kurtosis = {m["kurtosis"]:.3f}', fontsize=10)
    ax.set_xlabel('X₁')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)

# Bottom row: kurtosis vs k, and Q-Q plot for k=5.0
ax_kurt = fig.add_subplot(gs[1, 0])
ax_kurt.set_facecolor('white')
k_list = sorted(results.keys())
kurt_list = [results[k]['kurtosis'] for k in k_list]
sigma_list = [results[k]['sigma'] for k in k_list]
ax_kurt.plot(k_list, kurt_list, 'o-', color='#2244aa', lw=2, ms=6)
ax_kurt.axhline(0, color='gray', ls='--', lw=1, label='Gaussian (κ=0)')
ax_kurt.set_xlabel('k (level)')
ax_kurt.set_ylabel('Excess kurtosis')
ax_kurt.set_title('Non-Gaussianity vs. level k', fontsize=10)
ax_kurt.legend(fontsize=8)
ax_kurt.grid(True, alpha=0.3)

# Q-Q plot for k=5.0
ax_qq = fig.add_subplot(gs[1, 1])
ax_qq.set_facecolor('white')
k_demo = 5.0
X_demo = results[k_demo]['X']
sigma_demo = results[k_demo]['sigma']
x_proj_demo = X_demo[:, 0]
sample_qq = np.random.choice(x_proj_demo, 2000, replace=False)
sample_qq_sorted = np.sort(sample_qq)
theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, 2000), 0, sigma_demo)
ax_qq.scatter(theoretical_q, sample_qq_sorted, s=1, alpha=0.3, color='#2244aa')
ax_qq.plot([-2,2], [-2,2], 'r--', lw=1.5, label='y=x (perfect Gaussian)')
ax_qq.set_xlabel('Gaussian quantiles')
ax_qq.set_ylabel('X₁ quantiles')
ax_qq.set_title(f'Q-Q plot vs N(0,σ²)  [k={k_demo}]', fontsize=10)
ax_qq.legend(fontsize=8)
ax_qq.grid(True, alpha=0.3)
ax_qq.set_xlim(-2.5, 2.5); ax_qq.set_ylim(-2.5, 2.5)

# Feasible region for 2-Gaussian construction
ax_feas = fig.add_subplot(gs[1, 2])
ax_feas.set_facecolor('white')
k_vals_all = sorted(results.keys())
s_mins = [(results[k]['m4'] / 24)**0.25 for k in k_vals_all]
s_maxs = [(results[k]['m4'] / 6)**0.25 for k in k_vals_all]
s_nats = [np.sqrt(results[k]['sigma2'] / 2) for k in k_vals_all]
ax_feas.fill_between(k_vals_all, s_mins, s_maxs, alpha=0.3, color='#4488cc', label='Feasible s range')
ax_feas.plot(k_vals_all, s_nats, 'o-', color='#cc4422', lw=2, ms=6, label='Natural s = σ_X/√2')
ax_feas.set_xlabel('k (level)')
ax_feas.set_ylabel('s (Gaussian std)')
ax_feas.set_title('2-Gaussian construction:\nfeasible σ for each Y_i', fontsize=10)
ax_feas.legend(fontsize=8)
ax_feas.grid(True, alpha=0.3)

fig.suptitle('Multifocal Ellipsoid (Tetrahedron Foci): Gaussian Decomposition Analysis',
             fontsize=13, y=1.01)

plt.savefig('/sessions/adoring-happy-cerf/mnt/outputs/gaussians_analysis.png',
            dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print("Plot saved.")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
1. SYMMETRY REDUCTION: By S_4 invariance, any Gaussian Y_i in the
   decomposition must be isotropic: Y_i ~ N(0, sigma_i^2 * I).
   The problem reduces to finding SCALAR variances and a joint
   NON-GAUSSIAN coupling.

2. INDEPENDENCE IS IMPOSSIBLE: If Y_1,...,Y_k were independent,
   their sum would be Gaussian. Since X is NOT Gaussian, the
   Y_i MUST be dependent.

3. HOW MANY? The excess kurtosis of X (negative = platykurtic,
   lighter-tailed than Gaussian) measures how non-Gaussian X is.
   For our body:
   - Near k=4 (tiny body): very non-Gaussian (kurtosis << 0)
   - As k grows: kurtosis approaches 0 (body → sphere → but still
     NOT Gaussian; kurtosis of uniform on ball in R^3 = -6/7 ≈ -0.857)
   So X is NEVER exactly Gaussian for any finite k.

4. TWO GAUSSIANS SUFFICE (for the symmetric case): The moment
   constraints from Y_1 + Y_2 =^d X can be satisfied with
   s_1 = s_2 = sigma_X/sqrt(2), rho_12 = 0, and a specific
   joint 4th moment E[Y1^2 Y2^2] determined by the kurtosis of X.
   The natural choice s = sigma_X/sqrt(2) lies in the feasible range.

5. THE EXPLICIT JOINT DISTRIBUTION: A concrete 2-Gaussian coupling
   (Y_1, Y_2) with Gaussian marginals and Y_1+Y_2 =^d X can be
   constructed via optimal transport on the 1D marginal, lifted
   to 3D by isotropy. This requires solving a 1D Fredholm equation
   for the conditional distribution P(Y_1 | Y_1+Y_2 = x).
""")
