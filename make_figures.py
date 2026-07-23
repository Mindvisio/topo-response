"""Figures for the README.

Three panels that together tell the study's story honestly:
  1. the headline negative, shown per seed rather than as a table row;
  2. whether z_PH encodes molecular topology at all;
  3. what the descriptor literally looks like, read straight out of cache/zph.npy.

Run: python make_figures.py   (CPU, ~1 min)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.titleweight': 'bold', 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
# blue/orange primary pair: safe for red-green colour blindness
BLUE, ORANGE, GREY = '#3b6ea5', '#d1791f', '#8a8f98'
SEEDS = [0, 1, 2, 3, 4]
BINS = 64
GRID = np.linspace(0.0, 1.0, BINS)


def load_5seed():
    import csv
    rows = list(csv.DictReader(open('results_5seed.csv')))
    out = {}
    for r in rows:
        out[(r['property'], r['conditioning'], int(r['seed']))] = float(r['metric_value'])
    return out


def paired_ci(a, b):
    from scipy import stats
    d = a - b
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return m, t * se, float(stats.ttest_rel(a, b).pvalue)


def fig_paired_seeds():
    """Per-seed values with the pairing drawn in, next to the paired differences."""
    d = load_5seed()
    props = [('dipole', 'component-wise MAE (D)'), ('polar', 'Frobenius error (a.u.)')]
    arms = ['none', 'tda', 'random']
    names = {'none': 'baseline', 'tda': 'TDA', 'random': 'random\n(matched)'}
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0),
                             gridspec_kw={'width_ratios': [1.0, 1.25]})
    for row, (prop, ylab) in enumerate(props):
        ax = axes[row][0]
        vals = {a: np.array([d[(prop, a, s)] for s in SEEDS]) for a in arms}
        x = np.arange(len(arms))
        for i, s in enumerate(SEEDS):                      # one line per seed: the pairing
            y = [vals[a][i] for a in arms]
            ax.plot(x, y, '-o', color=GREY, alpha=.75, lw=1.1, ms=4.5, zorder=2)
        for j, a in enumerate(arms):                       # arm means on top
            ax.plot([x[j] - .16, x[j] + .16], [vals[a].mean()] * 2,
                    color=BLUE, lw=2.6, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([names[a] for a in arms])
        ax.set_ylabel(ylab)
        ax.set_title('%s: every seed, every arm' % prop, loc='left')
        ax.margins(x=.18)

        ax = axes[row][1]
        comps = [('tda', 'none', 'TDA - baseline'), ('tda', 'random', 'TDA - random'),
                 ('random', 'none', 'random - baseline')]
        ys = np.arange(len(comps))[::-1]
        for (A, B, lab), yy in zip(comps, ys):
            m, h, p = paired_ci(vals[A], vals[B])
            hit = (m - h) * (m + h) > 0                    # does the interval clear zero?
            ax.errorbar(m, yy, xerr=h, fmt='o', color=ORANGE if hit else BLUE,
                        ms=6, lw=1.8, capsize=4, zorder=3)
            ax.text(m, yy + .22, 'p = %.2f' % p, ha='center', fontsize=8.5, color='#333')
        ax.axvline(0, color='#444', lw=1.1, ls='--', zorder=1)
        ax.set_yticks(ys); ax.set_yticklabels([c[2] for c in comps])
        ax.set_xlabel('paired difference, 95%% CI (%s)' % ('D' if prop == 'dipole' else 'a.u.'))
        ax.set_title('positive = the first arm is worse', loc='left')
        ax.set_ylim(-.6, len(comps) - .3)
    fig.suptitle('No seed-consistent effect: TDA conditioning separates from neither the '
                 'baseline nor a matched random control',
                 fontsize=12.5, fontweight='bold', y=.985)
    fig.text(.5, .008, 'SQuIRL topology-OOD split - 5 training seeds - pairing is by seed, '
                       'not by molecule', ha='center', fontsize=8.6, color='#555')
    fig.tight_layout(rect=[0, .022, 1, .955])
    fig.savefig('assets/fig_paired_seeds.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_paired_seeds.png')


def _rings_and_zph():
    idx = json.load(open('cache/index.json'))
    rings = np.array([m['n_rings'] for m in idx], dtype=int)
    z = np.load('cache/zph.npy').astype(np.float64)
    return rings, z


def fig_zph_structure():
    """Does the descriptor encode ring count at all? PCA plus a linear read-out."""
    rings, z = _rings_and_zph()
    # PCA on the mean-centered Betti curves.  Do NOT standardise per column: 35 of the
    # 130 entries have zero variance (grid bins identical for every molecule) and
    # dividing by their standard deviation turns numerical dust into the leading
    # component, collapsing the whole cloud into a line.
    X = z[:, :2 * BINS]
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2] * 100
    if np.corrcoef(pcs[:, 0], rings)[0, 1] < 0:          # orient PC1 so more rings = right
        pcs[:, 0] *= -1

    # honest read-out: fit on a random half, score on the other half
    sd = z.std(0)
    keep = sd > 1e-6                                     # drop the degenerate bins here too
    zs = (z[:, keep] - z[:, keep].mean(0)) / sd[keep]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(zs))
    tr, te = perm[:len(perm) // 2], perm[len(perm) // 2:]
    X = np.hstack([zs, np.ones((len(zs), 1))])
    W = np.linalg.solve(X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1]), X[tr].T @ rings[tr])
    pred = X[te] @ W
    r2 = 1 - ((rings[te] - pred) ** 2).sum() / ((rings[te] - rings[te].mean()) ** 2).sum()
    mae = np.abs(rings[te] - pred).mean()

    groups = [0, 1, 2, 3, 4]
    cmap = plt.get_cmap('viridis')
    cols = [cmap(i / (len(groups) - 1)) for i in range(len(groups))]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    ax = axes[0]
    sub = rng.permutation(len(zs))[:18000]
    for g, c in zip(groups, cols):
        sel = sub[(rings[sub] == g) if g < 4 else (rings[sub] >= 4)]
        ax.scatter(pcs[sel, 0], pcs[sel, 1], s=2.2, alpha=.30, color=c, linewidths=0,
                   label=('%d rings' % g) if g < 4 else '4+ rings')
    ax.set_xlabel('PC1 (%.0f%% of variance)' % var[0])
    ax.set_ylabel('PC2 (%.0f%%)' % var[1])
    ax.set_title('z_PH, coloured by ring count', loc='left')
    lg = ax.legend(loc='upper right', markerscale=5, framealpha=.9)
    for h in lg.legend_handles:
        h.set_alpha(1)

    ax = axes[1]
    data = [pcs[rings == g, 0] if g < 4 else pcs[rings >= 4, 0] for g in groups]
    bp = ax.boxplot(data, vert=True, widths=.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', lw=1.6))
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(.75); patch.set_edgecolor('#333')
    ax.set_xticklabels(['0', '1', '2', '3', '4+'])
    ax.set_xlabel('rings in the molecule')
    ax.set_ylabel('PC1')
    ax.set_title('PC1 tracks ring count monotonically', loc='left')
    ax.text(.02, .03, 'linear read-out of ring count from z_PH\n'
                      'held-out $R^2$ = %.2f, MAE = %.2f rings' % (r2, mae),
            transform=ax.transAxes, fontsize=8.8, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f2f4f7', ec='#c9ced6'))

    fig.suptitle('The descriptor does encode molecular topology - so the null result is about '
                 'how the model used it, not an empty input',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig('assets/fig_zph_structure.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_zph_structure.png  (ring-count R2=%.3f, MAE=%.3f)' % (r2, mae))


def fig_betti_curves():
    """What the descriptor literally is: the H0/H1 Betti curves stored in z_PH."""
    rings, z = _rings_and_zph()
    h0, h1 = z[:, :BINS], z[:, BINS:2 * BINS]
    groups = [0, 1, 2, 3, 4]
    cmap = plt.get_cmap('viridis')
    cols = [cmap(i / (len(groups) - 1)) for i in range(len(groups))]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True)
    for ax, curves, lab, note in (
            (axes[0], h0, '$H_0$ - connected components',
             'starts at the atom count, merges as the radius grows'),
            (axes[1], h1, '$H_1$ - independent loops',
             'the channel that carries ring information')):
        for g, c in zip(groups, cols):
            sel = (rings == g) if g < 4 else (rings >= 4)
            med = np.median(curves[sel], axis=0)
            lo, hi = np.percentile(curves[sel], [25, 75], axis=0)
            ax.fill_between(GRID, lo, hi, color=c, alpha=.16, linewidth=0)
            ax.plot(GRID, med, color=c, lw=2.0,
                    label=('%d rings' % g) if g < 4 else '4+ rings')
        ax.set_xlabel('filtration radius (molecule scaled to unit diameter)')
        ax.set_title(lab, loc='left')
        ax.text(.98, .93, note, transform=ax.transAxes, ha='right', fontsize=8.4,
                color='#555', style='italic')
    axes[0].set_ylabel('Betti number (median, IQR shaded)')
    axes[1].legend(loc='upper right', framealpha=.9)
    fig.suptitle('z_PH is these two curves, sampled on a fixed 64-point grid, plus two '
                 'persistence entropies',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig('assets/fig_betti_curves.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_betti_curves.png')


if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)
    fig_paired_seeds()
    fig_zph_structure()
    fig_betti_curves()
