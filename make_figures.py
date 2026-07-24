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


def fig_baselines():
    """Where the equivariant family sits once models without that bias share the axis."""
    import csv
    rows = list(csv.DictReader(open('results_5seed.csv')))
    b = json.load(open('results/baselines.json'))
    eq = np.array([float(r['metric_value']) for r in rows
                   if r['property'] == 'dipole' and r['conditioning'] == 'none'])
    fc = np.array([x['compMAE'] for x in b['fcnn_dipole']])
    fr = np.array([x['compMAE_rotated'] for x in b['fcnn_dipole']])
    mp = b['mean_predictor_dipole_compMAE']
    labels = ['PaiNN (equivariant)', 'FCNN on raw coordinates',
              'FCNN, rotated test molecules', 'predicting the training mean']
    vals = [eq.mean(), fc.mean(), fr.mean(), mp]
    errs = [eq.std(ddof=1), fc.std(ddof=1), fr.std(ddof=1), 0.0]
    cols = [BLUE, ORANGE, ORANGE, GREY]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4),
                             gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, xerr=errs, color=cols, height=.62, capsize=0,
            error_kw=dict(lw=1.6, ecolor='#2b2b2b'))
    for yy, v, e in zip(y, vals, errs):
        lab = '%.3f' % v if e == 0 else '%.3f ± %.3f' % (v, e)
        ax.text(v + e + .05, yy, lab, va='center', fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('dipole component-wise MAE (D), lower is better')
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_title('the same task, with and without the geometric bias', loc='left')

    ax = axes[1]
    for name, pair, c in (('FCNN', (fc.mean(), fr.mean()), ORANGE),
                          ('PaiNN', (eq.mean(), eq.mean()), BLUE)):
        ax.plot([0, 1], pair, '-o', color=c, lw=2.4, ms=7, label=name)
        ax.annotate('%+.0f%%' % (100 * (pair[1] / pair[0] - 1)), xy=(1.05, pair[1]),
                    color=c, fontsize=10, va='center', fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['as given', 'rigidly rotated'])
    ax.set_xlim(-.2, 1.45)
    ax.set_ylim(0, fr.mean() * 1.28)
    ax.set_ylabel('dipole compMAE (D)')
    ax.set_title('rotating the test set', loc='left')
    ax.legend(loc='upper left', frameon=True)

    fig.suptitle('Dropping the geometric inductive bias costs a factor of seven, and leaves '
                 'the model sensitive to orientation',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.text(.5, .012, 'Rotation test: molecule and reference dipole rotate together, so the task is unchanged; PaiNN is invariant here by construction, verified to float32 round-off',
             ha='center', fontsize=8.4, color='#555')
    fig.tight_layout(rect=[0, .055, 1, .92])
    fig.savefig('assets/fig_baselines.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_baselines.png')

def _probe_stats(path, arm, prop, col):
    import csv
    rows = [r for r in csv.DictReader(open(path))
            if r['property'] == prop and r['basis'] == 'primary' and r['arm'] == arm]
    per = {}
    for r in rows:
        v = r[col]
        if v not in ('', None):
            per.setdefault(int(r['seed']), []).append(float(v))
    vals = np.array([np.mean(per[s]) for s in sorted(per)])
    return vals.mean(), vals.std(ddof=1)


def fig_residual_probes():
    """Both probes, both readouts: no positive R2 and no useful correction."""
    paths = {'Ridge': 'results/residual_probe_per_seed.csv',
             'MLP': 'results/residual_probe_mlp_per_seed.csv'}
    arms = ['tda', 'random', 'shuffled']
    probes = [('Ridge', BLUE), ('MLP', ORANGE)]
    props = [('dipole', 'compMAE (D)'), ('polar', 'Frobenius (a.u.)')]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    x = np.arange(len(arms))
    wid = .34
    for row, (prop, unit) in enumerate(props):
        for col, (metric, title) in enumerate(
                [('probe_r2_mean', 'probe R2 on held-out coefficients'),
                 ('delta', 'change in %s vs the frozen baseline' % unit)]):
            ax = axes[row][col]
            for k, (name, colr) in enumerate(probes):
                m = [_probe_stats(paths[name], a, prop, metric) for a in arms]
                ax.bar(x + (k - .5) * wid, [v[0] for v in m], wid,
                       yerr=[v[1] for v in m], color=colr, label=name,
                       error_kw=dict(lw=1.3, ecolor='#2b2b2b'), capsize=0)
            ax.axhline(0, color='#333', lw=1.3, ls='--', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(arms)
            ax.set_title('%s - %s' % (prop, title), loc='left', fontsize=10)
            if col == 0:
                ax.set_ylabel('R2 (test-mean reference)')
            else:
                ax.set_ylabel('positive = worse than baseline')
            if row == 0 and col == 0:
                ax.legend(frameon=True, fontsize=9)

    fig.suptitle('Neither probe reaches a positive R2, and neither correction lowers the error; '
                 'TDA does not separate from the controls',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.text(.5, .012, 'Bars are means over the 5 baseline seeds, whiskers the sample standard '
                       'deviation; realizations and initializations are averaged within a seed. '
                       'Primary basis only.',
             ha='center', fontsize=8.4, color='#555')
    fig.tight_layout(rect=[0, .04, 1, .93])
    fig.savefig('assets/fig_residual_probes.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_residual_probes.png')

def _box(ax, cx, cy, w, h, text, fc, ec, fs=9.2, weight='normal'):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.05,rounding_size=0.14',
                                linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight=weight, zorder=3)


def _arrow(ax, xy_from, xy_to, color, style='-|>', lw=1.9, ls='-'):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, color=color,
                                 linewidth=lw, linestyle=ls, mutation_scale=15,
                                 shrinkA=2, shrinkB=2, zorder=1))


def fig_method():
    """How the two paths meet: conditioning acts after the backbone, never across irreps."""
    NL = chr(10)
    MU, AL, OPLUS = chr(956), chr(945), chr(8853)
    SUB0, SUB1 = chr(8320), chr(8321)
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.6)
    ax.axis('off')
    PALE_B, PALE_O, PALE_G = '#e8eef6', '#fbeee0', '#eef0f2'
    ytop, yfilm, ybot = 4.45, 2.55, 1.05

    _box(ax, 1.3, ytop, 2.2, 1.05,
         NL.join(['molecule', 'atoms + 3D coordinates']), PALE_G, '#6b7280')
    _box(ax, 4.1, ytop, 2.3, 1.05,
         NL.join(['PaiNN', 'equivariant message passing']), PALE_B, BLUE)
    _box(ax, 7.0, ytop, 2.4, 1.25,
         NL.join(['representations', 's : l=0  (invariant)', 'v : l=1  (equivariant)']),
         PALE_B, BLUE)
    _box(ax, 10.0, ytop, 2.2, 1.25,
         NL.join(['response heads', MU + ' : l=1',
                  AL + ' : l=0 ' + OPLUS + ' l=2']), PALE_B, BLUE)

    _box(ax, 4.1, ybot, 2.6, 1.05,
         NL.join(['Vietoris' + chr(8211) + 'Rips persistence',
                  'all atoms, no element types']), PALE_O, ORANGE)
    _box(ax, 7.0, ybot, 2.6, 1.25,
         NL.join(['z_PH : 130-D, invariant',
                  'H' + SUB0 + ' / H' + SUB1 + ' Betti curves',
                  '+ 2 persistence entropies']), PALE_O, ORANGE)
    _box(ax, 7.0, yfilm, 3.5, 0.95,
         NL.join(['FiLM: scale + shift on s,  invariant gate on v',
                  'no mixing between l=0 and l=1']), PALE_O, ORANGE, fs=9.0)

    _arrow(ax, (2.4, ytop), (2.95, ytop), '#6b7280')
    _arrow(ax, (5.25, ytop), (5.8, ytop), BLUE)
    _arrow(ax, (8.2, ytop), (8.9, ytop), BLUE)
    _arrow(ax, (1.3, ytop - 0.53), (1.3, ybot), '#6b7280')
    _arrow(ax, (1.3, ybot), (2.8, ybot), '#6b7280')
    _arrow(ax, (5.4, ybot), (5.7, ybot), ORANGE)
    _arrow(ax, (7.0, ybot + 0.63), (7.0, yfilm - 0.48), ORANGE)
    _arrow(ax, (7.0, yfilm + 0.48), (7.0, ytop - 0.63), ORANGE)

    NL2 = chr(10)
    ax.text(7.28, 3.62, 'applied after message passing', fontsize=8.6,
            color=ORANGE, style='italic', ha='left', va='center')
    fig.text(0.012, 0.055, NL2.join([
        'The conditioning path is invariant end to end, so exact E(3) equivariance survives: scalars are rescaled',
        'and shifted, vectors are scaled by an invariant gate, and nothing is ever mixed across irreducible',
        'representations. The last FiLM layer is zero-initialized, so at initialization the conditioned model',
        'reproduces the baseline exactly.']),
        fontsize=8.6, color='#444', ha='left', va='bottom')
    ax.text(11.25, 0.30, NL2.join(['topology-OOD split:', 'train ' + chr(8804) + ' 1 ring',
                                   'test ' + chr(8805) + ' 2 rings']),
            fontsize=8.6, color='#444', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f5f6f8', ec='#c9ced6'))
    fig.suptitle('Two paths, joined after the backbone: geometry stays equivariant, topology '
                 'enters only as invariant gain',
                 fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.20, 1, 0.93])
    fig.savefig('assets/fig_method.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote assets/fig_method.png')

if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)
    fig_method()
    fig_paired_seeds()
    fig_zph_structure()
    fig_betti_curves()
    fig_baselines()
    fig_residual_probes()
