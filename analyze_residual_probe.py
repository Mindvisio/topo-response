"""Aggregate the residual-probe per-seed table into the paired n=5 statistics.

The unit of independent replication is the baseline training seed (n=5). random and
shuffled contribute five realizations per seed; those are averaged WITHIN a seed
first, so they never masquerade as extra training seeds. Paired differences are
TDA vs each control, with a Holm correction over the pre-declared family of primary
tests. p>0.05 is reported as "not detected", never as equivalence.
"""
import argparse, csv, json
import numpy as np

PRIMARY_METRIC = {'dipole': 'compMAE', 'polar': 'Frob'}


def _safe_mean(vals):
    """Mean over the non-NaN entries; NaN if there are none (the null arm fits
    no probe, so it has no R^2 -- nanmean would warn on an empty slice)."""
    v = [x for x in vals if x == x]
    return float(np.mean(v)) if v else float('nan')


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ('seed', 'realization'):
            r[k] = int(r[k])
        for k in ('baseline', 'corrected', 'delta'):
            r[k] = float(r[k])
        # optional columns: the MLP probe records neither a sensitivity refit nor
        # the basis conditioning, so treat them as absent rather than required
        for k in ('frac_small', 'sens_delta'):
            v = r.get(k, '')
            r[k] = float(v) if v not in ('', None) else np.nan
        r['probe_r2_mean'] = float(r['probe_r2_mean']) if r['probe_r2_mean'] else np.nan
    return rows


EXPECTED_SEEDS = 5
# Rows per seed per arm.  The Ridge probe fits one model per descriptor; the MLP
# probe multiplies each descriptor by its weight initialisations, so the counts
# are overridable from the command line rather than hard-coded to one design.
EXPECTED_REALIZATIONS = {'null': 1, 'tda': 1, 'random': 5, 'shuffled': 5}


def seed_means(rows, prop, basis, arm):
    """Delta and probe-R2 per seed, averaging realizations WITHIN each seed.

    Averaging first is what keeps the 5 random/shuffled realizations from being
    passed off as extra training seeds.  The design is checked rather than
    assumed: a missing seed or realization is a silent power inflation.
    """
    out = {}
    for r in rows:
        if r['property'] == prop and r['basis'] == basis and r['arm'] == arm:
            out.setdefault(r['seed'], []).append(r)
    seeds = sorted(out)
    if not seeds:
        return [], np.array([]), np.array([])
    if len(seeds) != EXPECTED_SEEDS:
        raise SystemExit('incomplete design: %s/%s/%s has %d seeds, expected %d'
                         % (prop, basis, arm, len(seeds), EXPECTED_SEEDS))
    want = EXPECTED_REALIZATIONS.get(arm, 1)
    for s in seeds:
        if len(out[s]) != want:
            raise SystemExit('incomplete design: %s/%s/%s seed %d has %d realizations, expected %d'
                             % (prop, basis, arm, s, len(out[s]), want))
    delta = np.array([np.mean([x['delta'] for x in out[s]]) for s in seeds])
    r2 = np.array([_safe_mean([x['probe_r2_mean'] for x in out[s]]) for s in seeds])
    return seeds, delta, r2


def paired(a, b):
    """Paired t on seed-aligned vectors: mean diff, 95% CI, p."""
    from scipy import stats
    d = a - b
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    p = float(stats.ttest_rel(a, b).pvalue)
    return m, m - tc * se, m + tc * se, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving."""
    idx = np.argsort(pvals); m = len(pvals)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


def main():
    global EXPECTED_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/residual_probe_per_seed.csv')
    ap.add_argument('--out', default='results/residual_probe_summary.json')
    ap.add_argument('--seeds', type=int, default=EXPECTED_SEEDS,
                    help='required number of baseline seeds')
    ap.add_argument('--rows-per-seed', nargs='*', default=[],
                    help='override rows expected per seed, e.g. tda=3 random=15 shuffled=15')
    ap.add_argument('--label', default='Ridge probe',
                    help='name of this pre-declared family, used in the printout')
    a = ap.parse_args()
    EXPECTED_SEEDS = a.seeds
    for item in a.rows_per_seed:
        arm, _, cnt = item.partition('=')
        EXPECTED_REALIZATIONS[arm] = int(cnt)
    rows = load(a.csv)
    summary = {}
    family = []                                          # (label, pvalue) for the Holm family
    for prop in ['dipole', 'polar']:
        pm = PRIMARY_METRIC[prop]
        summary[prop] = {'primary_metric': pm}
        for basis in ['primary', 'secondary']:
            seeds, d_tda, r2_tda = seed_means(rows, prop, basis, 'tda')
            if not seeds:
                continue
            _, d_null, _ = seed_means(rows, prop, basis, 'null')
            _, d_rand, r2_rand = seed_means(rows, prop, basis, 'random')
            _, d_shuf, r2_shuf = seed_means(rows, prop, basis, 'shuffled')
            block = {
                'seeds': seeds,
                'delta_tda_per_seed': d_tda.tolist(),
                'delta_tda_mean': float(d_tda.mean()), 'delta_tda_std': float(d_tda.std(ddof=1)),
                'probe_r2_tda_per_seed': r2_tda.tolist(), 'probe_r2_tda_mean': float(np.nanmean(r2_tda)),
                'probe_r2_random_mean': float(np.nanmean(r2_rand)),
                'probe_r2_shuffled_mean': float(np.nanmean(r2_shuf)),
            }
            for ctrl_name, ctrl in [('null', d_null), ('random', d_rand), ('shuffled', d_shuf)]:
                m, lo, hi, p = paired(d_tda, ctrl)
                block['tda_vs_%s' % ctrl_name] = dict(mean=m, ci=[lo, hi], p=p)
                if basis == 'primary':                   # only the primary basis enters the declared family
                    family.append(('%s tda-vs-%s' % (prop, ctrl_name), p))
            sens = [r['sens_delta'] for r in rows
                    if r['property'] == prop and r['basis'] == basis and r['arm'] == 'tda'
                    and not np.isnan(r['sens_delta'])]
            if sens:
                block['sensitivity_delta_mean'] = float(np.mean(sens))
            fs = [r['frac_small'] for r in rows
                  if r['property'] == prop and r['basis'] == basis and r['frac_small'] == r['frac_small']]
            if fs:
                block['frac_small'] = float(np.mean(fs))
            summary[prop][basis] = block
    labels = [x[0] for x in family]; ps = np.array([x[1] for x in family])
    adj = holm(ps)
    summary['holm_family'] = {lab: {'p': float(p), 'p_holm': float(pa)}
                              for lab, p, pa in zip(labels, ps, adj)}
    summary['family_label'] = a.label
    summary['holm_note'] = ('family = the %d primary-basis tda-vs-control tests; '
                            'Holm threshold for the smallest is 0.05/%d = %.4f' %
                            (len(family), len(family), 0.05 / max(len(family), 1)))
    json.dump(summary, open(a.out, 'w'), indent=1)

    print('=== %s: residual probe summary (n=%d seeds, topology-OOD) ===' % (a.label, a.seeds))
    for prop in ['dipole', 'polar']:
        pm = summary[prop]['primary_metric']
        print('\n%s  (primary metric %s; positive delta = correction HURT)' % (prop.upper(), pm))
        for basis in ['primary', 'secondary']:
            b = summary[prop].get(basis)
            if not b:
                continue
            if 'frac_small' in b:
                print('  [%s basis]  frac |mu|<0.1D or near-iso = %.4f' % (basis, b['frac_small']))
            else:
                print('  [%s basis]' % basis)
            print('    probe R2: tda %.3f | random %.3f | shuffled %.3f'
                  % (b['probe_r2_tda_mean'], b['probe_r2_random_mean'], b['probe_r2_shuffled_mean']))
            print('    delta_tda per seed: %s' % ['%+.4f' % x for x in b['delta_tda_per_seed']])
            print('    delta_tda = %+.4f +/- %.4f' % (b['delta_tda_mean'], b['delta_tda_std']))
            for c in ['null', 'random', 'shuffled']:
                t = b['tda_vs_%s' % c]
                print('    tda vs %-8s: %+.4f [%+.4f, %+.4f] p=%.3f'
                      % (c, t['mean'], t['ci'][0], t['ci'][1], t['p']))
            if 'sensitivity_delta_mean' in b:
                print('    sensitivity (val-only refit) mean delta = %+.4f' % b['sensitivity_delta_mean'])
    print('\nHolm family (primary basis):')
    for lab, v in summary['holm_family'].items():
        print('  %-22s p=%.3f  p_holm=%.3f' % (lab, v['p'], v['p_holm']))
    print(summary['holm_note'])
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
