"""Paired statistics for the 5-seed topology-OOD matrix.

By default this reads results_5seed.csv, which is committed, so the numbers in
README.md and RUN_MANIFEST.md can be reproduced from a clean clone with no
training logs present:

    python3 compute_ci.py

Pass --from-logs to rebuild results_5seed.csv from the raw training logs in
logs/ (those are git-excluded and only exist on the machine that trained).
"""
import argparse, csv, glob, re, sys
import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CSV = 'results_5seed.csv'
LOGS = 'logs'
PROPS = [('dipole', 'compMAE', 'dip'), ('polar', 'Frob', 'polar')]
CONDS = ['none', 'tda', 'random']
SEEDS = [0, 1, 2, 3, 4]


def from_logs():
    def grab(pattern, metric):
        fs = glob.glob(pattern)
        if not fs:
            return None
        hits = re.findall(metric + r' ([0-9.]+)', open(fs[0]).read())
        return float(hits[-1]) if hits else None
    rows = []
    for prop, metric, pref in PROPS:
        for cond in CONDS:
            for s in SEEDS:
                if s == 0:
                    pat = '%s/mx_%s_topology_ood_%s_s0.log' % (LOGS, pref, cond)
                elif s in (1, 2):
                    pat = '%s/s4b_%s_topology_ood_%s_s%d.log' % (LOGS, pref, cond, s)
                else:
                    pat = '%s/s4c_%s_topology_ood_%s_s%d.log' % (LOGS, pref, cond, s)
                rows.append([prop, cond, s, grab(pat, metric)])
    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['property', 'conditioning', 'seed', 'metric_value'])
        w.writerows(rows)
    print('rebuilt %s from %s/ (%d rows)' % (CSV, LOGS, len(rows)))


def load():
    """Read the CSV and refuse to report statistics on an incomplete matrix."""
    try:
        rows = list(csv.DictReader(open(CSV)))
    except FileNotFoundError:
        sys.exit('%s not found. Run with --from-logs on the training machine.' % CSV)
    res, problems = {}, []
    for prop, _, _ in PROPS:
        for cond in CONDS:
            got = {}
            for r in rows:
                if r['property'] == prop and r['conditioning'] == cond:
                    v = (r['metric_value'] or '').strip()
                    if v:
                        got[int(r['seed'])] = float(v)
            missing = [s for s in SEEDS if s not in got]
            if missing:
                problems.append('%s/%s missing seeds %s' % (prop, cond, missing))
            extra = sorted(set(got) - set(SEEDS))
            if extra:
                problems.append('%s/%s unexpected seeds %s' % (prop, cond, extra))
            res[(prop, cond)] = np.array([got.get(s, np.nan) for s in SEEDS], dtype=float)
    if problems:
        sys.exit('incomplete paired design, refusing to report:\n  ' + '\n  '.join(problems))
    return res


def paired(a, b):
    """Seed-aligned paired difference: a and b are indexed by SEEDS in order."""
    d = a - b
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if HAVE_SCIPY:
        tcrit = stats.t.ppf(0.975, n - 1)
        p = float(stats.ttest_rel(a, b).pvalue)
    else:
        tcrit, p = 2.776, float('nan')
    return mean, mean - tcrit * se, mean + tcrit * se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-logs', action='store_true', help='rebuild the CSV from logs/ first')
    a = ap.parse_args()
    if a.from_logs:
        from_logs()
    res = load()
    print('source: %s | scipy: %s | %d seeds per arm' % (CSV, HAVE_SCIPY, len(SEEDS)))
    print('%-7s %-16s %-9s %-22s %s' % ('property', 'comparison', 'mean', '95% CI', 'p'))
    for prop, _, _ in PROPS:
        for A, B, label in [('tda', 'none', 'tda-baseline'),
                            ('tda', 'random', 'tda-random'),
                            ('random', 'none', 'random-baseline')]:
            m, lo, hi, p = paired(res[(prop, A)], res[(prop, B)])
            print('%-7s %-16s %+.4f   [%+.4f, %+.4f]   p=%.3f %s'
                  % (prop, label, m, lo, hi, p, 'SIG' if (p == p and p < 0.05) else 'n.s.'))
    print('\nSix comparisons are reported; Holm over this family requires p < %.4f for the smallest.' % (0.05 / 6))


if __name__ == '__main__':
    main()
