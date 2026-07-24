"""Regenerate the consolidated results table in README.md.

Every number is read from a committed artifact -- results_5seed.csv for the
equivariant arms, results/baselines.json for the references without geometric
inductive bias -- so the table cannot drift away from the results it reports.

    python make_results_table.py            # print it
    python make_results_table.py --write    # splice it into README.md
"""
import argparse, csv, json
import numpy as np

START = '<!-- results-table:start -->'
END = '<!-- results-table:end -->'


def arm_stats(rows, prop, arm):
    v = np.array([float(r['metric_value']) for r in rows
                  if r['property'] == prop and r['conditioning'] == arm])
    return (v.mean(), v.std(ddof=1)) if len(v) else (None, None)


def fmt(mean, sd, digits):
    if mean is None:
        return '--'
    if sd is None:
        return ('%%.%df' % digits) % mean
    return ('%%.%df ± %%.%df' % (digits, digits)) % (mean, sd)


def build():
    rows = list(csv.DictReader(open('results_5seed.csv')))
    b = json.load(open('results/baselines.json'))
    f = np.array([x['compMAE'] for x in b['fcnn_dipole']])
    fr = np.array([x['compMAE_rotated'] for x in b['fcnn_dipole']])

    lines = [
        '| model | geometric inductive bias | dipole, compMAE (D) | polarizability, Frobenius (a.u.) |',
        '| --- | --- | --- | --- |',
    ]
    for arm, label in (('none', 'PaiNN baseline'),
                       ('tda', 'PaiNN + TDA conditioning'),
                       ('random', 'PaiNN + matched random features')):
        d = arm_stats(rows, 'dipole', arm)
        p = arm_stats(rows, 'polar', arm)
        lines.append('| %s | E(3)-equivariant | %s | %s |'
                     % (label, fmt(*d, digits=4), fmt(*p, digits=3)))
    lines.append('| FCNN on raw coordinates | none | %s | -- |'
                 % fmt(f.mean(), f.std(ddof=1), 4))
    lines.append('| the same FCNN, test molecules rotated | none | %s | -- |'
                 % fmt(fr.mean(), fr.std(ddof=1), 4))
    lines.append('| predicting the training mean | none | %s | -- |'
                 % fmt(b['mean_predictor_dipole_compMAE'], None, 4))
    lines += [
        '',
        'Lower is better; ± is the sample standard deviation over the five training seeds.',
        'The equivariant arms differ only in what the conditioning path is fed. Cells marked',
        '`--` are not defined for that model: the non-equivariant references were run on the',
        'dipole task only.',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    table = build()
    if not a.write:
        print(table)
        return
    s = open('README.md', encoding='utf-8').read()
    i, j = s.index(START), s.index(END)
    s = s[:i + len(START)] + '\n' + table + '\n' + s[j:]
    open('README.md', 'w', encoding='utf-8').write(s)
    print('README.md results table regenerated')


if __name__ == '__main__':
    main()
