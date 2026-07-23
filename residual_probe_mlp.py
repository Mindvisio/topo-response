"""Nonlinear residual probe: the same question as residual_probe.py, asked with a
small MLP instead of Ridge.

A linear probe finding nothing leaves one reading open -- that the relation exists
but is nonlinear.  This closes that reading as far as a small network can.

Specification, fixed before this probe was run (but after the Ridge results on the
same test set had been seen -- so this is a follow-up sensitivity analysis, not an
independent confirmatory experiment):
  * the well-conditioned PRIMARY basis only (the gyration-augmented bases are
    exploratory and ill-conditioned, so they are not used here);
  * MLP 130 -> 64 -> 32 -> k with ReLU, fitted on train, early-stopped on val;
  * identical architecture, optimizer, schedule and stopping rule for tda,
    random and shuffled -- the arms differ only in which descriptor they see;
  * 5 baseline seeds x 3 weight initialisations, and for the random/shuffled
    controls x5 descriptor realizations, all averaged WITHIN a baseline seed so
    the unit of replication stays the seed (n=5);
  * each fitted model evaluated once on test, with test results used for neither
    training, early stopping nor model selection; both readouts reported (probe
    R² against a test-mean reference, and the physical metric delta).

Analyzed as its own family of six tests rather than as an extension of the Ridge
family, so it does not retroactively change that family's multiplicity or revise
its conclusions.
"""
import argparse, csv, os
import numpy as np
import torch
import torch.nn as nn

from residual_probe import (ALPHAS, PRIMARY_METRIC, build_targets, corrected_metric,
                            dipole_metrics, polar_metrics, gyration_lookup, load_export,
                            load_zph, make_descriptor, probe_r2, standardize)

HIDDEN = (64, 32)
MAX_EPOCHS = 150
PATIENCE = 15
BATCH = 4096
LR = 3e-3
N_INITS = 3


def fit_mlp(Ztr, Ytr, Zva, Yva, Zte, init_seed):
    """Train on train, early-stop on val, return test predictions.

    The val split is used ONLY to decide when to stop -- never for gradient
    steps -- and the test split is touched once, at the end.
    """
    torch.manual_seed(init_seed)
    k = Ytr.shape[1]
    layers, d = [], Ztr.shape[1]
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, k)]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.MSELoss()

    Xtr = torch.tensor(Ztr, dtype=torch.float32); ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xva = torch.tensor(Zva, dtype=torch.float32); yva = torch.tensor(Yva, dtype=torch.float32)
    Xte = torch.tensor(Zte, dtype=torch.float32)

    n = len(Xtr)
    best_val, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(init_seed)
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xva), yva))
        if v < best_val - 1e-7:
            best_val, bad = v, 0
            best_state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Xte).numpy(), epoch + 1


def run_arm(Z, coef_tr, coef_va, coef_te, n_inits):
    """Fit n_inits networks on one descriptor and return their test predictions.

    Descriptors and coefficient targets are standardized on TRAIN only, exactly
    as in the Ridge probe, so the two probes differ in the estimator and nothing
    else.
    """
    Ztr_s, Zva_s, Zte_s = standardize(Z['train'], Z['val'], Z['test'])
    ymu = coef_tr.mean(0)
    ysd = coef_tr.std(0); ysd[ysd < 1e-12] = 1.0
    Ytr = (coef_tr - ymu) / ysd
    Yva = (coef_va - ymu) / ysd
    out = []
    for j in range(n_inits):
        pred_s, epochs = fit_mlp(Ztr_s, Ytr, Zva_s, Yva, Zte_s, init_seed=1234 + j)
        out.append((pred_s * ysd + ymu, epochs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--props', nargs='+', default=['dipole', 'polar'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--arms', nargs='+', default=['null', 'tda', 'random', 'shuffled'])
    ap.add_argument('--realizations', type=int, default=5)
    ap.add_argument('--inits', type=int, default=N_INITS)
    ap.add_argument('--cache', default='probe_cache')
    ap.add_argument('--out', default='results/residual_probe_mlp_per_seed.csv')
    a = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    zph_all = load_zph()
    basis = 'primary'
    rows = []
    for prop in a.props:
        pm = PRIMARY_METRIC[prop]
        metric_fn = dipole_metrics if prop == 'dipole' else polar_metrics
        for seed in a.seeds:
            ex = load_export(prop, seed, cache=a.cache)
            S_by = {s: gyration_lookup(ex[s]['idx'], cache=a.cache) for s in ('train', 'val', 'test')}
            ids = {s: ex[s]['idx'] for s in ('train', 'val', 'test')}
            base = metric_fn(ex['test']['pred'], ex['test']['target'])
            tgt, masks, names, _ = build_targets(prop, ex, ids, S_by, basis)
            for arm in a.arms:
                if arm == 'null':
                    rows.append(dict(property=prop, seed=seed, basis=basis, arm='null',
                                     realization=0, init=0, epochs='', baseline=base[pm],
                                     corrected=base[pm], delta=0.0, probe_r2_mean='',
                                     **{'base_' + k: v for k, v in base.items()}))
                    continue
                reals = range(a.realizations) if arm in ('random', 'shuffled') else [0]
                for real in reals:
                    Z = make_descriptor(arm, ids, zph_all, seed * 10 + real)
                    for j, (coef_pred, epochs) in enumerate(
                            run_arm(Z, tgt['train'], tgt['val'], tgt['test'], a.inits)):
                        r2 = float(np.mean(probe_r2(tgt['test'], coef_pred)))
                        m = corrected_metric(prop, ex['test']['pred'], S_by['test'],
                                             coef_pred, basis, masks['test'], ex['test']['target'])
                        rows.append(dict(property=prop, seed=seed, basis=basis, arm=arm,
                                         realization=real, init=j, epochs=epochs,
                                         baseline=base[pm], corrected=m[pm],
                                         delta=m[pm] - base[pm], probe_r2_mean=r2,
                                         **{'base_' + k: v for k, v in base.items()},
                                         **{'corr_' + k: v for k, v in m.items()}))
            print('%s seed %d done (%d rows so far)' % (prop, seed, len(rows)), flush=True)
    cols = ['property', 'seed', 'basis', 'arm', 'realization', 'init', 'epochs',
            'baseline', 'corrected', 'delta', 'probe_r2_mean']
    extra = sorted({k for r in rows for k in r} - set(cols))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote %s (%d rows)' % (a.out, len(rows)), flush=True)


if __name__ == '__main__':
    main()
