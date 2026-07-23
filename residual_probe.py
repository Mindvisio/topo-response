"""Bonus experiment: can z_PH predict the frozen baseline's residual?

The main study asked whether a TDA descriptor helps when trained jointly through a
FiLM path. This asks a strictly easier, more direct question: freeze the baseline,
take its error on each molecule, and see whether z_PH linearly predicts the part of
that error an equivariance-preserving correction is allowed to touch. A null result
here is much stronger evidence that z_PH carries no usable residual signal than the
main negative alone -- and unlike the main study, the probe R^2 is measured on the
full 66,485-molecule test set, so it is decisive at a single seed.

Coefficients are invariant (regressed from invariant z_PH); the correction is built
from equivariant tensors (see test_probe_equivariance.py). Two bases per property:
  primary   -- the minimal spec form
  secondary -- augmented with the gyration tensor S, declared in advance
Standardisation is fit on train only. All fits are Ridge; alpha is chosen on val.
"""
import argparse, json, os
import numpy as np
from schnetpack.data import ASEAtomsData

AU2D = 2.541746
# Exported dipoles are ALREADY in Debye (export_baseline_predictions.py converts),
# so the small-dipole floor is a plain 0.1 D and must not be divided by AU2D again.
DIPOLE_FLOOR_D = 0.1
DB = 'cache/squirl.db'


def load_export(prop, seed, split='topology_ood', cache='probe_cache'):
    z = np.load('%s/%s_%s_s%d.npz' % (cache, prop, split, seed))
    return {s: {'idx': z['%s_idx' % s], 'target': z['%s_target' % s], 'pred': z['%s_pred' % s]}
            for s in ('train', 'val', 'test')}


def load_zph(split='topology_ood'):
    """z_PH is indexed by global molecule id; return a lookup by id."""
    z = np.load('cache/zph.npy').astype(np.float64)
    return z


def gyration_tensors(ids):
    """S = centred-coordinate second moment, one 3x3 per molecule id."""
    db = ASEAtomsData(DB)
    out = np.zeros((len(ids), 3, 3))
    for k, i in enumerate(ids):
        X = np.asarray(db[int(i)]['_positions']).reshape(-1, 3)
        Xc = X - X.mean(0)
        out[k] = (Xc.T @ Xc) / len(Xc)
    return out


def standardize(train, *others):
    mu = train.mean(0); sd = train.std(0); sd[sd < 1e-8] = 1.0
    f = lambda A: (A - mu) / sd
    return (f(train),) + tuple(f(o) for o in others)


# ---- dipole targets: project the residual onto each equivariant basis vector ----
# The correction is a sum c_k * B_k(geometry). Given the true residual r = mu_true -
# mu_base, the best-fit coefficients (for a fixed molecule) solve the small normal
# system G c = B^T r with G = B^T B. We regress z_PH -> those per-molecule optimal
# coefficients, so the probe learns the coefficient an oracle would have used.
def dipole_targets(pred, target, S, basis):
    r = target - pred                                    # (N,3) residual to explain
    N = len(pred)
    if basis == 'primary':
        n = np.linalg.norm(pred, axis=1, keepdims=True)
        u = np.where(n > 1e-9, pred / np.maximum(n, 1e-9), 0.0)
        a = (r * u).sum(1)                               # scalar projection onto unit(mu_base)
        small = (n[:, 0] < DIPOLE_FLOOR_D)               # |mu_base| < 0.1 D -> correction is ~0
        return a[:, None], ['a'], small, np.ones(N)      # 1x1 Gram: conditioning is trivially 1
    Smu = np.einsum('nij,nj->ni', S, pred)
    S2mu = np.einsum('nij,nj->ni', S, Smu)
    B = np.stack([pred, Smu, S2mu], axis=2)              # (N,3,3): columns are the basis vectors
    # per-molecule least squares of r onto the 3 basis vectors, ridge-stabilised
    G_raw = np.einsum('nik,nil->nkl', B, B)
    cond = np.linalg.cond(G_raw)
    G = G_raw + 1e-6 * np.eye(3)
    rhs = np.einsum('nik,ni->nk', B, r)
    coef = np.linalg.solve(G, rhs)                       # (N,3) = [a-ish, b, c]
    small = (np.linalg.norm(pred, axis=1) < DIPOLE_FLOOR_D)
    return coef, ['coef0', 'coef1', 'coef2'], small, cond


# ---- polar targets: same idea on symmetric 3x3 tensors, Frobenius inner product ----
def _dev(A):
    tr = np.einsum('nii->n', A) / 3.0
    return A - tr[:, None, None] * np.eye(3)


def polar_targets(pred, target, S, basis):
    Pm = pred.reshape(-1, 3, 3); Tm = target.reshape(-1, 3, 3)
    R = Tm - Pm                                          # residual tensor
    Q = _dev(Pm)
    I = np.repeat(np.eye(3)[None], len(Pm), axis=0)
    if basis == 'primary':
        comps = [I, Q]
        names = ['a_iso', 'b_dev']
    else:
        SQ = 0.5 * (np.einsum('nij,njk->nik', S, Q) + np.einsum('nij,njk->nik', Q, S))
        comps = [I, Q, np.ascontiguousarray(S), SQ]
        names = ['a_iso', 'b_dev', 'c_S', 'd_SQ']
    B = np.stack(comps, axis=1)                          # (N,k,3,3)
    fro = lambda X, Y: (X.reshape(len(X), -1) * Y.reshape(len(Y), -1)).sum(1)
    k = B.shape[1]
    G = np.zeros((len(Pm), k, k))
    for i in range(k):
        for j in range(k):
            G[:, i, j] = fro(B[:, i], B[:, j])
    cond = np.linalg.cond(G)
    G = G + 1e-6 * np.eye(k)
    rhs = np.stack([fro(B[:, i], R) for i in range(k)], axis=1)
    coef = np.linalg.solve(G, rhs)                       # (N,k)
    aniso = np.sqrt(np.maximum(fro(Q, Q), 0.0))
    near_iso = aniso < 0.05 * np.sqrt(fro(Pm, Pm) + 1e-12)  # ill-conditioned b when almost isotropic
    return coef, names, near_iso, cond


# ---- reconstruct the corrected prediction from fitted coefficients ----
def apply_dipole(pred, S, coef, basis):
    if basis == 'primary':
        n = np.linalg.norm(pred, axis=1, keepdims=True)
        u = np.where(n > 1e-9, pred / np.maximum(n, 1e-9), 0.0)
        return pred + coef[:, [0]] * u
    Smu = np.einsum('nij,nj->ni', S, pred)
    S2mu = np.einsum('nij,nj->ni', S, Smu)
    return pred + coef[:, [0]] * pred + coef[:, [1]] * Smu + coef[:, [2]] * S2mu


def apply_polar(pred, S, coef, basis):
    Pm = pred.reshape(-1, 3, 3)
    I = np.repeat(np.eye(3)[None], len(Pm), axis=0)
    Q = _dev(Pm)
    out = Pm + coef[:, 0, None, None] * I + coef[:, 1, None, None] * Q
    if basis != 'primary':
        SQ = 0.5 * (np.einsum('nij,njk->nik', S, Q) + np.einsum('nij,njk->nik', Q, S))
        out = out + coef[:, 2, None, None] * np.ascontiguousarray(S) + coef[:, 3, None, None] * SQ
    return out.reshape(-1, 9)


# ---- metrics, matched to the canonical eval scripts ----
def dipole_metrics(pred, target):
    d = pred - target
    comp = np.abs(d).mean()
    vec = np.linalg.norm(d, axis=1).mean()
    mag = np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(target, axis=1)).mean()
    m = np.linalg.norm(target, axis=1) > 0.1
    cos = (pred[m] * target[m]).sum(1) / (np.linalg.norm(pred[m], axis=1) * np.linalg.norm(target[m], axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return dict(compMAE=comp, vecMAE=vec, magMAE=mag, angErr=ang)


def polar_metrics(pred, target):
    P = pred.reshape(-1, 3, 3); T = target.reshape(-1, 3, 3)
    P = 0.5 * (P + P.transpose(0, 2, 1)); T = 0.5 * (T + T.transpose(0, 2, 1))
    frob = np.linalg.norm((P - T).reshape(-1, 9), axis=1).mean()
    elem = np.abs(P - T).mean()
    iso = np.abs((np.einsum('nii->n', P) - np.einsum('nii->n', T)) / 3).mean()
    ep = np.linalg.eigvalsh(P); et = np.linalg.eigvalsh(T)
    eig = np.abs(ep - et).mean()
    return dict(Frob=frob, elem=elem, iso=iso, eigMAE=eig)


def _ridge_solve(X, Y, alpha):
    """Closed-form ridge with the intercept column left UNPENALISED.

    The last design column is the constant 1.  Penalising it would shrink the
    fitted mean toward zero, so a large alpha would not reduce the probe to
    'predict the training mean' and a negative R^2 could no longer be read as
    'no better than predicting the mean'.
    """
    from numpy.linalg import solve
    d = X.shape[1]
    P = alpha * np.eye(d)
    P[-1, -1] = 0.0
    return solve(X.T @ X + P, X.T @ Y)


def fit_predict(Z, coef_tr, coef_va, alphas):
    """Standardise on train, choose alpha on val, refit on train+val, predict test.

    Coefficient targets are standardised per column on train as well: the
    secondary bases mix tensors of different physical scale, so an unscaled
    multi-output MSE would let one coordinate dominate the alpha choice.
    """
    add1 = lambda A: np.hstack([A, np.ones((len(A), 1))])
    Ztr_s, Zva_s, Zte_s = standardize(Z['train'], Z['val'], Z['test'])
    Ztr_s, Zva_s, Zte_s = add1(Ztr_s), add1(Zva_s), add1(Zte_s)
    ymu = coef_tr.mean(0)
    ysd = coef_tr.std(0); ysd[ysd < 1e-12] = 1.0
    Ytr = (coef_tr - ymu) / ysd
    Yva = (coef_va - ymu) / ysd
    best, best_mse = alphas[0], np.inf
    for al in alphas:
        W = _ridge_solve(Ztr_s, Ytr, al)
        mse = ((Zva_s @ W - Yva) ** 2).mean()
        if mse < best_mse:
            best_mse, best = mse, al
    W = _ridge_solve(np.vstack([Ztr_s, Zva_s]), np.vstack([Ytr, Yva]), best)
    return (Zte_s @ W) * ysd + ymu, best


def probe_r2(coef_true_te, coef_pred_te):
    """Coefficient-of-determination of the probe on the held-out coefficients."""
    ss_res = ((coef_true_te - coef_pred_te) ** 2).sum(0)
    ss_tot = ((coef_true_te - coef_true_te.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def make_descriptor(kind, ids, zph_all, seed):
    """Return the (train,val,test) descriptor matrices for one control arm."""
    base = {s: zph_all[ids[s]] for s in ('train', 'val', 'test')}
    if kind == 'tda':
        return base
    if kind == 'random':
        rng = np.random.default_rng(1000 + seed)
        mu = base['train'].mean(0); sd = base['train'].std(0)
        return {s: rng.normal(mu, sd + 1e-9, size=base[s].shape) for s in base}
    if kind == 'shuffled':
        rng = np.random.default_rng(2000 + seed)
        out = {}
        for s in base:
            perm = rng.permutation(len(base[s]))
            out[s] = base[s][perm]
        return out
    raise ValueError(kind)


_GYR = {}


def gyration_lookup(ids, split='topology_ood'):
    """S for the given global ids, cached across the whole run (indexed by id)."""
    if 'arr' not in _GYR:
        path = 'probe_cache/gyration_all.npy'
        if os.path.exists(path):
            _GYR['arr'] = np.load(path)
        else:
            sp = np.load('cache/split_%s.npz' % split)
            every = np.concatenate([sp['train_idx'], sp['val_idx'], sp['test_idx']])
            hi = int(every.max()) + 1
            arr = np.zeros((hi, 3, 3), dtype=np.float64)
            arr[every] = gyration_tensors(every)
            np.save(path, arr)
            _GYR['arr'] = arr
    return _GYR['arr'][ids]


def run_arm(coef_tr, coef_va, coef_te, Z, alphas):
    """Fit the probe for one descriptor arm; return predicted test coeffs + diagnostics."""
    from scipy.stats import spearmanr
    coef_pred_te, alpha = fit_predict(Z, coef_tr, coef_va, alphas)
    r2 = probe_r2(coef_te, coef_pred_te)
    cmae = np.abs(coef_te - coef_pred_te).mean(0)
    sp = []
    for j in range(coef_te.shape[1]):
        rho = spearmanr(coef_te[:, j], coef_pred_te[:, j]).correlation
        sp.append(0.0 if rho != rho else float(rho))
    return dict(coef_pred=coef_pred_te, alpha=float(alpha),
                r2=r2.tolist(), coef_mae=cmae.tolist(), spearman=sp)


def corrected_metric(prop, pred, S, coef_pred, basis, small, target):
    """Apply the correction (zeroed on small / ill-conditioned rows) and score it."""
    if prop == 'dipole':
        corr = apply_dipole(pred, S, coef_pred, basis)
    else:
        corr = apply_polar(pred, S, coef_pred, basis)
    corr[small] = pred[small]
    return dipole_metrics(corr, target) if prop == 'dipole' else polar_metrics(corr, target)


ALPHAS = [10.0 ** k for k in range(-6, 7)]
PRIMARY_METRIC = {'dipole': 'compMAE', 'polar': 'Frob'}


def build_targets(prop, ex, split_ids, S_by_split, basis):
    """Per-split invariant coefficient targets + the small/ill-conditioned masks."""
    tgt = {}
    masks = {}
    names = None
    cond_test = None
    for s in ('train', 'val', 'test'):
        fn = dipole_targets if prop == 'dipole' else polar_targets
        coef, nm, mask, cond = fn(ex[s]['pred'], ex[s]['target'], S_by_split[s], basis)
        coef = coef.copy()
        coef[mask] = 0.0                                 # don't ask the probe to fit noise where correction is defined-zero
        tgt[s] = coef; masks[s] = mask; names = nm
        if s == 'test':
            cond_test = cond
    return tgt, masks, names, cond_test


def arm_realizations(kind, seed):
    """random and shuffled get 5 independent realizations; tda/null a single one."""
    if kind in ('random', 'shuffled'):
        return list(range(5))
    return [0]


def sensitivity_direction(prop, ex, S_by_split, coef_full, masks, basis, alphas):
    """Refit using ONLY a val-derived split (no in-sample train residuals) and
    report the sign of Delta on test, to check the direction is not an artefact
    of the baseline's optimistic train residuals."""
    nva = len(ex['val']['pred'])
    rng = np.random.default_rng(12345)
    perm = rng.permutation(nva)
    half = nva // 2
    pt, pv = perm[:half], perm[half:]
    Zva = load_zph()[ex['val']['idx']]
    Ztr_s, Zva_s, Zte_s = standardize(Zva[pt], Zva[pv], load_zph()[ex['test']['idx']])
    add1 = lambda A: np.hstack([A, np.ones((len(A), 1))])
    Ztr_s, Zva_s, Zte_s = add1(Ztr_s), add1(Zva_s), add1(Zte_s)
    from numpy.linalg import solve
    ytr, yv = coef_full['val'][pt], coef_full['val'][pv]
    d = Ztr_s.shape[1]
    best, bmse = alphas[0], np.inf
    for al in alphas:
        W = solve(Ztr_s.T @ Ztr_s + al * np.eye(d), Ztr_s.T @ ytr)
        mse = ((Zva_s @ W - yv) ** 2).mean()
        if mse < bmse:
            bmse, best = mse, al
    Xtv = np.vstack([Ztr_s, Zva_s]); Ytv = np.vstack([ytr, yv])
    W = solve(Xtv.T @ Xtv + best * np.eye(d), Xtv.T @ Ytv)
    coef_te = Zte_s @ W
    base = (dipole_metrics if prop == 'dipole' else polar_metrics)(ex['test']['pred'], ex['test']['target'])
    corr = corrected_metric(prop, ex['test']['pred'], S_by_split['test'], coef_te, basis,
                            masks['test'], ex['test']['target'])
    pm = PRIMARY_METRIC[prop]
    return float(corr[pm] - base[pm])


def main():
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument('--props', nargs='+', default=['dipole', 'polar'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--bases', nargs='+', default=['primary', 'secondary'])
    ap.add_argument('--arms', nargs='+', default=['null', 'tda', 'random', 'shuffled'])
    ap.add_argument('--out', default='results/residual_probe_per_seed.csv')
    a = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    zph_all = load_zph()
    rows = []
    for prop in a.props:
        pm = PRIMARY_METRIC[prop]
        for seed in a.seeds:
            ex = load_export(prop, seed)
            S_by = {s: gyration_lookup(ex[s]['idx']) for s in ('train', 'val', 'test')}
            ids = {s: ex[s]['idx'] for s in ('train', 'val', 'test')}
            base = (dipole_metrics if prop == 'dipole' else polar_metrics)(ex['test']['pred'], ex['test']['target'])
            for basis in a.bases:
                tgt, masks, names, cond_te = build_targets(prop, ex, ids, S_by, basis)
                frac_small = float(masks['test'].mean())
                cond_med = float(np.median(cond_te))
                cond_hi = float((cond_te > 1e6).mean())   # share with a badly conditioned basis
                for arm in a.arms:
                    if arm == 'null':
                        row = dict(property=prop, seed=seed, basis=basis, arm='null', realization=0,
                                   alpha='', baseline=base[pm], corrected=base[pm], delta=0.0,
                                   probe_r2_mean='', coef_mae_mean='', spearman_mean='',
                                   frac_small=frac_small, cond_median=cond_med,
                                   cond_frac_gt_1e6=cond_hi, sens_delta='')
                        for k, v in base.items():
                            row['base_' + k] = v; row['corr_' + k] = v
                        rows.append(row); continue
                    for real in arm_realizations(arm, seed):
                        Z = make_descriptor(arm if arm != 'tda' else 'tda',
                                            ids, zph_all, seed * 10 + real)
                        res = run_arm(tgt['train'], tgt['val'], tgt['test'], Z, ALPHAS)
                        m = corrected_metric(prop, ex['test']['pred'], S_by['test'],
                                             res['coef_pred'], basis, masks['test'], ex['test']['target'])
                        sens = ''
                        if arm == 'tda' and basis == 'primary':
                            sens = sensitivity_direction(prop, ex, S_by, tgt, masks, basis, ALPHAS)
                        row = dict(property=prop, seed=seed, basis=basis, arm=arm, realization=real,
                                   alpha=res['alpha'], baseline=base[pm], corrected=m[pm],
                                   delta=m[pm] - base[pm],
                                   probe_r2_mean=float(np.mean(res['r2'])),
                                   coef_mae_mean=float(np.mean(res['coef_mae'])),
                                   spearman_mean=float(np.mean(res['spearman'])),
                                   frac_small=frac_small, cond_median=cond_med,
                                   cond_frac_gt_1e6=cond_hi, sens_delta=sens)
                        for k, v in base.items():
                            row['base_' + k] = v
                        for k, v in m.items():
                            row['corr_' + k] = v
                        rows.append(row)
            print('%s seed %d done (frac_small=%.4f, cond_median=%.3g)'
                  % (prop, seed, frac_small, cond_med), flush=True)
    cols = ['property', 'seed', 'basis', 'arm', 'realization', 'alpha', 'baseline', 'corrected',
            'delta', 'probe_r2_mean', 'coef_mae_mean', 'spearman_mean', 'frac_small',
            'cond_median', 'cond_frac_gt_1e6', 'sens_delta']
    extra = sorted({k for r in rows for k in r} - set(cols))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote %s (%d rows)' % (a.out, len(rows)), flush=True)


if __name__ == '__main__':
    main()
