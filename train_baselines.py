"""Baselines without geometric inductive bias, for the comparison the brief asks for.

Three reference points against the equivariant model, all on the same topology-OOD
split and the same units as the main table:

  * FCNN on raw padded coordinates -> the three dipole components.  Nothing in this
    model knows that rotating a molecule should rotate its dipole, so it also gets
    scored on a randomly rotated copy of the test set: the gap between those two
    numbers is what the geometric inductive bias buys.
  * Gradient boosting on invariant descriptors (composition, size, rings, z_PH) ->
    the two invariant scalars |mu| and the isotropic polarizability.  This is the
    tabular reference: how much of the task needs no geometry at all.
  * The training-mean predictor, as the floor any model must beat.

The equivariant numbers are read from the frozen baseline exports in probe_cache/,
so nothing is retrained here.
"""
import json
import numpy as np
import torch
import torch.nn as nn

AU2D = 2.541746
SEEDS = [0, 1, 2, 3, 4]


def load_split():
    z = np.load('cache/split_topology_ood.npz')
    return z['train_idx'], z['val_idx'], z['test_idx']


def load_targets(prop, seed=0):
    """Targets straight from the frozen export: same units, same rows as the main study."""
    z = np.load('probe_cache/%s_topology_ood_s%d.npz' % (prop, seed))
    return {s: (z['%s_idx' % s], z['%s_target' % s], z['%s_pred' % s])
            for s in ('train', 'val', 'test')}


def rot_matrices(n, seed):
    """Random rotations via QR of a Gaussian matrix, determinant fixed to +1."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        out[i] = Q
    return out


def fcnn_dipole(seed, cache, tgt, hidden=(512, 256, 128), epochs=120, batch=512, lr=1e-3):
    """Plain MLP: flattened coordinates + element one-hots -> dipole components (D)."""
    pos, oh, mask = cache['pos'], cache['onehot'], cache['mask']
    feats = np.concatenate([pos, oh, mask[:, :, None]], axis=2).reshape(len(pos), -1)
    tr_i, tr_y, _ = tgt['train']
    va_i, va_y, _ = tgt['val']
    te_i, te_y, _ = tgt['test']
    mu, sd = feats[tr_i].mean(0), feats[tr_i].std(0) + 1e-6
    prep = lambda I: torch.tensor((feats[I] - mu) / sd, dtype=torch.float32)
    Xtr, Xva, Xte = prep(tr_i), prep(va_i), prep(te_i)
    Ytr = torch.tensor(tr_y, dtype=torch.float32)
    Yva = torch.tensor(va_y, dtype=torch.float32)

    torch.manual_seed(seed)
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, 3)]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            j = perm[i:i + batch]
            opt.zero_grad(); lossf(net(Xtr[j]), Ytr[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xva), Yva))
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = net(Xte).numpy()

    # the same test molecules, rigidly rotated: an equivariant model is unchanged here
    R = rot_matrices(len(te_i), seed + 991)
    prot = pos[te_i] @ R.transpose(0, 2, 1)
    frot = np.concatenate([prot, oh[te_i], mask[te_i][:, :, None]], axis=2).reshape(len(te_i), -1)
    with torch.no_grad():
        pred_rot = net(torch.tensor((frot - mu) / sd, dtype=torch.float32)).numpy()
    y_rot = np.einsum('nij,nj->ni', R, te_y)      # the true dipole rotates with the molecule
    # compMAE averages |e_x|+|e_y|+|e_z|, an L1 norm, which is NOT invariant under
    # rotation even for a perfectly equivariant model. The rotation comparison
    # therefore also records the vector L2 error, which is.
    return dict(compMAE=float(np.abs(pred - te_y).mean()),
                compMAE_rotated=float(np.abs(pred_rot - y_rot).mean()),
                vecMAE=float(np.linalg.norm(pred - te_y, axis=1).mean()),
                vecMAE_rotated=float(np.linalg.norm(pred_rot - y_rot, axis=1).mean()),
                epochs=ep + 1)


def invariant_features(cache):
    """Everything a tabular model can legitimately see: no orientation anywhere."""
    z = np.load('cache/zph.npy').astype(np.float32)
    return np.hstack([cache['comp'], cache['n_atoms'][:, None],
                      cache['n_rings'][:, None], z])


def tabular_scalars(cache):
    """Gradient boosting on invariant descriptors for the two invariant scalars."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = invariant_features(cache)
    out = {}

    d = load_targets('dipole')
    tr_i, tr_y, _ = d['train']; va_i, va_y, _ = d['val']; te_i, te_y, te_p = d['test']
    ytr, yva, yte = (np.linalg.norm(a, axis=1) for a in (tr_y, va_y, te_y))
    m = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(X[tr_i], ytr)
    # val is drawn from the training regime, test is the shifted one: reporting both
    # separates "the model learned nothing" from "the model does not extrapolate"
    out['mu_abs'] = dict(tabular=float(np.abs(m.predict(X[te_i]) - yte).mean()),
                         tabular_val=float(np.abs(m.predict(X[va_i]) - yva).mean()),
                         mean_predictor_val=float(np.abs(ytr.mean() - yva).mean()),
                         equivariant=float(np.abs(np.linalg.norm(te_p, axis=1) - yte).mean()),
                         mean_predictor=float(np.abs(ytr.mean() - yte).mean()), unit='D')

    p = load_targets('polar')
    tr_i, tr_y, _ = p['train']; va_i, va_y, _ = p['val']; te_i, te_y, te_p = p['test']
    iso = lambda A: A.reshape(-1, 3, 3)[:, [0, 1, 2], [0, 1, 2]].sum(1) / 3.0
    ytr, yva, yte = iso(tr_y), iso(va_y), iso(te_y)
    m = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(X[tr_i], ytr)
    out['alpha_iso'] = dict(tabular=float(np.abs(m.predict(X[te_i]) - yte).mean()),
                            tabular_val=float(np.abs(m.predict(X[va_i]) - yva).mean()),
                            mean_predictor_val=float(np.abs(ytr.mean() - yva).mean()),
                            equivariant=float(np.abs(iso(te_p) - yte).mean()),
                            mean_predictor=float(np.abs(ytr.mean() - yte).mean()), unit='a.u.')
    return out


def main():
    cache = dict(np.load('cache/baseline_inputs.npz'))
    tgt = load_targets('dipole')
    res = {'fcnn_dipole': [], 'equivariant_dipole_compMAE': []}
    for s in SEEDS:
        r = fcnn_dipole(s, cache, tgt)
        res['fcnn_dipole'].append(r)
        print('  FCNN seed %d: compMAE %.4f D | same molecules rotated %.4f D (%d epochs)'
              % (s, r['compMAE'], r['compMAE_rotated'], r['epochs']), flush=True)
    for s in SEEDS:
        z = np.load('probe_cache/dipole_topology_ood_s%d.npz' % s)
        res['equivariant_dipole_compMAE'].append(
            float(np.abs(z['test_pred'] - z['test_target']).mean()))
    tr_i, tr_y, _ = tgt['train']; te_i, te_y, _ = tgt['test']
    res['mean_predictor_dipole_compMAE'] = float(np.abs(tr_y.mean(0) - te_y).mean())
    res['scalars'] = tabular_scalars(cache)
    json.dump(res, open('results/baselines.json', 'w'), indent=1)

    f = [r['compMAE'] for r in res['fcnn_dipole']]
    fr = [r['compMAE_rotated'] for r in res['fcnn_dipole']]
    e = res['equivariant_dipole_compMAE']
    print('\n  dipole compMAE (D), mean over %d seeds' % len(SEEDS))
    print('    equivariant PaiNN      %.4f' % np.mean(e))
    print('    FCNN on coordinates    %.4f' % np.mean(f))
    print('    FCNN, rotated test     %.4f' % np.mean(fr))
    print('    training-mean          %.4f' % res['mean_predictor_dipole_compMAE'])
    for k, v in res['scalars'].items():
        print('  %s (%s): equivariant %.4f | tabular+z_PH %.4f | mean %.4f'
              % (k, v['unit'], v['equivariant'], v['tabular'], v['mean_predictor']))
    print('wrote results/baselines.json')


def fcnn_polar(seed, cache, tgt, hidden=(512, 256, 128), epochs=120, batch=512, lr=1e-3):
    """Plain MLP on the same inputs, predicting the full 3x3 polarizability tensor.

    Frobenius error IS rotation-invariant for an equivariant model, since
    ||R E R^T||_F = ||E||_F, so the rotated column is a fair comparison here.
    """
    pos, oh, mask = cache['pos'], cache['onehot'], cache['mask']
    feats = np.concatenate([pos, oh, mask[:, :, None]], axis=2).reshape(len(pos), -1)
    tr_i, tr_y, _ = tgt['train']
    va_i, va_y, _ = tgt['val']
    te_i, te_y, _ = tgt['test']
    mu, sd = feats[tr_i].mean(0), feats[tr_i].std(0) + 1e-6
    prep = lambda I: torch.tensor((feats[I] - mu) / sd, dtype=torch.float32)
    Xtr, Xva, Xte = prep(tr_i), prep(va_i), prep(te_i)
    ysc = float(np.abs(tr_y).mean())
    Ytr = torch.tensor(tr_y / ysc, dtype=torch.float32)
    Yva = torch.tensor(va_y / ysc, dtype=torch.float32)

    torch.manual_seed(seed)
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, 9)]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            j = perm[i:i + batch]
            opt.zero_grad(); lossf(net(Xtr[j]), Ytr[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xva), Yva))
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state); net.eval()
    sym = lambda A: 0.5 * (A.reshape(-1, 3, 3) + A.reshape(-1, 3, 3).transpose(0, 2, 1))
    with torch.no_grad():
        P = sym(net(Xte).numpy() * ysc)
    T = sym(te_y)

    R = rot_matrices(len(te_i), seed + 991)
    prot = pos[te_i] @ R.transpose(0, 2, 1)
    frot = np.concatenate([prot, oh[te_i], mask[te_i][:, :, None]], axis=2).reshape(len(te_i), -1)
    with torch.no_grad():
        Prot = sym(net(torch.tensor((frot - mu) / sd, dtype=torch.float32)).numpy() * ysc)
    Trot = np.einsum('nij,njk,nlk->nil', R, T, R)      # A -> R A R^T
    frob = lambda A, B: float(np.linalg.norm((A - B).reshape(len(A), -1), axis=1).mean())
    return dict(Frob=frob(P, T), Frob_rotated=frob(Prot, Trot), epochs=ep + 1)


def painn_rotation_reference():
    """The equivariant model under the same rotations, from the frozen exports.

    Predictions rotate with the frame (verified numerically in e3_test.py), so the
    rotated error is R e for the dipole and R E R^T for the tensor.  Component-wise
    MAE is an L1 quantity and shifts; the vector L2 and Frobenius norms do not.
    """
    out = {}
    dc, dv, pf = [], [], []
    for s in SEEDS:
        z = np.load('probe_cache/dipole_topology_ood_s%d.npz' % s)
        e = z['test_pred'] - z['test_target']
        R = rot_matrices(len(e), s + 991)
        er = np.einsum('nij,nj->ni', R, e)
        dc.append((float(np.abs(e).mean()), float(np.abs(er).mean())))
        dv.append((float(np.linalg.norm(e, axis=1).mean()),
                   float(np.linalg.norm(er, axis=1).mean())))
        z = np.load('probe_cache/polar_topology_ood_s%d.npz' % s)
        E = (z['test_pred'] - z['test_target']).reshape(-1, 3, 3)
        R = rot_matrices(len(E), s + 991)
        Er = np.einsum('nij,njk,nlk->nil', R, E, R)
        f = lambda A: float(np.linalg.norm(A.reshape(len(A), -1), axis=1).mean())
        pf.append((f(E), f(Er)))
    out['dipole_compMAE'] = dc
    out['dipole_vecMAE'] = dv
    out['polar_Frob'] = pf
    return out


def run_polar():
    """Adds the polarizability reference models to results/baselines.json."""
    cache = dict(np.load('cache/baseline_inputs.npz'))
    tgt = load_targets('polar')
    res = json.load(open('results/baselines.json'))
    res['fcnn_polar'] = []
    for s in SEEDS:
        r = fcnn_polar(s, cache, tgt)
        res['fcnn_polar'].append(r)
        print('  FCNN polar seed %d: Frobenius %.4f a.u. | rotated %.4f (%d epochs)'
              % (s, r['Frob'], r['Frob_rotated'], r['epochs']), flush=True)
    res['equivariant_polar_Frob'] = []
    for s in SEEDS:
        z = np.load('probe_cache/polar_topology_ood_s%d.npz' % s)
        d = (z['test_pred'] - z['test_target'])
        res['equivariant_polar_Frob'].append(
            float(np.linalg.norm(d, axis=1).mean()))
    tr_i, tr_y, _ = tgt['train']; te_i, te_y, _ = tgt['test']
    res['mean_predictor_polar_Frob'] = float(
        np.linalg.norm(tr_y.mean(0)[None, :] - te_y, axis=1).mean())
    res['painn_rotation'] = painn_rotation_reference()
    json.dump(res, open('results/baselines.json', 'w'), indent=1)
    f = np.array([x['Frob'] for x in res['fcnn_polar']])
    fr = np.array([x['Frob_rotated'] for x in res['fcnn_polar']])
    print('\n  polarizability Frobenius (a.u.), mean over %d seeds' % len(SEEDS))
    print('    equivariant PaiNN   %.4f' % np.mean(res['equivariant_polar_Frob']))
    print('    FCNN                %.4f' % f.mean())
    print('    FCNN, rotated       %.4f' % fr.mean())
    print('    naive constant      %.4f' % res['mean_predictor_polar_Frob'])
    print('wrote results/baselines.json')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'polar':
        run_polar()
    else:
        main()
