"""P0 TDA features: per-molecule z_PH = [BettiCurve(H0,H1) 64x2, PersistenceEntropy 2] = 130-dim.
Reuses the qm9-egnn-tda pipeline (center -> unit-diameter scale -> VietorisRips H0/H1)."""
import numpy as np, h5py, json, warnings
from multiprocessing import Pool
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy

H5 = '/root/topoci/data/squirl/SQuIRL_v1.0.h5'
BINS = 64; MAXDIM = 1
_vr = _betti = _ent = None

def _init():
    global _vr, _betti, _ent
    _vr = VietorisRipsPersistence(homology_dimensions=list(range(MAXDIM + 1)), metric='euclidean', n_jobs=1)
    _betti = BettiCurve(n_bins=BINS); _ent = PersistenceEntropy()

def compute_vec(coords):
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords - coords.mean(0, keepdims=True)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)); diam = float(d.max())
    if diam > 0: coords = coords / diam
    X = coords[None, :, :]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        diag = _vr.fit_transform(X); betti = _betti.fit_transform(diag); ent = _ent.fit_transform(diag)
    return np.concatenate([betti.reshape(1, -1), ent.reshape(1, -1)], 1)[0].astype(np.float32)

def work(args):
    mid, pos = args
    try: return mid, compute_vec(pos)
    except Exception: return mid, None

def main():
    idx = json.load(open('cache/index.json')); ids = [r['id'] for r in idx]
    with h5py.File(H5, 'r') as f:
        items = [(mid, np.asarray(f['data'][mid]['structure']['pos'][...])) for mid in ids]
    print('loaded %d point clouds; computing z_PH...' % len(items), flush=True)
    with Pool(10, initializer=_init) as P:
        res = P.map(work, items, chunksize=200)
    order, vecs, fails = [], [], []
    for mid, v in res:
        if v is None: fails.append(mid)
        else: order.append(mid); vecs.append(v)
    Z = np.stack(vecs).astype(np.float32)
    np.save('cache/zph.npy', Z)
    json.dump(dict(order=order, dim=int(Z.shape[1]), bins=BINS, maxdim=MAXDIM, n_fail=len(fails), fails=fails[:50]),
              open('cache/zph_meta.json', 'w'))
    h1 = Z[:, BINS:2 * BINS]; nz = float((np.count_nonzero(h1, axis=1) > 0).mean())
    print('z_PH DONE: %d x %d, %d fails | H1-nonzero rate %.2f%%' % (Z.shape[0], Z.shape[1], len(fails), nz * 100), flush=True)

if __name__ == '__main__': main()
