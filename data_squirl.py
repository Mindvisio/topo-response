"""SQuIRL data pipeline (P0): h5 -> per-molecule (z, pos, dipole[3], polarizability[3,3]) + splits.

Equivariant targets: dipole = l=1 vector (Debye); polarizability = symmetric 3x3 = l=0 (isotropic,
trace/3) (+) l=2 (traceless anisotropic part), in a.u. Geometry structure/pos in Angstrom.
"""
import h5py, numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

H5 = '/root/topoci/data/squirl/SQuIRL_v1.0.h5'


def build_index(h5_path=H5):
    idx = []
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        for mid in data.keys():
            g = data[mid]
            z = g['structure']['z'][...]
            smiles = g['structure'].attrs.get('smiles')
            n_rings = 0
            if smiles:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    n_rings = m.GetRingInfo().NumRings()
            idx.append(dict(id=mid, n_atoms=int(len(z)), n_heavy=int((np.asarray(z) > 1).sum()),
                            n_rings=int(n_rings), smiles=(smiles or '')))
    return idx


def make_splits(idx, seed=0, val_frac=0.1, test_frac=0.1):
    rng = np.random.default_rng(seed)
    ids = np.array([r['id'] for r in idx]); N = len(ids)
    n_heavy = np.array([r['n_heavy'] for r in idx]); n_rings = np.array([r['n_rings'] for r in idx])
    def _split_rest(rest_ids):
        pr = rng.permutation(len(rest_ids)); nv = int(val_frac * len(rest_ids))
        return rest_ids[pr[nv:]].tolist(), rest_ids[pr[:nv]].tolist()
    perm = rng.permutation(N); n_test = int(test_frac * N); n_val = int(val_frac * N)
    random = dict(train=ids[perm[n_test + n_val:]].tolist(), val=ids[perm[n_test:n_test + n_val]].tolist(),
                  test=ids[perm[:n_test]].tolist())
    thr = float(np.quantile(n_heavy, 0.8))            # size-OOD: top ~20% largest -> test
    tr, va = _split_rest(ids[n_heavy <= thr])
    size_ood = dict(train=tr, val=va, test=ids[n_heavy > thr].tolist(), threshold_heavy=thr)
    tr2, va2 = _split_rest(ids[n_rings <= 1])         # topology-OOD: 0-1 ring train, >=2 rings test
    topology_ood = dict(train=tr2, val=va2, test=ids[n_rings >= 2].tolist())
    return dict(random=random, size_ood=size_ood, topology_ood=topology_ood)


def load_molecule(f, mid):
    g = f['data'][mid]
    return (np.asarray(g['structure']['z'][...]), np.asarray(g['structure']['pos'][...]),
            np.asarray(g['electrostatics']['dipole_moment'][...]),
            np.asarray(g['electrostatics']['dipole_polarizability'][...]))


def alpha_irreps(alpha):
    """Split symmetric 3x3 polarizability into l=0 (iso scalar) and l=2 (traceless 3x3)."""
    iso = np.trace(alpha) / 3.0
    return float(iso), (alpha - iso * np.eye(3))
