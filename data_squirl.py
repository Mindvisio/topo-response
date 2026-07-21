import h5py, numpy as np
import warnings; warnings.filterwarnings('ignore', module=r'h5py')
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
H5 = '/home/yc-user/data/squirl/SQuIRL_v1.0.h5'
def build_index(h5_path=H5):
    idx = []
    with h5py.File(h5_path, 'r') as f:
        for mid in f['data'].keys():
            g = f['data'][mid]; z = g['structure']['z'][...]
            smiles = g['structure'].attrs.get('smiles'); nr = 0
            if smiles:
                m = Chem.MolFromSmiles(smiles)
                if m is not None: nr = m.GetRingInfo().NumRings()
            idx.append(dict(id=mid, n_atoms=int(len(z)), n_heavy=int((np.asarray(z)>1).sum()), n_rings=int(nr), smiles=(smiles or '')))
    return idx
def make_splits(idx, seed=0, val_frac=0.1, test_frac=0.1):
    rng = np.random.default_rng(seed); ids = np.array([r['id'] for r in idx]); N = len(ids)
    nr = np.array([r['n_rings'] for r in idx]); na = np.array([r['n_atoms'] for r in idx])
    def rest(rids):
        pr = rng.permutation(len(rids)); nv = int(val_frac*len(rids)); return rids[pr[nv:]].tolist(), rids[pr[:nv]].tolist()
    perm = rng.permutation(N); nt = int(test_frac*N); nv = int(val_frac*N)
    random = dict(train=ids[perm[nt+nv:]].tolist(), val=ids[perm[nt:nt+nv]].tolist(), test=ids[perm[:nt]].tolist())
    thr = float(np.quantile(na, 0.85)); tr, va = rest(ids[na<=thr])
    size_ood = dict(train=tr, val=va, test=ids[na>thr].tolist(), threshold_n_atoms=thr)
    tr2, va2 = rest(ids[nr<=1]); topology_ood = dict(train=tr2, val=va2, test=ids[nr>=2].tolist())
    return dict(random=random, size_ood=size_ood, topology_ood=topology_ood)
def load_molecule(f, mid):
    g = f['data'][mid]
    return (np.asarray(g['structure']['z'][...]), np.asarray(g['structure']['pos'][...]),
            np.asarray(g['electrostatics']['dipole_moment'][...]), np.asarray(g['electrostatics']['dipole_polarizability'][...]))
