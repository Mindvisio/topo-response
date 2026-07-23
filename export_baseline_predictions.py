"""Stage 1 of the residual probe: freeze a baseline checkpoint and export its
predictions on every split of one experiment split-file.

Only cond=none checkpoints are exported -- the probe asks what a plain
equivariant baseline gets wrong, so no conditioning path may be involved.
Targets are cloned before the forward pass: SchNetPack writes predictions into
the batch dict under the same key and would otherwise overwrite them.

    python export_baseline_predictions.py --property dipole --seed 0
"""
import argparse, hashlib, json, os, glob
import numpy as np
import torch
import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.properties as P
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, Polarizability, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
from train_dipole_tda import FlatMSE, FlatMAE

NBASIS = 128
AU2D = 2.541746
KEY = {'dipole': 'dipole_moment', 'polar': 'polarizability'}
CKPT = {'dipole': 'ckpt_topology_ood_none_s%d', 'polar': 'ckpt_polar_topology_ood_none_s%d'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--property', required=True, choices=['dipole', 'polar'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--split', default='topology_ood')
    ap.add_argument('--cutoff', type=float, default=5.0)
    ap.add_argument('--cache', default='cache/nbh_cache')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', default='probe_cache')
    a = ap.parse_args()

    key = KEY[a.property]
    ckpts = sorted(glob.glob((CKPT[a.property] % a.seed) + '/best-*.ckpt'))
    assert len(ckpts) == 1, 'expected exactly one best checkpoint, got %s' % ckpts
    ckpt = ckpts[0]

    tfs = [trn.SubtractCenterOfGeometry(),
           trn.CachedNeighborList(cache_path='%s_cut%g' % (a.cache, a.cutoff),
                                  neighbor_list=trn.ASENeighborList(cutoff=a.cutoff),
                                  keep_cache=True),
           trn.CastTo32()]
    dm = AtomsDataModule('cache/squirl.db', batch_size=a.batch,
                         split_file='cache/split_%s.npz' % a.split,
                         load_properties=[key], transforms=tfs, num_workers=6)
    dm.setup()

    painn = PaiNN(n_atom_basis=NBASIS, n_interactions=3,
                  radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=a.cutoff),
                  cutoff_fn=spk.nn.CosineCutoff(a.cutoff))
    head = (DipoleMoment(n_in=NBASIS, dipole_key=key, use_vector_representation=True)
            if a.property == 'dipole' else Polarizability(n_in=NBASIS, polarizability_key=key))
    model = NeuralNetworkPotential(representation=painn, input_modules=[PairwiseDistances()],
                                   output_modules=[head])
    task = AtomisticTask(model, outputs=[ModelOutput(name=key, loss_fn=FlatMSE(), loss_weight=1.0,
                                                     metrics={'MAE': FlatMAE()})],
                         optimizer_cls=torch.optim.AdamW, optimizer_args={'lr': 5e-4})
    task.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'], strict=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    task.to(dev); task.eval()

    dim = 3 if a.property == 'dipole' else 9
    store = {}
    for name, loader in [('train', dm.train_dataloader()), ('val', dm.val_dataloader()),
                         ('test', dm.test_dataloader())]:
        idx, tgt, prd = [], [], []
        for batch in loader:
            assert P.idx in batch, 'batch has no global molecule index'
            gid = batch[P.idx].detach().cpu().numpy().reshape(-1)
            t = batch[key].detach().clone().cpu().reshape(-1, dim)
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = task.model(batch)
            p = out[key].detach().cpu().reshape(-1, dim)
            assert len(gid) == len(t) == len(p), 'row count mismatch in %s' % name
            idx.append(gid); tgt.append(t.numpy()); prd.append(p.numpy())
        I = np.concatenate(idx); T = np.concatenate(tgt); Q = np.concatenate(prd)
        if a.property == 'dipole':
            T, Q = T * AU2D, Q * AU2D                       # a.u. -> Debye, as in the canonical eval
        else:
            sym = lambda X: 0.5 * (X.reshape(-1, 3, 3) + X.reshape(-1, 3, 3).transpose(0, 2, 1))
            T, Q = sym(T).reshape(-1, 9), sym(Q).reshape(-1, 9)
        for nm, arr in (('target', T), ('pred', Q)):
            assert np.isfinite(arr).all(), 'non-finite %s in %s' % (nm, name)
        assert len(np.unique(I)) == len(I), 'duplicate molecule indices in %s' % name
        store[name] = dict(idx=I, target=T, pred=Q)
        print('%-5s n=%6d  idx %d..%d' % (name, len(I), I.min(), I.max()), flush=True)

    d = store['test']['pred'] - store['test']['target']
    metric = (np.abs(d).mean() if a.property == 'dipole'
              else np.linalg.norm(d, axis=1).mean())
    label = 'compMAE (D)' if a.property == 'dipole' else 'Frobenius (a.u.)'
    print('TEST %s = %.4f   [reproduces the published baseline number for this seed]' % (label, metric), flush=True)

    os.makedirs(a.out, exist_ok=True)
    stem = '%s/%s_%s_s%d' % (a.out, a.property, a.split, a.seed)
    np.savez_compressed(stem + '.npz', **{'%s_%s' % (s, k): v for s, dd in store.items() for k, v in dd.items()})
    json.dump(dict(property=a.property, seed=a.seed, split=a.split, checkpoint=ckpt,
                   checkpoint_sha256=sha256(ckpt), cutoff=a.cutoff, units='Debye' if a.property == 'dipole' else 'a.u.',
                   test_metric=float(metric), metric_name=label,
                   n={k: int(len(v['idx'])) for k, v in store.items()}),
              open(stem + '.json', 'w'), indent=1)
    print('wrote %s.npz + .json' % stem, flush=True)


if __name__ == '__main__':
    main()
