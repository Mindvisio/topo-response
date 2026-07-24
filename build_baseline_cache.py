"""Padded coordinates + element one-hots for the non-equivariant baselines.

A plain network has no notion of rotation, so it has to be handed raw numbers in
some fixed layout: positions in the dataset's own frame, atoms in the dataset's
own order, zero-padded to the largest molecule. That is precisely the input a
model without geometric inductive bias works from.
"""
import json
import numpy as np
from schnetpack.data import ASEAtomsData

MAXAT = 29                      # largest molecule in QM9
ELEMS = [1, 6, 7, 8, 9]         # H C N O F


def main():
    idx = json.load(open('cache/index.json'))
    db = ASEAtomsData('cache/squirl.db')
    n = len(idx)
    pos = np.zeros((n, MAXAT, 3), dtype=np.float32)
    onehot = np.zeros((n, MAXAT, len(ELEMS)), dtype=np.float32)
    mask = np.zeros((n, MAXAT), dtype=np.float32)
    comp = np.zeros((n, len(ELEMS)), dtype=np.float32)
    emap = {z: k for k, z in enumerate(ELEMS)}
    for i in range(n):
        p = db[i]
        R = np.asarray(p['_positions'], dtype=np.float32).reshape(-1, 3)
        Z = np.asarray(p['_atomic_numbers']).reshape(-1).astype(int)
        m = len(Z)
        R = R - R.mean(0)                       # centering is free: translation is trivial
        pos[i, :m] = R
        mask[i, :m] = 1.0
        for k, z in enumerate(Z):
            onehot[i, k, emap[z]] = 1.0
            comp[i, emap[z]] += 1.0
        if i % 20000 == 0:
            print('  %d / %d' % (i, n), flush=True)
    np.savez_compressed('cache/baseline_inputs.npz', pos=pos, onehot=onehot,
                        mask=mask, comp=comp,
                        n_atoms=np.array([m['n_atoms'] for m in idx], dtype=np.float32),
                        n_rings=np.array([m['n_rings'] for m in idx], dtype=np.float32))
    print('wrote cache/baseline_inputs.npz', pos.shape, flush=True)


if __name__ == '__main__':
    main()
