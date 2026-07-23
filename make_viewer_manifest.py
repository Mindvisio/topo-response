"""Provenance manifest for the published viewer assets.

Records, for every molecule shown in index.html: its global row in
cache/index.json, which split it belongs to, the checkpoint whose predictions
are displayed, and SHA256 digests of the density cube and the fitted-charge
file.  This lets anyone check that the viewer shows held-out molecules scored
by the stated baseline, without rerunning inference.
"""
import hashlib, json, re
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    html = open('index.html', encoding='utf-8').read()
    mols = json.loads(re.search(r'const MOLS = (\[.*?\]);', html, re.S).group(1))
    preds = json.load(open('viewer_preds.json'))
    idx = json.load(open('cache/index.json'))
    pos = {r['id']: i for i, r in enumerate(idx)}
    z = np.load('cache/split_topology_ood.npz')
    sets = {k: set(z[k].tolist()) for k in z.files}

    ckpt = sorted(__import__('glob').glob('ckpt_topology_ood_none_s0/best-*.ckpt'))[0]
    entries = []
    for m in mols:
        row = pos[m['id']]
        where = [k.replace('_idx', '') for k, s in sets.items() if row in s]
        entries.append(dict(
            id=m['id'], smiles=m['smiles'], row=row, split=where,
            n_atoms=len(m['atoms']),
            dipole_true_au=m['dip'], dipole_pred_au=preds[m['id']],
            cube='dens/%s.cube' % m['id'],
            cube_sha256=sha256('dens/%s.cube' % m['id'])))
    out = dict(
        description=('Provenance for the molecules shown in index.html. Dipoles are in '
                     'atomic units as stored; the viewer multiplies by 2.541746 for display in Debye.'),
        split_file='cache/split_topology_ood.npz',
        split_sizes={k: int(len(v)) for k, v in sets.items()},
        prediction_checkpoint=ckpt,
        prediction_checkpoint_sha256=sha256(ckpt),
        charges_file='dens/charges.json',
        charges_sha256=sha256('dens/charges.json'),
        density_method='RHF/6-31G* at the SQuIRL geometry, isosurface 0.002 e/a0^3',
        molecules=entries)
    json.dump(out, open('viewer_manifest.json', 'w'), indent=1)
    bad = [e['id'] for e in entries if e['split'] != ['test']]
    print('wrote viewer_manifest.json for %d molecules' % len(entries))
    print('all held-out:', not bad, '' if not bad else '(NOT held out: %s)' % bad)


if __name__ == '__main__':
    main()
