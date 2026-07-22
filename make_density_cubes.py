"""Electron-density cubes for the six viewer molecules (RHF/6-31G* at the SQuIRL geometry)."""
import json, re, os, time
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

N = 48
HTML = 'index.html'
OUT = 'dens'

html = open(HTML).read()
MOLS = json.loads(re.search(r'const MOLS = (\[.*?\]);', html, re.S).group(1))
os.makedirs(OUT, exist_ok=True)

def compact(src, dst):
    lines = open(src).read().splitlines()
    nat = abs(int(lines[2].split()[0]))
    head, rest = lines[:6 + nat], lines[6 + nat:]
    v = np.array(' '.join(rest).split(), dtype=float)
    v[np.abs(v) < 1e-6] = 0.0
    with open(dst, 'w') as f:
        f.write('\n'.join(head) + '\n')
        buf = []
        for i, x in enumerate(v):
            buf.append('0' if x == 0.0 else '%.4e' % x)
            if len(buf) == 6:
                f.write(' '.join(buf) + '\n'); buf = []
        if buf:
            f.write(' '.join(buf) + '\n')
    return v

for m in MOLS:
    dst = os.path.join(OUT, m['id'] + '.cube')
    if os.path.exists(dst):
        print('skip', m['id'], flush=True); continue
    t0 = time.time()
    mol = gto.M(atom=[[a[0], (a[1], a[2], a[3])] for a in m['atoms']],
                basis='6-31g*', charge=0, spin=0, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 200; mf.kernel()
    if not mf.converged:
        print(m['id'], 'plain SCF failed, retrying with newton', flush=True)
        mf = mf.newton(); mf.kernel()
    tmp = '/tmp/%s_raw.cube' % m['id']
    cubegen.density(mol, tmp, mf.make_rdm1(), nx=N, ny=N, nz=N)
    v = compact(tmp, dst)
    frac = float((v > 0.002).mean())
    print('%s  nelec=%d conv=%s E=%.4f  %.0fs  raw=%dKB out=%dKB  vox>0.002=%.1f%%'
          % (m['id'], mol.nelectron, mf.converged, mf.e_tot, time.time() - t0,
             os.path.getsize(tmp) // 1024, os.path.getsize(dst) // 1024, 100 * frac), flush=True)
    os.remove(tmp)
print('ALL DONE', flush=True)
