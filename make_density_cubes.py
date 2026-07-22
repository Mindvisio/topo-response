"""Electron-density cubes + ESP-fitted charges for the six viewer molecules.

RHF/6-31G* at the SQuIRL geometry. The cube box is padded generously so the
0.002 e/a0^3 isosurface closes instead of being clipped by the grid boundary.
Charges are fitted to reproduce the exact QM electrostatic potential on a
Merz-Kollman style shell outside the van der Waals surface, so the viewer can
rebuild the potential (for surface colouring) from a handful of numbers.
"""
import json, os, re, time
import numpy as np
from pyscf import df, gto, lib, scf
from pyscf.tools import cubegen

RESOLUTION = 0.5    # bohr between grid points
MARGIN = 6.5        # bohr of padding around the molecule
ISOVAL = 0.002      # the isovalue the viewer draws
FLOOR = 5e-4        # densities below this are written as 0 (far below ISOVAL)
BOHR = 1.8897259886
VDW = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'S': 1.80, 'Cl': 1.75}


def compact(src, dst):
    """Rewrite a cube with small values zeroed; returns the density array."""
    lines = open(src).read().splitlines()
    nat = abs(int(lines[2].split()[0]))
    head = lines[:6 + nat]
    dims = [int(lines[3 + k].split()[0]) for k in range(3)]
    v = np.array(' '.join(lines[6 + nat:]).split(), dtype=float)
    v[np.abs(v) < FLOOR] = 0.0
    with open(dst, 'w') as f:
        f.write('\n'.join(head) + '\n')
        buf = []
        for x in v:
            buf.append('0' if x == 0.0 else '%.4e' % x)
            if len(buf) == 6:
                f.write(' '.join(buf) + '\n'); buf = []
        if buf:
            f.write(' '.join(buf) + '\n')
    return v.reshape(dims)


def face_max(v):
    return max(v[0].max(), v[-1].max(), v[:, 0].max(), v[:, -1].max(),
               v[:, :, 0].max(), v[:, :, -1].max())


def shell_points(symbols, pos_bohr, scales=(1.4, 1.6, 1.8, 2.0), per_sphere=180):
    """Merz-Kollman style sampling: points between 1.4 and 2.0 van der Waals radii."""
    idx = np.arange(per_sphere) + 0.5
    phi = np.arccos(1 - 2 * idx / per_sphere)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    rvdw = np.array([VDW.get(s, 1.7) * BOHR for s in symbols])
    keep = []
    for s in scales:
        for a in range(len(symbols)):
            pts = pos_bohr[a] + unit * (s * rvdw[a])
            d = np.linalg.norm(pts[:, None, :] - pos_bohr[None, :, :], axis=2)
            inside = (d < 1.4 * rvdw[None, :]).any(axis=1)
            outside = (d > 2.0 * rvdw[None, :]).all(axis=1)
            keep.append(pts[~inside & ~outside])
    return np.vstack(keep)


def exact_esp(mol, dm, coords):
    """QM electrostatic potential (a.u.) at arbitrary points, batched over memory."""
    v = np.zeros(len(coords))
    for i in range(mol.natm):
        v += mol.atom_charge(i) / np.linalg.norm(coords - mol.atom_coord(i), axis=1)
    ele = []
    for p0, p1 in lib.prange(0, len(coords), 400):
        fake = gto.fakemol_for_charges(coords[p0:p1])
        ele.append(np.einsum('ijp,ij->p', df.incore.aux_e2(mol, fake), dm))
    return v - np.concatenate(ele)


def fit_charges(coords, v, pos_bohr, qtot=0.0):
    """Least-squares charges reproducing v, constrained to the total charge."""
    A = 1.0 / np.linalg.norm(coords[:, None, :] - pos_bohr[None, :, :], axis=2)
    n = len(pos_bohr)
    K = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)
    K[:n, :n] = 2 * A.T @ A; K[:n, n] = 1.0; K[n, :n] = 1.0
    b[:n] = 2 * A.T @ v; b[n] = qtot
    q = np.linalg.solve(K, b)[:n]
    resid = A @ q - v
    return q, float(np.sqrt((resid ** 2).mean())), float(np.sqrt((resid ** 2).mean() / (v ** 2).mean()))


def main():
    html = open('index.html').read()
    mols = json.loads(re.search(r'const MOLS = (\[.*?\]);', html, re.S).group(1))
    os.makedirs('dens', exist_ok=True)
    charges = {}
    for m in mols:
        t0 = time.time()
        symbols = [a[0] for a in m['atoms']]
        mol = gto.M(atom=[[a[0], (a[1], a[2], a[3])] for a in m['atoms']],
                    basis='6-31g*', charge=0, spin=0, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol); mf.max_cycle = 200; mf.kernel()
        if not mf.converged:
            mf = mf.newton(); mf.kernel()
        dm = mf.make_rdm1()

        tmp = '/tmp/%s_raw.cube' % m['id']
        cubegen.density(mol, tmp, dm, resolution=RESOLUTION, margin=MARGIN)
        dst = 'dens/%s.cube' % m['id']
        v = compact(tmp, dst)
        os.remove(tmp)
        edge = face_max(v)

        pos = mol.atom_coords()
        pts = shell_points(symbols, pos)
        q, rms, rrms = fit_charges(pts, exact_esp(mol, dm, pts), pos)
        charges[m['id']] = [round(float(x), 4) for x in q]

        print('%-8s %s grid=%s  edge_rho=%.5f %s  esp_pts=%d rms=%.4f (%.1f%%)  q=[%.2f..%.2f]  %.0fs  %dKB'
              % (m['id'], 'conv' if mf.converged else 'UNCONVERGED', 'x'.join(str(d) for d in v.shape),
                 edge, 'OK' if edge < ISOVAL else 'STILL CLIPPED', len(pts), rms, 100 * rrms,
                 min(q), max(q), time.time() - t0, os.path.getsize(dst) // 1024), flush=True)

    json.dump(charges, open('dens/charges.json', 'w'))
    print('wrote dens/charges.json for %d molecules' % len(charges), flush=True)


if __name__ == '__main__':
    main()
