"""Prove the residual corrections are E(3)-equivariant BEFORE fitting anything.

The probe predicts INVARIANT coefficients from an invariant descriptor z_PH and
assembles the correction from equivariant tensors built out of the geometry:

  dipole  primary   : mu_corr = mu_base + a * unit(mu_base)
          secondary : mu_corr = (1+a) mu_base + b (S mu_base) + c (S^2 mu_base)
  polar   primary   : A_corr  = A_base + a I + b Q          (Q = deviatoric A_base)
          secondary : A_corr  = A_base + a I + b Q + c S + d (S Q + Q S)/2

where S is the gyration tensor of the centred coordinates. Under an orthogonal R
the coefficients are unchanged (they are functions of invariants) while every
tensor carries its own indices, so mu_corr -> R mu_corr and A_corr -> R A_corr R^T.
This file checks exactly that, with the coefficients held fixed across the transform.
"""
import numpy as np
from scipy.spatial.transform import Rotation

rng = np.random.default_rng(0)


def gyration(Xc):
    return (Xc.T @ Xc) / len(Xc)


def dipole_primary(mu, a):
    n = np.linalg.norm(mu)
    u = mu / n if n > 1e-9 else np.zeros(3)
    return mu + a * u


def dipole_secondary(mu, S, a, b, c):
    return (1 + a) * mu + b * (S @ mu) + c * (S @ S @ mu)


def deviatoric(A):
    return A - np.trace(A) / 3.0 * np.eye(3)


def polar_primary(A, a, b):
    return A + a * np.eye(3) + b * deviatoric(A)


def polar_secondary(A, S, a, b, c, d):
    Q = deviatoric(A)
    return A + a * np.eye(3) + b * Q + c * S + d * (S @ Q + Q @ S) / 2.0


def rand_geometry(n):
    X = rng.normal(size=(n, 3))
    return X - X.mean(0)


def run():
    worst = {}
    for trial in range(200):
        R = Rotation.random(random_state=trial).as_matrix()
        if trial % 2:                                   # exercise reflections too
            R = R @ np.diag([1.0, 1.0, -1.0])
        Xc = rand_geometry(int(rng.integers(5, 25)))
        S = gyration(Xc)
        Sr = gyration(Xc @ R.T)                          # = R S R^T numerically
        mu = rng.normal(size=3)
        A = rng.normal(size=(3, 3)); A = 0.5 * (A + A.T)
        a, b, c, d = rng.normal(size=4)

        checks = {
            'dipole primary':   (dipole_primary(R @ mu, a),           R @ dipole_primary(mu, a)),
            'dipole secondary': (dipole_secondary(R @ mu, Sr, a, b, c), R @ dipole_secondary(mu, S, a, b, c)),
            'polar primary':    (polar_primary(R @ A @ R.T, a, b),    R @ polar_primary(A, a, b) @ R.T),
            'polar secondary':  (polar_secondary(R @ A @ R.T, Sr, a, b, c, d),
                                 R @ polar_secondary(A, S, a, b, c, d) @ R.T),
        }
        for name, (lhs, rhs) in checks.items():
            e = np.linalg.norm(lhs - rhs) / (np.linalg.norm(rhs) + 1e-12)
            worst[name] = max(worst.get(name, 0.0), e)

    print('max relative error over 200 random rotations+reflections:')
    ok = True
    for name, e in worst.items():
        flag = 'OK' if e < 1e-10 else 'FAIL'
        ok = ok and e < 1e-10
        print('  %-18s %.2e  %s' % (name, e, flag))
    print('PROBE_EQUIVARIANCE', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)
