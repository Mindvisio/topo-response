"""E(3) equivariance check for both response heads.

Why the trained cases matter: TDACondition is identity-initialised (its last layer
starts at zero), so on a freshly built model the conditioning path emits exactly
zero and the test passes without ever exercising it. The trained cases load real
weights, report how strong the learned FiLM modulation actually is, and only then
check equivariance -- so the claim covers the model that was actually evaluated.
"""
import glob
import torch
import schnetpack as spk, schnetpack.transform as trn, schnetpack.properties as P
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, Polarizability, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
from train_dipole_tda import FlatMSE, FlatMAE, TDACondition, PaiNNWithTDA, make_zph_transform
from scipy.spatial.transform import Rotation

NBASIS = 128
CUTOFF = 5.0
TOL = 1e-4          # float32 forward passes land ~1e-6; 1e-4 catches real breakage
BATCH_SEED = 0      # fixes which 24 molecules are audited, so runs are comparable
SPLIT = 'cache/split_topology_ood.npz'
KEY = {'dipole': 'dipole_moment', 'polar': 'polarizability'}


def build(head, cond):
    painn = PaiNN(n_atom_basis=NBASIS, n_interactions=3,
                  radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF),
                  cutoff_fn=spk.nn.CosineCutoff(CUTOFF))
    rep = PaiNNWithTDA(painn, TDACondition()) if cond else painn
    out = (DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)
           if head == 'dipole' else Polarizability(n_in=NBASIS, polarizability_key='polarizability'))
    return NeuralNetworkPotential(representation=rep, input_modules=[PairwiseDistances()],
                                  output_modules=[out])


def load_trained(head, cond, ckpt):
    model = build(head, cond)
    task = AtomisticTask(model,
                         outputs=[ModelOutput(name=KEY[head], loss_fn=FlatMSE(), loss_weight=1.0,
                                              metrics={'MAE': FlatMAE()})],
                         optimizer_cls=torch.optim.AdamW, optimizer_args={'lr': 5e-4})
    task.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'], strict=True)
    return task.model


def get_batch(head, cond):
    tfs = [trn.CachedNeighborList(cache_path='cache/nbh_cache_cut%g' % CUTOFF,
                                  neighbor_list=trn.ASENeighborList(cutoff=CUTOFF), keep_cache=True),
           trn.CastTo32()]
    if cond:
        tfs.append(make_zph_transform('tda', SPLIT))
    dm = AtomsDataModule('cache/squirl.db', batch_size=24, split_file=SPLIT,
                         load_properties=[KEY[head]], transforms=tfs, num_workers=2)
    dm.setup()
    # the train loader shuffles, so without a fixed seed each run audits a different
    # batch and the reported conditioning strengths drift between runs
    torch.manual_seed(BATCH_SEED)
    return next(iter(dm.train_dataloader()))


def center(b):
    R = b[P.R]; im = b[P.idx_m]; n = int(im.max()) + 1
    s = torch.zeros(n, 3).index_add_(0, im, R)
    c = torch.zeros(n, 1).index_add_(0, im, torch.ones(len(R), 1))
    b2 = dict(b); b2[P.R] = R - (s / c)[im]
    return b2


def forward(model, b, key):
    with torch.no_grad():
        return model(center(dict(b)))[key].detach()


def rot(seed, det=1):
    M = torch.tensor(Rotation.random(random_state=seed).as_matrix(), dtype=torch.float32)
    return M @ torch.diag(torch.tensor([1., 1., -1.])) if det < 0 else M


def film_strength(model, b):
    """RMS of the scale/shift/gate the conditioner emits -- exactly 0 at identity init."""
    cond = model.representation.cond
    grab = {}
    h = cond.net.register_forward_hook(lambda m, i, o: grab.setdefault('p', o.detach()))
    with torch.no_grad():
        model(center(dict(b)))
    h.remove()
    p = grab['p']; n = cond.n
    rms = lambda x: float(x.pow(2).mean().sqrt())
    return rms(p[:, :n]), rms(p[:, n:2 * n]), rms(p[:, 2 * n:])


def check(head, cond, ckpt=None):
    key = KEY[head]
    model = (load_trained(head, cond, ckpt) if ckpt else build(head, cond)).eval()
    b = get_batch(head, cond)
    if cond:
        sc, sh, g = film_strength(model, b)
        state = 'INERT (conditioning not exercised)' if max(sc, sh, g) < 1e-9 else 'active'
        print('    conditioning RMS  scale %.4f  shift %.4f  gate %.4f  -> %s' % (sc, sh, g, state))
    o0 = forward(model, b, key)
    Rr, Rf, t = rot(0), rot(1, -1), torch.tensor([10., -4., 2.])
    bR = dict(b); bR[P.R] = b[P.R] @ Rr.T
    bF = dict(b); bF[P.R] = b[P.R] @ Rf.T
    bT = dict(b); bT[P.R] = b[P.R] + t
    oR, oF, oT = forward(model, bR, key), forward(model, bF, key), forward(model, bT, key)
    if head == 'dipole':
        eR, eF, eT = o0 @ Rr.T, o0 @ Rf.T, o0
    else:
        A = o0.reshape(-1, 3, 3)
        eR = (Rr @ A @ Rr.T).reshape(-1, 9)
        eF = (Rf @ A @ Rf.T).reshape(-1, 9)
        eT = o0.reshape(-1, 9)
        oR, oF, oT = oR.reshape(-1, 9), oF.reshape(-1, 9), oT.reshape(-1, 9)
    rel = lambda a, e: float((a - e).norm() / (e.norm() + 1e-8))
    errs = (rel(oR, eR), rel(oF, eF), rel(oT, eT))
    print('    %-6s  rotation %.2e | reflection %.2e | translation %.2e'
          % (head, errs[0], errs[1], errs[2]))
    # a real gate, not a printout: float32 round-off is ~1e-6, so anything above
    # TOL means the symmetry is genuinely broken
    assert max(errs) < TOL, ('E(3) equivariance violated for %s: rotation/reflection/translation '
                             '= %.2e/%.2e/%.2e (tolerance %.0e)' % ((head,) + errs + (TOL,)))


CASES = [
    ('fresh model, identity-init conditioning', 'dipole', True, None),
    ('trained baseline', 'dipole', False, 'ckpt_topology_ood_none_s0'),
    ('trained TDA conditioning', 'dipole', True, 'ckpt_topology_ood_tda_s0'),
    ('fresh model, identity-init conditioning', 'polar', True, None),
    ('trained baseline', 'polar', False, 'ckpt_polar_topology_ood_none_s0'),
    ('trained TDA conditioning', 'polar', True, 'ckpt_polar_topology_ood_tda_s0'),
]

if __name__ == '__main__':
    for label, head, cond, d in CASES:
        ck = sorted(glob.glob(d + '/best-*.ckpt'))[0] if d else None
        print('%s [%s]%s' % (label, head, '' if ck is None else '  ' + ck), flush=True)
        check(head, cond, ck)
    print('E3_TEST PASS (all cases within %.0e)' % TOL, flush=True)
