import torch
import train_dipole_tda as T
import schnetpack.properties as P
print('building TDA model + datamodule...', flush=True)
dm, model = T.build(use_tda=True); dm.setup(); model.eval()
batch = next(iter(dm.train_dataloader()))
with torch.no_grad(): mu0 = model(batch)['dipole_moment']
print('FUNCTIONAL OK | dipole shape', tuple(mu0.shape), '| zph in batch:', 'zph' in batch, '| n_mol', int(batch[P.idx].shape[0]), flush=True)
try:
    from scipy.spatial.transform import Rotation
    Rot = torch.tensor(Rotation.random(random_state=0).as_matrix(), dtype=mu0.dtype)
except Exception:
    import math; th=0.7; Rot=torch.tensor([[math.cos(th),-math.sin(th),0.],[math.sin(th),math.cos(th),0.],[0.,0.,1.]],dtype=mu0.dtype)
b2=dict(batch); b2[P.R]=batch[P.R] @ Rot.T
with torch.no_grad(): mu1 = model(b2)['dipole_moment']
err = (mu1 - mu0 @ Rot.T).norm() / (mu0.norm()+1e-8)
print('EQUIVARIANCE err = %.3e  (want <1e-4)'%err.item(), flush=True)
print('SMOKE_RESULT:', 'PASS' if err.item()<1e-4 else 'FAIL-equivariance', flush=True)
