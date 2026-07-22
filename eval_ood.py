import argparse, torch, torch.nn.functional as F
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
from train_dipole_tda import FlatMSE, FlatMAE, AddZPH, TDACondition, PaiNNWithTDA, make_zph_transform
NBASIS=128; DB='cache/squirl.db'; AU2D=2.541746
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--split', default='topology_ood')
ap.add_argument('--cond', default='none', choices=['none','tda','shuffled','random','elem4d'])
ap.add_argument('--cutoff', type=float, default=5.0); ap.add_argument('--cache', default='cache/nbh_cache')
a=ap.parse_args(); use_cond=a.cond!='none'
tfs=[trn.SubtractCenterOfGeometry(), trn.CachedNeighborList(cache_path='%s_cut%g'%(a.cache,a.cutoff), neighbor_list=trn.ASENeighborList(cutoff=a.cutoff), keep_cache=True), trn.CastTo32()]
if use_cond: tfs.append(make_zph_transform(a.cond, 'cache/split_%s.npz'%a.split))
dm=AtomsDataModule(DB, batch_size=256, split_file='cache/split_%s.npz'%a.split, load_properties=['dipole_moment'], transforms=tfs, num_workers=6); dm.setup()
painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=a.cutoff), cutoff_fn=spk.nn.CosineCutoff(a.cutoff))
rep=PaiNNWithTDA(painn, TDACondition()) if use_cond else painn
model=NeuralNetworkPotential(representation=rep, input_modules=[PairwiseDistances()], output_modules=[DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)])
output=ModelOutput(name='dipole_moment', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE':FlatMAE()})
task=AtomisticTask(model, outputs=[output], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4})
sd=torch.load(a.ckpt, map_location='cpu')['state_dict']
# strict: a silently half-loaded model would still produce numbers, and they would look like results
task.load_state_dict(sd, strict=True)
print('loaded %d tensors from %s (strict)'%(len(sd), a.ckpt))
dev='cuda' if torch.cuda.is_available() else 'cpu'; task.to(dev); task.eval()
Ps=[]; Ts=[]
for batch in dm.test_dataloader():
    batch={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
    tgt=batch['dipole_moment'].detach().clone().cpu().reshape(-1,3)
    with torch.no_grad(): out=task.model(batch)
    Ps.append(out['dipole_moment'].detach().cpu()); Ts.append(tgt)
P=torch.cat(Ps)*AU2D; T=torch.cat(Ts)*AU2D  # a.u. -> Debye
mse=((P-T)**2).mean().item()
vecMAE=(P-T).norm(dim=1).mean().item(); compMAE=(P-T).abs().mean().item()
magMAE=(P.norm(dim=1)-T.norm(dim=1)).abs().mean().item()
thr=0.1; m=T.norm(dim=1)>thr
cos=F.cosine_similarity(P[m],T[m],dim=1).clamp(-1,1); ang=torch.rad2deg(torch.arccos(cos)).mean().item()
print('OOD_TEST cond=%s N=%d | MSE %.5f D2 | vectorMAE %.4f D | compMAE %.4f D | magMAE %.4f D | angErr %.2f deg (|mu|>%.1fD n=%d)'%(a.cond,len(P),mse,vecMAE,compMAE,magMAE,ang,thr,int(m.sum())))
