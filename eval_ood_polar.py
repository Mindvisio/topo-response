import argparse, torch
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import Polarizability, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
from train_dipole_tda import FlatMSE, FlatMAE, AddZPH, TDACondition, PaiNNWithTDA, make_zph_transform
NBASIS=128; DB='cache/squirl.db'
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--split', default='topology_ood')
ap.add_argument('--cond', default='none', choices=['none','tda','shuffled','random','density'])
ap.add_argument('--cutoff', type=float, default=5.0); ap.add_argument('--cache', default='cache/nbh_cache')
a=ap.parse_args(); use_cond=a.cond!='none'
tfs=[trn.SubtractCenterOfGeometry(), trn.CachedNeighborList(cache_path='%s_cut%g'%(a.cache,a.cutoff), neighbor_list=trn.ASENeighborList(cutoff=a.cutoff), keep_cache=True), trn.CastTo32()]
if use_cond: tfs.append(make_zph_transform(a.cond, 'cache/split_%s.npz'%a.split))
dm=AtomsDataModule(DB, batch_size=256, split_file='cache/split_%s.npz'%a.split, load_properties=['polarizability'], transforms=tfs, num_workers=6); dm.setup()
painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=a.cutoff), cutoff_fn=spk.nn.CosineCutoff(a.cutoff))
rep=PaiNNWithTDA(painn, TDACondition()) if use_cond else painn
model=NeuralNetworkPotential(representation=rep, input_modules=[PairwiseDistances()], output_modules=[Polarizability(n_in=NBASIS, polarizability_key='polarizability')])
output=ModelOutput(name='polarizability', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE':FlatMAE()})
task=AtomisticTask(model, outputs=[output], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4})
sd=torch.load(a.ckpt, map_location='cpu')['state_dict']; miss=task.load_state_dict(sd, strict=False)
print('loaded; missing=%d unexpected=%d'%(len(miss.missing_keys),len(miss.unexpected_keys)))
dev='cuda' if torch.cuda.is_available() else 'cpu'; task.to(dev); task.eval()
Ps=[]; Ts=[]
for batch in dm.test_dataloader():
    batch={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
    tgt=batch['polarizability'].detach().clone().cpu().reshape(-1,3,3)
    with torch.no_grad(): out=task.model(batch)
    Ps.append(out['polarizability'].detach().cpu().reshape(-1,3,3)); Ts.append(tgt)
P=torch.cat(Ps); T=torch.cat(Ts)
P=0.5*(P+P.transpose(1,2)); T=0.5*(T+T.transpose(1,2))
frob=(P-T).reshape(-1,9).norm(dim=1).mean().item(); elem=(P-T).abs().mean().item()
iso=(((P[:,0,0]+P[:,1,1]+P[:,2,2])-(T[:,0,0]+T[:,1,1]+T[:,2,2]))/3).abs().mean().item()
ep=torch.linalg.eigvalsh(P); et=torch.linalg.eigvalsh(T); eigMAE=(ep-et).abs().mean().item()
def aniso(l): return torch.sqrt(0.5*((l[:,0]-l[:,1])**2+(l[:,1]-l[:,2])**2+(l[:,2]-l[:,0])**2)+1e-12)
aniMAE=(aniso(ep)-aniso(et)).abs().mean().item()
vp=torch.linalg.eigh(P).eigenvectors[:,:,-1]; vt=torch.linalg.eigh(T).eigenvectors[:,:,-1]
cos=(vp*vt).sum(1).abs().clamp(0,1); pax=torch.rad2deg(torch.arccos(cos)).mean().item()
print('OOD_TEST_POLAR cond=%s N=%d | Frob %.4f | elem %.4f | iso %.4f | eigMAE %.4f | anisoMAE %.4f | paxErr %.2f deg (a.u.)'%(a.cond,len(P),frob,elem,iso,eigMAE,aniMAE,pax))
