import json, re, glob, numpy as np, torch
import schnetpack as spk, schnetpack.transform as trn
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
from schnetpack.interfaces import AtomsConverter
from train_dipole_tda import FlatMSE, FlatMAE
from ase import Atoms
NBASIS=128; CUT=5.0
html=open('index.html').read()
mols=json.loads(re.search(r'const MOLS = (\[.*?\]);', html, re.S).group(1))
painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=CUT), cutoff_fn=spk.nn.CosineCutoff(CUT))
model=NeuralNetworkPotential(representation=painn, input_modules=[PairwiseDistances()], output_modules=[DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)])
task=AtomisticTask(model, outputs=[ModelOutput(name='dipole_moment', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE':FlatMAE()})], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4})
ck=sorted(glob.glob('ckpt_topology_ood_none_s0/best-*.ckpt'))[0]
print('ckpt:', ck)
task.load_state_dict(torch.load(ck, map_location='cpu')['state_dict'], strict=True)
task.eval(); dev='cuda' if torch.cuda.is_available() else 'cpu'; task.to(dev)
conv=AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=CUT), transforms=[trn.SubtractCenterOfGeometry(), trn.CastTo32()], device=dev)
preds={}
for m in mols:
    at=Atoms(symbols=[a[0] for a in m['atoms']], positions=[[a[1],a[2],a[3]] for a in m['atoms']])
    inp=conv(at)
    with torch.no_grad(): out=task.model(inp)
    mu=out['dipole_moment'].detach().cpu().numpy().reshape(-1).tolist()
    preds[m['id']]=[round(float(x),4) for x in mu]
    print(m['id'],'true',[round(x,2) for x in m['dip']],'pred',[round(float(x),2) for x in mu])
json.dump(preds, open('viewer_preds.json','w'))
print('WROTE viewer_preds.json', len(preds))
