import torch
import schnetpack as spk, schnetpack.transform as trn, schnetpack.properties as P
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, Polarizability, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from train_dipole_tda import AddZPH, TDACondition, PaiNNWithTDA
from scipy.spatial.transform import Rotation
NBASIS=128; CUTOFF=5.0
KEY={'dipole':'dipole_moment','polar':'polarizability'}
def build(head):
    painn=PaiNN(n_atom_basis=NBASIS,n_interactions=3,radial_basis=spk.nn.GaussianRBF(n_rbf=20,cutoff=CUTOFF),cutoff_fn=spk.nn.CosineCutoff(CUTOFF))
    rep=PaiNNWithTDA(painn,TDACondition())
    h=DipoleMoment(n_in=NBASIS,dipole_key='dipole_moment',use_vector_representation=True) if head=='dipole' else Polarizability(n_in=NBASIS,polarizability_key='polarizability')
    return NeuralNetworkPotential(representation=rep,input_modules=[PairwiseDistances()],output_modules=[h])
def get_batch(k):
    tfs=[trn.CachedNeighborList(cache_path='cache/nbh_cache',neighbor_list=trn.ASENeighborList(cutoff=CUTOFF),keep_cache=True),trn.CastTo32(),AddZPH('cache/zph.npy', 'cache/split_random.npz')]
    dm=AtomsDataModule('cache/squirl.db',batch_size=24,split_file='cache/split_random.npz',load_properties=[k],transforms=tfs,num_workers=2)
    dm.setup(); return next(iter(dm.train_dataloader()))
def center_batch(b):
    R=b[P.R]; im=b[P.idx_m]; n=int(im.max())+1
    s=torch.zeros(n,3).index_add_(0,im,R); c=torch.zeros(n,1).index_add_(0,im,torch.ones(len(R),1))
    b2=dict(b); b2[P.R]=R-(s/c)[im]; return b2
def Rof(seed,det=1):
    M=torch.tensor(Rotation.random(random_state=seed).as_matrix(),dtype=torch.float32)
    return M@torch.diag(torch.tensor([1.,1.,-1.])) if det<0 else M
def out(m,b,k):
    with torch.no_grad(): return m(center_batch(dict(b)))[k].detach()
def test(head):
    k=KEY[head]; m=build(head).eval(); b=get_batch(k); o0=out(m,b,k)
    Rr=Rof(0); Rf=Rof(1,-1); t=torch.tensor([10.,-4.,2.])
    bR=dict(b); bR[P.R]=b[P.R]@Rr.T; oR=out(m,bR,k)
    bF=dict(b); bF[P.R]=b[P.R]@Rf.T; oF=out(m,bF,k)
    bT=dict(b); bT[P.R]=b[P.R]+t;     oT=out(m,bT,k)
    if head=='dipole':
        eR=o0@Rr.T; eF=o0@Rf.T; eT=o0
    else:
        A=o0.reshape(-1,3,3); eR=(Rr@A@Rr.T).reshape(-1,9); eF=(Rf@A@Rf.T).reshape(-1,9); eT=o0.reshape(-1,9)
        oR=oR.reshape(-1,9); oF=oF.reshape(-1,9); oT=oT.reshape(-1,9)
    re=lambda a,e:((a-e).norm()/(e.norm()+1e-8)).item()
    print('%-6s | rotation %.2e | reflection %.2e | translation %.2e'%(head,re(oR,eR),re(oF,eF),re(oT,eT)),flush=True)
for h in ['dipole','polar']: test(h)
print('E3_TEST DONE',flush=True)
