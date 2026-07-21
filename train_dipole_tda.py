import json, os, sys, numpy as np, torch, torchmetrics
import torch.nn as nn
import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.properties as P
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
DB='cache/squirl.db'; CUTOFF=5.0; NBASIS=128; ZPH_DIM=130
class FlatMSE(nn.Module):
    def forward(self, pred, target): return torch.nn.functional.mse_loss(pred.reshape(-1), target.reshape(-1))
class FlatMAE(torchmetrics.MeanAbsoluteError):
    def update(self, pred, target): super().update(pred.reshape(-1), target.reshape(-1))
class AddZPH(trn.Transform):
    # TRAIN-ONLY standardized TDA features by db index. shuffle_within: permute features
    # WITHIN each split partition (control keeping per-partition distribution, breaking alignment).
    is_preprocessor=True; is_postprocessor=False
    def __init__(self, zph_path, split_file, shuffle_within=False):
        super().__init__()
        z=np.load(zph_path).astype(np.float32); sp=np.load(split_file); tr=sp['train_idx']
        mean=z[tr].mean(0,keepdims=True); std=np.maximum(z[tr].std(0,keepdims=True), 1e-2)
        z=np.clip((z-mean)/std, -10.0, 10.0)  # floor std + clip: near-constant Betti bins had std~0 -> outliers blew up to 1e6
        if shuffle_within:
            rng=np.random.default_rng(0)
            for part in ('train_idx','val_idx','test_idx'):
                ix=sp[part]; z[ix]=z[ix][rng.permutation(len(ix))]
        self.register_buffer('zph', torch.tensor(z, dtype=torch.float32))
    def forward(self, inputs):
        inputs['zph']=self.zph[inputs[P.idx]]; return inputs
def make_zph_transform(cond, split_file):
    if cond=='shuffled': return AddZPH('cache/zph.npy', split_file, shuffle_within=True)
    path={'tda':'cache/zph.npy','random':'cache/zph_random.npy','elem4d':'cache/zph_elem4d.npy'}[cond]
    return AddZPH(path, split_file)
class TDACondition(nn.Module):
    # irrep-preserving FiLM, IDENTITY-INIT (zero last layer): at init scale=shift=gate=0 -> output == baseline.
    # invariant (1+.) x equivariant vector = equivariant -> exact E(3) preserved.
    def __init__(self, n_atom_basis=NBASIS, zph_dim=ZPH_DIM):
        super().__init__(); self.n=n_atom_basis
        self.net=nn.Sequential(nn.Linear(zph_dim,128), nn.SiLU(), nn.Linear(128,3*n_atom_basis))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, inputs):
        s=inputs['scalar_representation']; v=inputs['vector_representation']
        p=self.net(inputs['zph'])[inputs[P.idx_m]]
        sc,sh,g=p[:,:self.n],p[:,self.n:2*self.n],p[:,2*self.n:]
        inputs['scalar_representation']=s*(1.0+sc)+sh
        inputs['vector_representation']=v*(1.0+g).unsqueeze(1)
        return inputs
class PaiNNWithTDA(nn.Module):
    def __init__(self, painn, cond):
        super().__init__(); self.painn=painn; self.cond=cond
    def forward(self, inputs): return self.cond(self.painn(inputs))
def build(use_tda, split_file='cache/split_random.npz'):
    tfs=[trn.SubtractCenterOfGeometry(), trn.CachedNeighborList(cache_path='cache/nbh_cache_cut%g'%CUTOFF, neighbor_list=trn.ASENeighborList(cutoff=CUTOFF), keep_cache=True), trn.CastTo32()]
    if use_tda: tfs.append(AddZPH('cache/zph.npy', split_file))
    dm=AtomsDataModule(DB, batch_size=128, split_file=split_file, load_properties=['dipole_moment'], transforms=tfs, num_workers=6, pin_memory=True)
    painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF), cutoff_fn=spk.nn.CosineCutoff(CUTOFF))
    rep=PaiNNWithTDA(painn, TDACondition()) if use_tda else painn
    model=NeuralNetworkPotential(representation=rep, input_modules=[PairwiseDistances()], output_modules=[DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)])
    return dm, model
if __name__=='__main__':
    use_tda='--baseline' not in sys.argv; tag='tda' if use_tda else 'baseline'
    dm,model=build(use_tda)
    output=ModelOutput(name='dipole_moment', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE': FlatMAE()})
    task=AtomisticTask(model, outputs=[output], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4},
      scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args={'mode':'min','factor':0.7,'patience':5}, scheduler_monitor='val_loss')
    d='ckpt_%s'%tag; os.makedirs(d, exist_ok=True)
    ck=ModelCheckpoint(dirpath=d, filename='best-{epoch:02d}-{val_loss:.4f}', save_top_k=1, monitor='val_loss', mode='min', save_last=True, every_n_epochs=1)
    trainer=pl.Trainer(accelerator='gpu', devices=1, max_epochs=60, callbacks=[ck, LearningRateMonitor()], default_root_dir='runs_%s'%tag, gradient_clip_val=10.0, log_every_n_steps=100)
    resume='%s/last.ckpt'%d if os.path.exists('%s/last.ckpt'%d) else None
    print('TDA=%s tag=%s RESUME=%s'%(use_tda,tag,resume), flush=True)
    trainer.fit(task, datamodule=dm, ckpt_path=resume)
    print('DONE tag=%s'%tag, flush=True)
