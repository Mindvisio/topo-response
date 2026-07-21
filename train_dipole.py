import json, os, numpy as np, torch, torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
class FlatMSE(torch.nn.Module):
    def forward(self, pred, target): return torch.nn.functional.mse_loss(pred.reshape(-1), target.reshape(-1))
class FlatMAE(torchmetrics.MeanAbsoluteError):
    def update(self, pred, target): super().update(pred.reshape(-1), target.reshape(-1))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
DB='cache/squirl.db'; CUTOFF=5.0; NBASIS=128
idx=json.load(open('cache/index.json')); ids=[r['id'] for r in idx]; pos={m:k for k,m in enumerate(ids)}
sp=json.load(open('cache/splits.json'))['random']
np.savez('cache/split_random.npz', train_idx=np.array([pos[i] for i in sp['train']]), val_idx=np.array([pos[i] for i in sp['val']]), test_idx=np.array([pos[i] for i in sp['test']]))
dm=AtomsDataModule(DB, batch_size=128, split_file='cache/split_random.npz', load_properties=['dipole_moment'],
  transforms=[trn.SubtractCenterOfGeometry(), trn.CachedNeighborList(cache_path='cache/nbh_cache_cut%g'%CUTOFF, neighbor_list=trn.ASENeighborList(cutoff=CUTOFF), keep_cache=True), trn.CastTo32()],
  num_workers=6, pin_memory=True)
painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF), cutoff_fn=spk.nn.CosineCutoff(CUTOFF))
dipole=DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)
model=NeuralNetworkPotential(representation=painn, input_modules=[PairwiseDistances()], output_modules=[dipole])
output=ModelOutput(name='dipole_moment', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE': FlatMAE()})
task=AtomisticTask(model, outputs=[output], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4},
  scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args={'mode':'min','factor':0.7,'patience':5}, scheduler_monitor='val_loss')
os.makedirs('ckpt', exist_ok=True)
ck=ModelCheckpoint(dirpath='ckpt', filename='best-{epoch:02d}-{val_loss:.4f}', save_top_k=1, monitor='val_loss', mode='min', save_last=True, every_n_epochs=1)
trainer=pl.Trainer(accelerator='gpu', devices=1, max_epochs=60, callbacks=[ck, LearningRateMonitor()], default_root_dir='runs', gradient_clip_val=10.0, log_every_n_steps=100)
resume='ckpt/last.ckpt' if os.path.exists('ckpt/last.ckpt') else None
print('RESUME from', resume, flush=True)
trainer.fit(task, datamodule=dm, ckpt_path=resume)
print('TRAINING DONE', flush=True)
