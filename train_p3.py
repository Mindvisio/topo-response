import os, argparse, torch
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule
from schnetpack.representation import PaiNN
from schnetpack.atomistic import DipoleMoment, PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.task import AtomisticTask, ModelOutput
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from train_dipole_tda import FlatMSE, FlatMAE, AddZPH, TDACondition, PaiNNWithTDA, make_zph_transform
NBASIS=128; DB='cache/squirl.db'
ZPH={'tda':'cache/zph.npy','shuffled':'cache/zph_shuffled.npy','random':'cache/zph_random.npy','density':'cache/zph_density.npy'}
ap=argparse.ArgumentParser()
ap.add_argument('--split', default='topology_ood')
ap.add_argument('--cond', default='tda', choices=['none','tda','shuffled','random','density'])
ap.add_argument('--cutoff', type=float, default=5.0)
ap.add_argument('--cache', default='cache/nbh_cache')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--tag', default=None)
ap.add_argument('--seed', type=int, default=0)
a=ap.parse_args()
import pytorch_lightning as _pl; _pl.seed_everything(a.seed, workers=True)
tag=a.tag or ('%s_%s_s%d'%(a.split,a.cond,a.seed)+('' if a.cutoff==5.0 else '_c%g'%a.cutoff))
use_cond=a.cond!='none'
tfs=[trn.SubtractCenterOfGeometry(), trn.CachedNeighborList(cache_path=a.cache, neighbor_list=trn.ASENeighborList(cutoff=a.cutoff), keep_cache=True), trn.CastTo32()]
if use_cond: tfs.append(make_zph_transform(a.cond, 'cache/split_%s.npz'%a.split))
dm=AtomsDataModule(DB, batch_size=128, split_file='cache/split_%s.npz'%a.split, load_properties=['dipole_moment'], transforms=tfs, num_workers=6, pin_memory=True)
painn=PaiNN(n_atom_basis=NBASIS, n_interactions=3, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=a.cutoff), cutoff_fn=spk.nn.CosineCutoff(a.cutoff))
rep=PaiNNWithTDA(painn, TDACondition()) if use_cond else painn
model=NeuralNetworkPotential(representation=rep, input_modules=[PairwiseDistances()], output_modules=[DipoleMoment(n_in=NBASIS, dipole_key='dipole_moment', use_vector_representation=True)])
output=ModelOutput(name='dipole_moment', loss_fn=FlatMSE(), loss_weight=1.0, metrics={'MAE': FlatMAE()})
task=AtomisticTask(model, outputs=[output], optimizer_cls=torch.optim.AdamW, optimizer_args={'lr':5e-4},
  scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args={'mode':'min','factor':0.7,'patience':5}, scheduler_monitor='val_loss')
d='ckpt_%s'%tag; os.makedirs(d, exist_ok=True)
ck=ModelCheckpoint(dirpath=d, filename='best-{epoch:02d}-{val_loss:.4f}', save_top_k=1, monitor='val_loss', mode='min', save_last=True, every_n_epochs=1)
trainer=pl.Trainer(accelerator='gpu', devices=1, max_epochs=a.epochs, callbacks=[ck, LearningRateMonitor()], default_root_dir='runs_%s'%tag, gradient_clip_val=10.0, log_every_n_steps=100)
resume='%s/last.ckpt'%d if os.path.exists('%s/last.ckpt'%d) else None
print('P3 RUN tag=%s split=%s cond=%s cutoff=%g RESUME=%s'%(tag,a.split,a.cond,a.cutoff,resume), flush=True)
trainer.fit(task, datamodule=dm, ckpt_path=resume)
print('P3 DONE tag=%s'%tag, flush=True)
