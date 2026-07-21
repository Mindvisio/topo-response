import numpy as np, torch
import torch.nn as nn
import train_dipole_tda as T
from train_dipole_tda import AddZPH
SPL='cache/split_topology_ood.npz'
# (i) train-only standardization
az=AddZPH('cache/zph.npy',SPL); tr=np.load(SPL)['train_idx']; ztr=az.zph[tr].numpy()
zraw=np.load('cache/zph.npy')
print('(i) standardized z_PH TRAIN: mean=%.2e std=%.3f | raw median-norm=%.1f -> std median-norm=%.2f'%(ztr.mean(),ztr.std(),np.median(np.linalg.norm(zraw,axis=1)),np.median(np.linalg.norm(az.zph.numpy(),axis=1))),flush=True)
# (ii) identity-init: does conditioning change output at init?
dm,model=T.build(use_tda=True,split_file=SPL); dm.setup(); batch=next(iter(dm.train_dataloader())); model.eval()
class Ident(nn.Module):
    def forward(self,x): return x
with torch.no_grad(): oc=model({k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()})['dipole_moment'].clone()
orig=model.representation.cond; model.representation.cond=Ident()
with torch.no_grad(): ob=model({k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()})['dipole_moment'].clone()
model.representation.cond=orig
print('(ii) identity-init: max|out_TDA - out_baseline| at init = %.2e (want ~0)'%(oc-ob).abs().max().item(),flush=True)
# (iii) param counts
_,mt=T.build(use_tda=True,split_file=SPL); _,mb=T.build(use_tda=False,split_file=SPL)
pt=sum(p.numel() for p in mt.parameters()); pb=sum(p.numel() for p in mb.parameters())
print('(iii) params: baseline=%d TDA=%d (delta=%d=conditioning MLP; shuffled/random controls share TDA count -> matched-signal test)'%(pb,pt,pt-pb),flush=True)
print('P1_3_VERIFY DONE',flush=True)
