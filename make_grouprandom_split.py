import json, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
idx=json.load(open('cache/index.json')); ids=[r['id'] for r in idx]
keys=[]
for r in idx:
    smi=r.get('smiles',''); m=Chem.MolFromSmiles(smi) if smi else None
    keys.append(Chem.MolToSmiles(m) if m is not None else ('__'+r['id']))
keys=np.array(keys, dtype=object)
uniq=np.unique(keys); rng=np.random.default_rng(0); perm=rng.permutation(len(uniq)); n=len(uniq)
ntest=int(0.1*n); nval=int(0.1*n)
test_g=set(uniq[perm[:ntest]]); val_g=set(uniq[perm[ntest:ntest+nval]])
parts={'train':[],'val':[],'test':[]}
for i in range(len(keys)):
    k=keys[i]; p='test' if k in test_g else ('val' if k in val_g else 'train'); parts[p].append(i)
np.savez('cache/split_grouprandom.npz', train_idx=np.array(parts['train']), val_idx=np.array(parts['val']), test_idx=np.array(parts['test']))
print('grouprandom: train %d val %d test %d | %d unique graphs of %d records'%(len(parts['train']),len(parts['val']),len(parts['test']),n,len(keys)),flush=True)
tr=set(keys[parts['train']]); va=set(keys[parts['val']]); te=set(keys[parts['test']])
print('SELFCHECK overlap: train-val=%d train-test=%d val-test=%d (want 0,0,0)'%(len(tr&va),len(tr&te),len(va&te)),flush=True)
print('GROUPSPLIT DONE',flush=True)
