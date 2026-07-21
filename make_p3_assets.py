import json, numpy as np
idx=json.load(open('cache/index.json')); ids=[r['id'] for r in idx]; pos={m:k for k,m in enumerate(ids)}
sp=json.load(open('cache/splits.json'))
ood=sp['topology_ood']
np.savez('cache/split_topology_ood.npz',
  train_idx=np.array([pos[i] for i in ood['train']]),
  val_idx=np.array([pos[i] for i in ood['val']]),
  test_idx=np.array([pos[i] for i in ood['test']]))
print('OOD split npz: train %d val %d test %d'%(len(ood['train']),len(ood['val']),len(ood['test'])))
Z=np.load('cache/zph.npy'); rng=np.random.default_rng(0)
perm=rng.permutation(Z.shape[0]); np.save('cache/zph_shuffled.npy', Z[perm].astype(np.float32))
mu=Z.mean(0,keepdims=True); sd=Z.std(0,keepdims=True)+1e-8
np.save('cache/zph_random.npy', (rng.standard_normal(Z.shape).astype(np.float32)*sd+mu).astype(np.float32))
print('controls written: zph_shuffled.npy (perm rows), zph_random.npy (matched mean/std)')
