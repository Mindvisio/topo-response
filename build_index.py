import json, time, numpy as np
from data_squirl import build_index, make_splits
t0 = time.time()
idx = build_index()
print('indexed %d molecules in %.1fs' % (len(idx), time.time() - t0))
json.dump(idx, open('cache/index.json', 'w'))
sp = make_splits(idx, seed=0)
json.dump(sp, open('cache/splits.json', 'w'))
nh = np.array([r['n_heavy'] for r in idx]); nr = np.array([r['n_rings'] for r in idx])
print('n_heavy  min/median/max = %d/%d/%d' % (nh.min(), int(np.median(nh)), nh.max()))
print('n_rings  dist:', {int(k): int(v) for k, v in zip(*np.unique(nr, return_counts=True))})
for name, s in sp.items():
    print('  %-14s train=%6d val=%5d test=%6d%s' % (
        name, len(s['train']), len(s['val']), len(s['test']),
        ('  thr_heavy=%.0f' % s['threshold_heavy']) if 'threshold_heavy' in s else ''))
