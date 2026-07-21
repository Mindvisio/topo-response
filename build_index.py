import json, time
from data_squirl import build_index, make_splits
t0=time.time(); idx=build_index(); print('indexed %d in %.1fs'%(len(idx),time.time()-t0))
json.dump(idx, open('cache/index.json','w'))
sp=make_splits(idx,seed=0); json.dump(sp, open('cache/splits.json','w'))
for n,s in sp.items(): print('  %-13s train=%d val=%d test=%d'%(n,len(s['train']),len(s['val']),len(s['test'])))
