import numpy as np, h5py, json, warnings, sys
from multiprocessing import Pool
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
H5='/home/yc-user/data/squirl/SQuIRL_v1.0.h5'; BINS=64; MAXDIM=1; ZSCALE=0.5
NPROC=int(sys.argv[1]) if len(sys.argv)>1 else 3
GRID=np.linspace(0.0,1.0,BINS,dtype=np.float32)
_vr=_ent=None
def _init():
    global _vr,_ent
    _vr=VietorisRipsPersistence(homology_dimensions=list(range(MAXDIM+1)), metric='euclidean', n_jobs=1)
    _ent=PersistenceEntropy()
def betti_on_grid(bd):
    if len(bd)==0: return np.zeros(BINS,dtype=np.float32)
    b=bd[:,0][:,None]; d=bd[:,1][:,None]
    return ((b<=GRID[None,:])&(GRID[None,:]<d)).sum(0).astype(np.float32)
def compute_vec(coords, z):
    coords=np.asarray(coords,dtype=np.float32); coords=coords-coords.mean(0,keepdims=True)
    zc=np.asarray(z,dtype=np.float32).reshape(-1,1); zc=(zc-zc.mean())/(zc.std()+1e-6)*ZSCALE
    pts=np.concatenate([coords, zc], axis=1); pts=pts-pts.mean(0,keepdims=True)
    dm=np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(-1)); diam=float(dm.max())
    if diam>0: pts=pts/diam
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        diag=_vr.fit_transform(pts[None,:,:]); ent=_ent.fit_transform(diag)[0]
    d0=diag[0]
    h0=betti_on_grid(d0[d0[:,2]==0][:,:2]); h1=betti_on_grid(d0[d0[:,2]==1][:,:2])
    return np.concatenate([h0,h1,np.asarray(ent,dtype=np.float32)]).astype(np.float32)
def work(a):
    mid,pos,z=a
    try: return mid, compute_vec(pos,z)
    except Exception: return mid, None
def main():
    idx=json.load(open('cache/index.json')); ids=[r['id'] for r in idx]
    with h5py.File(H5,'r') as f:
        items=[(mid, np.asarray(f['data'][mid]['structure']['pos'][...]), np.asarray(f['data'][mid]['structure']['z'][...])) for mid in ids]
    print('element-augmented 4D VR z_PH FIXED grid 4D for %d mols, %d procs...'%(len(items),NPROC),flush=True)
    with Pool(NPROC, initializer=_init) as P:
        res=P.map(work, items, chunksize=200)
    order,vecs,fails=[],[],[]
    for mid,v in res:
        (fails if v is None else order).append(mid)
        if v is not None: vecs.append(v)
    Z=np.stack(vecs).astype(np.float32); np.save('cache/zph_elem4d.npy', Z)
    json.dump(dict(order=order,dim=int(Z.shape[1]),n_fail=len(fails),grid='fixed_0_1_64',kind='Z-weighted-4D'), open('cache/zph_elem4d_meta.json','w'))
    print('element-augmented 4D VR z_PH DONE fixed: %dx%d, %d fails'%(Z.shape[0],Z.shape[1],len(fails)),flush=True)
if __name__=='__main__': main()
