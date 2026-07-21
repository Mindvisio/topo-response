import h5py, numpy as np, json, os, time
from ase import Atoms
from schnetpack.data import ASEAtomsData
H5='/home/yc-user/data/squirl/SQuIRL_v1.0.h5'
idx=json.load(open('cache/index.json')); ids=[r['id'] for r in idx]
if os.path.exists('cache/squirl.db'): os.remove('cache/squirl.db')
db=ASEAtomsData.create('cache/squirl.db', distance_unit='Ang', property_unit_dict={'dipole_moment':'a.u.','polarizability':'a.u.'})
t0=time.time(); AL=[]; P=[]
with h5py.File(H5,'r') as f:
    for k,mid in enumerate(ids):
        g=f['data'][mid]
        z=np.asarray(g['structure']['z'][...]); pos=np.asarray(g['structure']['pos'][...])
        dip=np.asarray(g['electrostatics']['dipole_moment'][...]).astype(np.float64)
        al=np.asarray(g['electrostatics']['dipole_polarizability'][...]).reshape(-1).astype(np.float64)
        AL.append(Atoms(numbers=np.asarray(z), positions=np.asarray(pos))); P.append({'dipole_moment':dip,'polarizability':al})
        if len(AL)>=5000: db.add_systems(P, AL); AL=[]; P=[]
if AL: db.add_systems(P, AL)
print('DB DONE: %d systems in %.0fs'%(len(db),time.time()-t0))
