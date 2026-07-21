import glob, re, numpy as np, csv
try:
    from scipy import stats; HAVE=True
except Exception: HAVE=False
LOGS='logs'
def gv(pat,m):
    fs=glob.glob(pat)
    if not fs: return None
    r=re.findall(m+r' ([0-9.]+)', open(fs[0]).read()); return float(r[-1]) if r else None
rows=[]; res={}
for prop,metric,pref in [('dipole','compMAE','dip'),('polar','Frob','polar')]:
    for cond in ['none','tda','random']:
        vals=[]
        for s in range(5):
            if s==0: pat=f'{LOGS}/mx_{pref}_topology_ood_{cond}_s0.log'
            elif s in (1,2): pat=f'{LOGS}/s4b_{pref}_topology_ood_{cond}_s{s}.log'
            else: pat=f'{LOGS}/s4c_{pref}_topology_ood_{cond}_s{s}.log'
            v=gv(pat,metric); vals.append(v); rows.append([prop,cond,s,v])
        res[(prop,cond)]=np.array([x for x in vals if x is not None],dtype=float)
with open('results_5seed.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['property','conditioning','seed','metric_value']); w.writerows(rows)
def ci(A,B):
    d=res[A]-res[B]; n=len(d); m=d.mean(); s=d.std(ddof=1); se=s/np.sqrt(n)
    if HAVE: tc=stats.t.ppf(0.975,n-1); p=float(stats.ttest_rel(res[A],res[B]).pvalue)
    else: tc=2.776; p=float('nan')
    return m,m-tc*se,m+tc*se,p
print('scipy:',HAVE)
print('property comparison        mean      95%CI                 p')
for prop in ['dipole','polar']:
    for A,B,lbl in [((prop,'tda'),(prop,'none'),'tda-baseline'),((prop,'tda'),(prop,'random'),'tda-random'),((prop,'random'),(prop,'none'),'random-baseline')]:
        m,lo,hi,p=ci(A,B); print('%-6s %-15s %+.4f [%+.4f,%+.4f] p=%.3f %s'%(prop,lbl,m,lo,hi,p,'SIG' if (p==p and p<0.05) else 'n.s.'))
print('CSV rows:',len(rows))
