import json
import numpy as np
from quantum_signals.replication import run_quantum_cv,run_classical_cv

rng=np.random.default_rng(2026)
X=np.r_[rng.normal(size=(24,20)),rng.normal(2,1,size=(12,20))]
y=np.r_[np.zeros(24,int),np.ones(12,int)]
print(json.dumps({'seed':2026,'data':'synthetic IID anomaly mixture; no financial returns',
 'classical':run_classical_cv(X,y,k=3,seed=2026),
 'quantum_CX':run_quantum_cv(X,y,'CX',k=3,seed=2026),
 'advantage_claim':False},indent=2))
