from numpy import *
import h5py
import numpy as np
from collections import Counter
import sys
from cxid9114.spots import spot_utils
import os
from cxid9114 import utils

indir = sys.argv[1]
img_sh = 185, 194
h5_name = [os.path.join(indir,f) for f in os.listdir(indir) if f.endswith(".h5py")][0]
pkl_name = [os.path.join(indir,f) for f in os.listdir(indir) if f.endswith(".pkl")][0]
print h5_name, pkl_name

f = h5py.File(h5_name, "r")
D = utils.open_flex(pkl_name)
print "Using %d reflections!" % len(D)
Dpp = spot_utils.refls_by_panelname(D)
pids = f['pids'].value
pid_map = {p:i for i,p in enumerate( pids)}
Nsim = sum([1 for k in f.keys() if k.startswith('sim_imgs')])
dsets = [f['sim_imgs_%d'%x] for x in range(Nsim)]

Dpp_pids = Dpp.keys()
strong_mask = array([spot_utils.strong_spot_mask(Dpp[p], img_sh) for p in Dpp_pids])
thresh = 2  # 2 photon threshold
scores = {}
for i,mask in enumerate(strong_mask):
    pid = Dpp_pids[i]
    dset_idx = pid_map[pid]
    pan = array([d[dset_idx] > thresh for d in dsets])
    scores[pid] = [(~logical_xor( mask, p)).sum() for p in pan]
    
vals = []
for p in Dpp_pids:
    s = scores[p]
    mx = max(s)
    idx = where(array(s)== mx)[0]
    vals += list(idx)
    
C = Counter(vals)

print ( sorted( C.items(), key=lambda x: x[1]) )

