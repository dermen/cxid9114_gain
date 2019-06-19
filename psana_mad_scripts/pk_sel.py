# coding: utf-8
import h5py
from cxid9114 import utils
from cxid9114.spots import spot_utils
import numpy as np

idx = 6641
tag = "_fine"
run = 63
root = "/reg/d/psdm/cxi/cxid9114/scratch/dermen/idx/mad/"
dump_file = "%s/results/run%d/dump_%d_feb8th.pkl" % (root, run, idx)
f = h5py.File("%s/videos/run%d/shot%d%s/dump_%d_feb8th_jitt.h5py" % (root,run, idx, tag, idx), "r")

pids = f['pids'].value
#pan_imgs = f['pan_imgs'].value
# for the simulated images
keys = [ k for k in f.keys() if k.startswith("sim_imgs_")]
Ntrials = len( keys)

d = utils.open_flex(dump_file)
R = d['refls_strong']
Rpp = spot_utils.refls_by_panelname(R)

Masks = spot_utils.strong_spot_mask( R, (185,194), as_composite=False)
min_prob = 0.1 / Ntrials

Ntotal_bad = Ntotal_spots = 0
for pidx, pid in enumerate( pids):

    sim = np.array([f[k][pidx] for k in keys])

    im = sim.mean(0)
    im /= im.max()  # make it a probability
    M0 = spot_utils.strong_spot_mask( Rpp[pid], (185,194), as_composite=False)
        
    vals = []
    for m in M0:
        vec = m*im
        val = vec[vec > 0].mean()
        vals.append(val)
        
    bad = np.array(vals) < min_prob

    print "PID %02d , number of bad spots = %d" % (pid, sum(bad) )
    Ntotal_bad += sum( bad)
    Ntotal_spots += len( vals)

print "Filtered %d / %d bad spots!" % (Ntotal_bad, Ntotal_spots)
print "Kept %d spots!" % (Ntotal_spots - Ntotal_bad)

