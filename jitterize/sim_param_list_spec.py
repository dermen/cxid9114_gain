
from copy import deepcopy
import pylab as plt
import matplotlib as mpl
import time
import numpy as np
from cxid9114.refine.jitter_refine import JitterFactory
from cxid9114.sim import sim_utils
import pylab as plt
from cxid9114 import parameters
from cxid9114 import utils
from cxid9114.spots import spot_utils
import os
import h5py

import sys

try:
    tag2 = sys.argv[2]
except IndexError:
    tag2 = ""

panel_ids = [32,34,39,38,50]

#spotdata = np.load("crystR.spotdata.pkl.npz")
#roi_pp = spotdata['roi_pp'][()]
#counts_pp = spotdata["counts_pp"][()]
#roi_pp = np.array([roi_pp[pid] for pid in panel_ids])
#counts_pp = np.array([counts_pp[pid] for pid in panel_ids])

param_name = sys.argv[1]
params = utils.open_flex(param_name)
tag = os.path.basename(param_name).replace(".pkl", "") + tag2
outdir = tag
if not os.path.exists( outdir):
    os.makedirs(outdir)
output_basename = os.path.join( outdir, "simparams_%s" % tag)

# load the project beam, crystal-base model, and detector
det = detector = utils.open_flex("xfel_det.pkl")

def sim_params(params):
    """special jitterization params list"""
    t = time.time()
    
    new_crystal = params['crystal']
    new_shape = params['shape']
    ENERGIES = params['ENERGIES']
    FLUX = np.array(params['FLUX']) * 1e12
    beam = params['beam']
    Ncells = params['Ncells_abc']

    #if 'mos_spread' in params.keys():
    #    mos_spread = params['mos_spread']
    #    Nmos_dom = 100
    #else:
    #    Nmos_dom = 1
    #    mos_spread = 0.10
    
    mos_spread=0.1
    Nmos_dom=50

    # setup the formfactors
    FF = [None]*len( FLUX)
    FF[0] = 1e5
    
    print "### CRYSTAL SIMULATION %d / %d ###" % (i_trial+1, N)
    print new_crystal.get_unit_cell().parameters()
    print new_shape
    print ENERGIES
    print FLUX
    simsAB = sim_utils.sim_twocolors2(
        new_crystal,
        detector,
        beam,
        FF,
        ENERGIES, 
        FLUX,
        pids = panel_ids,
        profile=new_shape,
        oversample=0,  # this should let nanoBragg decide how to oversample!
        Ncells_abc = Ncells,
        mos_dom=Nmos_dom, 
        mos_spread=mos_spread, 
        verbose=0,
        #roi_pp=roi_pp,
        #counts_pp=counts_pp,
        omp=True)
    print("\t... took %.4f seconds" % (time.time() - t))

    return simsAB

h5_fname = output_basename + ".h5py"

with  h5py.File(h5_fname, "w") as h5:

    N = len( params)
    for i_trial in range(N):

        simsAB = sim_params(params[i_trial])
        sim_imgs = np.array(simsAB[0]) 
        h5.create_dataset("sim_imgs_%d"% i_trial, data=sim_imgs)
   
    h5.create_dataset("panel_ids", data=panel_ids)

