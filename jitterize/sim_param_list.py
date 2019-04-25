
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
    tag = sys.argv[1]
except IndexError:
    tag = ""

PIDS = np.arange(32,40)

spotdata = np.load("crystR.spotdata.pkl.npz")
roi_pp = spotdata['roi_pp'][()]
counts_pp = spotdata["counts_pp"][()]

param_name = sys.argv[1]
params = utils.open_flex(param_name)
tag = os.path.basename(param_name).replace(".pkl", "")
outdir = tag
if not os.path.exists( outdir):
    os.makedirs(outdir)
output_basename = os.path.join( outdir, "simparams_%s" % tag)
h5_fname = output_basename + ".h5py"
h5 = h5py.File(h5_fname, "w")

# load the project beam, crystal-base model, and detector
det = detector = utils.open_flex("xfel_det.pkl")

###########
###########
###########

# inputs to the simulator:
# crystal 
# size
# mosaic parameters
# Energies


def sim_params(params):
    t = time.time()
    FLUX = [1e12, 1e12]
    FF = [1e4,1e4]
    new_crystal = params['crystal']
    new_shape = params['shape']
    ENERGIES = params['ENERGIES']
    beam = params['beam']
    Ncells = params['Ncells_abc']
   

    if 'mos_spread' in params.keys():
        mos_spread = params['mos_spread']
        Nmos_dom = 200
    else:
        mos_spread = 0
        Nmos_dom = 1
    print "### CRYSTAL SIMULATION %d / %d ###" % (i_trial+1, N)
    print new_crystal.get_unit_cell().parameters()
    print new_shape
    print ENERGIES
    simsAB = sim_utils.sim_twocolors2(
        new_crystal,
        detector,
        beam,
        FF,
        ENERGIES, 
        FLUX,
        pids = None,
        profile=new_shape,
        oversample=0,  # this should let nanoBragg decide how to oversample!
        Ncells_abc = Ncells,
        mos_dom=Nmos_dom, 
        mos_spread=mos_spread, 
        roi_pp=roi_pp,
        counts_pp=counts_pp,
        cuda=False)
    print("\t... took %.4f seconds" % (time.time() - t))

    return simsAB


N = len( params)
for i_trial in range(N):

    simsAB = sim_params(params[i_trial])
    sim_imgs = np.array(simsAB[0]) + np.array(simsAB[1])
    h5.create_dataset("sim_imgs_%d"% i_trial, data=sim_imgs)

    
h5.close()


