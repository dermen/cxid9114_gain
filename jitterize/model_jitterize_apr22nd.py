
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
outdir = "OUTS"
szx = szy = 11
ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]
FF = [1e4, None]
FLUX = [1e12, 1e12]
N = 30
Nmos_dom=1
cell_fwhm=0.14
rot_width=.3 
min_Ncell=5
max_Ncell=10
min_mos_spread=0.005
max_mos_sprea=0.1
szx = szy = 11


cryst_name = "ps2.crystR.pkl"
output_basename = cryst_name.replace(".pkl", tag)

# load the project beam, crystal-base model, and detector
det = detector = utils.open_flex("xfel_det.pkl")
beam = utils.open_flex("ps2.beam.pkl")
cryst = utils.open_flex(cryst_name)
crystalAB = cryst

print ("Doing the starting sim")
# generate base crystal pattern and 
simsAB = sim_utils.sim_twocolors2(
    cryst, 
    det, 
    beam, 
    fcalcs=FF,
    energies= ENERGIES,
    fluxes=FLUX,
    pids=None, 
    profile='tophat',
    oversample=0,
    Ncells_abc=(10, 10, 10), 
    mos_dom=1,
    verbose=1,
    mos_spread=0.0)

beamA = deepcopy(beam)
beamB = deepcopy( beam)
beamA.set_wavelength(parameters.WAVELEN_LOW)
beamB.set_wavelength(parameters.WAVELEN_HIGH)

refl_simA = spot_utils.refls_from_sims(simsAB[0], det, beamA, thresh=1e-3)
refl_simB = spot_utils.refls_from_sims(simsAB[1], det, beamB, thresh=1e-3)

simsDataSum = simsAB[0] + simsAB[1]
refl_data = refls_strong = spot_utils.refls_from_sims(simsDataSum, det, beamA, thresh=1e-3)

###########
###########
###########
###########

outputname = os.path.join(outdir, output_basename) 

h5_fname = outputname + ".h5py"

reflsPP = spot_utils.refls_by_panelname(refl_data)
pids = reflsPP.keys()

roi_pp = []
counts_pp =[]
img_sh = (185, 194)

Malls = {}
for pid in pids:
    panel = det[pid]
    rois = spot_utils.get_spot_roi(
        reflsPP[pid],
        dxtbx_image_size=panel.get_image_size(),
        szx=szx, szy=szy)
    counts = spot_utils.count_roi_overlap(rois, img_size=img_sh)

    roi_pp.append(rois)
    counts_pp.append(counts)

    spot_masks = spot_utils.strong_spot_mask(
        reflsPP[pid], img_sh, as_composite=False)

    # composite mask
    Mall = np.any( spot_masks, axis=0)
    
    Malls[pid] = spot_masks 

print "Saving the masks"
np.savez(outputname + "_spotdata.pkl", Malls=Malls, roi_pp=roi_pp, counts_pp = counts_pp)

from IPython import embed
embed()

pan_img_idx = {pid: idx for idx, pid in enumerate(pids)}

Nrefl = len(refls_strong)

###########
###########
###########
###########

# inputs to the simulator:
# crystal 
# size
# mosaic parameters
# Energies

with h5py.File(h5_fname, "w") as h5:

    Ncells_abc, mos_doms, mos_spread, xtal_shapes, ucell_a, ucell_b, ucell_c \
        = [], [], [], [], [], [], []
    new_crystals  = []
    master_img = None
    for i_trial in range(N):
        # add a jittered unit cell and a jittered U matrix ( in all 3 dimensions)
        new_crystal = JitterFactory.jitter_crystal(
            crystalAB, 
            cell_jitter_fwhm=cell_fwhm, 
            rot_jitter_width=rot_width)

        # jitter the size, shape and mosaicity of the crystal
        new_shape = JitterFactory.jitter_shape(
            min_Ncell=min_Ncell, max_Ncell=max_Ncell, 
            min_mos_spread=0.005, max_mos_spread=0.1)
        print "### CRYSTAL SIMULATION %d / %d ###" % (i_trial+1, N)
        print new_crystal.get_unit_cell().parameters()
        print new_shape
        print ENERGIES
        t = time.time()
        simsAB = sim_utils.sim_twocolors2(
            new_crystal,
            detector,
            beamA,
            FF,
            ENERGIES, 
            FLUX,
            pids = None,
            profile='tophat', # new_shape['shape'],
            oversample=0,  # this should let nanoBragg decide how to oversample!
            Ncells_abc = new_shape['Ncells_abc'],
            mos_dom=Nmos_dom, 
            mos_spread=0 , #new_shape['mos_spread'],
            roi_pp=roi_pp,
            counts_pp=counts_pp,
            cuda=False)
        JFC = 10
        sim_imgs = np.array(simsAB[0]) + np.array(simsAB[1])
        h5.create_dataset("sim_imgs_%d"% i_trial, data=sim_imgs)

        print("\t... took %.4f seconds" % (time.time() - t))
        
        Ncells_abc.append(new_shape['Ncells_abc'])
        mos_doms.append(Nmos_dom)
        mos_spread.append(new_shape['mos_spread'])
        xtal_shapes.append(new_shape['shape'])
        a, b, c, _, _, _ = new_crystal.get_unit_cell().parameters()
        ucell_a.append(a)
        ucell_b.append(b)
        ucell_c.append(c)
        new_crystals.append( new_crystal)

    utils.save_flex(refls_strong, outputname + "_refls.pkl") 


