
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
import sys

# OPEN THE DETECTOR AND THE BEAM!
det = detector = utils.open_flex("xfel_det.pkl")
beam = utils.open_flex("ps2.beam.pkl")

try:
    tag = sys.argv[1]
    outdir = tag
    if not os.path.exists(outdir):
        os.makedirs(outdir)
except IndexError:
    tag = ""  # DONT YOU EVEN CARE?
    outdir = "OUTS/"

szx = szy = 11
spec_data = np.load("spec_gamma5.npz")
ENERGIES = spec_data['en']
FLUX = spec_data['flux'] * 1e12

FF = [None]*len( FLUX)
FF[0] = 1e8
panel_ids = [32,34,50]
#ENERGIES = [parameters.ENERGY_LOW] #, parameters.ENERGY_HIGH]
#FF = [1e8] #, None]
#FLUX = [1e12] #, 1e12]
cell_fwhm=0.14
rot_width=.3 
min_Ncell=5
max_Ncell=10
Nmos = 500
mos_spread = 0.35
szx = szy = 11

cryst_name = "ps2.crystR.pkl"
output_basename = cryst_name.replace(".pkl", tag)

# load the project beam, crystal-base model, and detector
cryst = utils.open_flex(cryst_name)

print ("Doing the starting sim")
# generate base crystal pattern and 
t =time.time()
simsAB, Patt = sim_utils.sim_twocolors2(
    cryst, 
    det, 
    beam, 
    fcalcs=FF,
    energies= ENERGIES,
    fluxes=FLUX,
    pids=panel_ids,
    profile='tophat',
    oversample=0,
    Ncells_abc=(10, 10, 10), 
    mos_dom=Nmos,
    verbose=1,
    mos_spread=mos_spread,
    omp=False,
    gimmie_Patt=True)
print "TIME: %.4f" % (time.time()-t)

beamA = deepcopy(beam)
beamB = deepcopy(beam)
beamA.set_wavelength(parameters.WAVELEN_LOW)
beamB.set_wavelength(parameters.WAVELEN_HIGH)

sims = np.sum( [simsAB[k] for k in simsAB], axis=0)

from dxtbx.model.detector import Detector
from scitbx.array_family import flex

det2 = Detector()
det2_panel_mapping = {}
for i_pan, pid in enumerate(panel_ids):
    det2.add_panel(det[pid])
    det2_panel_mapping[i_pan] = pid

refl_data = refls_strong = spot_utils.refls_from_sims(sims, det2, beamA, thresh=1e-3)

refl_panel = refl_data['panel'].as_numpy_array()
for i_pan, pid in det2_panel_mapping.items():
    sel = refl_panel == i_pan
    refl_panel[sel] = pid

refl_data['panel'] = flex.size_t(refl_panel)

###########
###########

outputname = os.path.join(outdir, output_basename) 

reflsPP = spot_utils.refls_by_panelname(refl_data)
assert( set(reflsPP.keys()) == set(panel_ids))

roi_pp = []
counts_pp =[]
img_sh = (185, 194)

Malls = {}
for i_pan,pid in enumerate(panel_ids):
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

bg_noise = np.random.normal(sims.mean()*0.1, sims.std()*1.3, sims.shape)
sims_wNoise = sims + bg_noise

print "Saving the masks"
np.savez(outputname + "_spotdata.pkl", Malls=Malls, 
    roi_pp=roi_pp, 
    counts_pp = counts_pp, 
    sims=sims,
    sims_wNoise = sims_wNoise,
    img2=sims_wNoise,
    panel_ids=panel_ids)

utils.save_flex(refls_strong, outputname + "_refls.pkl") 

#
## USE TO PLOT COMPARISON:
#
#img2 = sims_wNoise
##%pylab
#def cscale(img, contrast=0.1):
#    m90 = np.percentile(img, 90) 
#    return np.min( [np.ones(img.shape), 
#        contrast * img/m90],axis=0)
#imshow(cscale(img2[32],0.15), cmap='gray_r')
#figure()
#img = load("noise_img3.npz")["img"]
#imshow(cscale(img[32],0.15), cmap='gray_r')
#
#
