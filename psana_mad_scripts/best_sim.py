
import sys

from cxid9114.refine.jitter_refine import JitterFactory
import os
from cxid9114 import utils
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from dxtbx.model.experiment_list import ExperimentListFactory
import numpy as np
import pandas
from itertools import izip
from scitbx.array_family import flex
import time
from copy import deepcopy


ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  

best_name = sys.argv[1]
run = int (sys.argv[2])
shot_idx = int( sys.argv[3])
szx = szy = 5

df = pandas.read_pickle( best_name)

idxmax = df.ave_score.idxmax()
best_Amat = tuple(df.Amat.iloc[ idxmax])
best_Ncell_abc = df[ ["Na", "Nb", "Nc"]].iloc[idxmax].values
best_mos_spread = df["mos_spread"].iloc[idxmax]
best_shape = df["xtals_shape"].iloc[idxmax]
best_Nmos_dom = df["Nmos_domain"].iloc[idxmax]

best_shape = "gauss"

exp_name = "results/run%d/exp_%d_feb8th.json" % (run, shot_idx)
data_name = "results/run%d/dump_%d_feb8th.pkl" % (run, shot_idx)
cuda = False

exp_lst = ExperimentListFactory.from_json_file(exp_name) #, check_format=False)
iset = exp_lst.imagesets()[0]
data = utils.open_flex( data_name)

waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e12, 1e12]  # fluxes of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]

data = utils.open_flex( data_name)
beamA = data["beamA"]
beamB = data["beamB"]
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

refls_strong = data["refls_strong"]
print ("###\n %d reflections\n" % len(refls_strong))

crystalAB = data["crystalAB"]

best_crystal = deepcopy(crystalAB)
best_crystal.set_A(best_Amat)

detector = data["detector"]
reflsPP = spot_utils.refls_by_panelname(refls_strong)
pids = reflsPP.keys()
raw_dat = iset.get_raw_data(0)
pan_imgs = [raw_dat[pid].as_numpy_array()
            for pid in pids]

roi_pp = []
counts_pp =[]
for pid, img in izip(pids, pan_imgs):
    panel = detector[pid]
    rois = spot_utils.get_spot_roi(
        reflsPP[pid],
        dxtbx_image_size=panel.get_image_size(),
        szx=szx, szy=szy)
    counts = spot_utils.count_roi_overlap(rois, img_size=img.shape)

    roi_pp.append(rois)
    counts_pp.append(counts)

t = time.time()
print "Entering the sim"
simsAB = sim_utils.sim_twocolors2(
    best_crystal,
    detector,
    beamA,
    FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX,
    pids=pids,
    profile='gauss', #best_shape,
    oversample=0,  # this should let nanoBragg decide how to oversample!
    Ncells_abc=best_Ncell_abc,
    mos_dom=best_Nmos_dom, 
    mos_spread=best_mos_spread,
    roi_pp=roi_pp,
    counts_pp=counts_pp,
    cuda=cuda)

print("\t... took %.4f seconds" % (time.time() - t))
print "Exiting the sim"

from IPython import embed
embed()

