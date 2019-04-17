
import sys
from cxid9114 import utils
from dxtbx.model.experiment_list import  ExperimentListFactory
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import numpy as np
from cxid9114.refine import jitter_refine
import scipy.ndimage
from cxid9114.refine import metrics

# TODO: replace data_name with reflection table (no weird format deps) 
# TODO: add argument parser and make many arguments

exp_name = sys.argv[1]
data_name = sys.argv[2]  
tag = sys.argv[3]
hkl_tol = .15

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [5000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e14, 1e14]  # fluxes of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
exp_lst = ExperimentListFactory.from_json_file(exp_name)
iset = exp_lst.imagesets()[0]
detector = iset.get_detector(0)
data = utils.open_flex( data_name)
beamA = deepcopy(iset.get_beam())
beamB = deepcopy(iset.get_beam())
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

refls_strong =data["refls_strong"]
crystalAB = data["crystalAB"]
reflsPP = spot_utils.refls_by_panelname(refls_strong)
pids = [i for i in reflsPP if len(reflsPP[i]) > 0]  # refine on these panels only
pan_imgs = [iset.get_raw_data(0)[pid].as_numpy_array()
            for pid in pids]

# helper wrapper for U-matrix grid search based refinement
# `scanZ = ...` can also be passed as an argument, to jitter rotation
# about the Z (beam) axis
jitt_out = jitter_refine.jitter_panels(panel_ids=pids,  # we only simulate the pids with strong spots
                                       crystal=crystalAB,
                                       refls=refls_strong,
                                       det=detector,
                                       beam=iset.get_beam(0),
                                       FF=FF,
                                       en=ENERGIES,
                                       data_imgs=pan_imgs,
                                       flux=FLUX,
                                       ret_best=False,
                                       Ncells_abc=(5,5, 5),
                                       oversample=1,
                                       Gauss=False,
                                       verbose=0,
                                       mos_dom=1,
                                       mos_spread=0.0,
                                       scanX=np.arange(-.35, .35, .01),  # these seemed to be sufficient ranges
                                       scanY=np.arange(-.35, .35, .01))
                                       #scanZ=np.arange(-.2,.2,.02))

# select the refined matrix based on overlap superposition
# overlap is a metric used in the JitterFactory (wrapped into jitter_panels)
# which checks agreement between data panels and simulated panels
overlap = np.sum([jitt_out[pid]['overlaps'] for pid in jitt_out], axis=0)
max_pos = np.argmax(overlap)
optA = jitt_out[jitt_out.keys()[0]]["A_seq"][
    max_pos]  # just grab the first A_seq cause same sequence is tested on all panels

optCrystal = deepcopy(crystalAB)
optCrystal.set_A(optA)

simsAB = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=None, Gauss=False, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0)

refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA) 
refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB) 
residA = metrics.check_indexable(
    refls_strong, refl_simA, detector, beamA, optCrystal, hkl_tol)
residB = metrics.check_indexable(
    refls_strong, refl_simB, detector, beamB, optCrystal, hkl_tol)

dump = {"crystalAB": crystalAB,
        "optCrystal": optCrystal,
        "residA": residA,
        "residB": residA,
        "beamA": beamA,
        "beamB": beamB,
        "overlap": overlap,
        "detector": detector,
        "refls_simA": refl_simA,
        "refls_simB": refl_simB,
        "rmsd": data['rmsd'],
        "refls_strong": refls_strong}

dump_name = data_name.replace(".pkl", "_%s.pkl" % tag)
utils.save_flex(dump, dump_name)

