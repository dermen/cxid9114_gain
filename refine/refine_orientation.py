
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

exp_name = sys.argv[1]
data_name = sys.argv[2]

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
                                       scanX=np.arange(-.35, .35, .025),  # these seemed to be sufficient ranges
                                       scanY=np.arange(-.35, .35, .025))

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
    Ncells_abc=(5, 5, 5), mos_dom=20, mos_spread=0.0)

spot_dataA = spot_utils.get_spot_data_multipanel(
    simsAB[0], detector=detector,
    beam=beamA, crystal=optCrystal, thresh=0,
    filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

spot_dataB = spot_utils.get_spot_data_multipanel(
    simsAB[1], detector=detector,
    beam=beamB, crystal=optCrystal, thresh=0,
    filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

spot_resid, dvecs, best = \
    metrics.indexing_residuals_twocolor(spot_dataA, spot_dataB, refls_strong, detector)

HA, HiA = spot_utils.refls_to_hkl(refls_strong, detector, beamA, optCrystal)
HB, HiB = spot_utils.refls_to_hkl(refls_strong, detector, beamB, optCrystal)

HAres = np.sqrt( np.sum((HA-HiA)**2, 1))
HBres = np.sqrt( np.sum((HB-HiB)**2, 1))
Hres = np.min( zip(HAres, HBres), axis=1)

hkl_tol = 0.15
d_idx = spot_resid[Hres < hkl_tol]
dvecs_idx = dvecs[Hres < hkl_tol]

dump = {"crystalAB": optCrystal,
        # "res_opt": res_opt,
        # "color_opt": color_opt,
        # "resAB": resAB,
        # "colorAB": colorAB,
        "beamA": beamA,
        "beamB": beamB,
        "overlap": overlap,
        "detector": detector,
        "spot_dataA": spot_dataA,
        "spot_dataB": spot_dataB,
        "d": spot_resid,
        "d_idx": d_idx,
        "dvecs_idx": dvecs_idx,
        "Hres": Hres,
        "dvecs": dvecs,
        "best": best,
        "rmsd": data['rmsd'],
        # "dist_vecs": dist_vecs,
        # "dists": dists,
        # "spot_data_combo": spot_data_combo,
        "refls_strong": refls_strong}

dump_name = data_name.replace(".pkl", "_ref.pkl")
utils.save_flex(dump, dump_name)

#sim_fname = data_name.replace(".pkl", "_ref_sim64.h5")
#sim_utils.save_twocolor(simsAB, iset, sim_fname, force=0)
