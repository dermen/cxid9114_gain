
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
from cxid9114.index.ddi import  params as mad_index_params
import sys
from cxid9114 import utils
from dxtbx.model.experiment_list import  ExperimentListFactory
from dials.array_family import flex
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import scipy.ndimage
from libtbx.utils import Sorry

from cxid9114.refine import metrics
ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [5000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e14, 1e14]  # fluxes of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]


exp_name = sys.argv[1]
data_name = sys.argv[2]
good_cut = 7


exp_lst = ExperimentListFactory.from_json_file(exp_name)
iset = exp_lst.imagesets()
detector = iset.get_detector(0)
data = utils.open_flex( data_name)
beamA = deepcopy(iset.get_beam())
beamB = deepcopy(iset.get_beam())
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

spot_resid = data['d'] / detector[0].get_pixel_size()[0]
refls_strong = data['refls_strong']

xtal_number = 0
while 1:

    not_indexed = spot_resid > good_cut
    refls_strong = refls_strong.select(flex.bool(not_indexed))

    orientAB = indexer_two_color(
        reflections=spot_utils.as_single_shot_reflections(refls_strong, inplace=False),
        imagesets=[iset],
        params=mad_index_params)

    try:
        orientAB.index()

    except (RuntimeError, Sorry):
        print("Cound not reindex")
        continue

    cryst_model = orientAB.refined_experiments.crystals()[0]
    simsAB = sim_utils.sim_twocolors2(
        cryst_model, detector, iset.get_beam(0), [5000, None],
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        [1e14, 1e14], pids=None, Gauss=False, oversample=4,
        Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=0.0)

    spot_dataA = spot_utils.get_spot_data_multipanel(
        simsAB[0], detector=detector,
        beam=beamA, crystal=cryst_model, thresh=0,
        filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

    spot_dataB = spot_utils.get_spot_data_multipanel(
        simsAB[1], detector=detector,
        beam=beamB, crystal=cryst_model, thresh=0,
        filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

    spot_resid, dvecs, best = \
        metrics.indexing_residuals_twocolor(spot_dataA, spot_dataB, refls_strong, detector)

    dump = {"crystalAB": cryst_model,
            # "res_opt": res_opt,
            # "color_opt": color_opt,
            # "resAB": resAB,
            # "colorAB": colorAB,
            "beamA": beamA,
            "beamB": beamB,
            "detector": detector,
            "spot_dataA": spot_dataA,
            "spot_dataB": spot_dataB,
            "d": spot_resid,
            "dvecs": dvecs,
            "best": best,
            "rmsd": orientAB.best_rmsd,
            # "dist_vecs": dist_vecs,
            # "dists": dists,
            # "spot_data_combo": spot_data_combo,
            "refls_strong": refls_strong}
    dump_name = data_name.replace(".pkl", "_%d.pkl" % xtal_number)
    utils.save_flex(dump, dump_name)
    xtal_number += 1

    sim_fname = data_name.replace(".pkl", "_%d_sim64.h5" % xtal_number)
    sim_utils.save_twocolor(simsAB, iset, sim_fname, force=0)
    #if jitt_ref:
    #    dump["crystalOpt"] = optCrystal
    #    dump["d2"] = d2
    #    dump["overlap"] = overlap
    #    dump["dvecs2"] = dvecs2
    #    dump["best2"] = best2

