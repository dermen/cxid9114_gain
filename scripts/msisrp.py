"""
msisrp stands for mad --> spot --> index --> simulate --> refine --> predict

Here we load a data image that has crystal diffraction
then we spot peaks and index the crystal image
Then simulate the indexed crystal and check overlap
Then refine using simulations
Then predict using simulations

"""
import numpy as np
from cxid9114.sim import sim_utils
from cxid9114 import utils
from copy import deepcopy
from cxid9114 import parameters
from cxid9114.spots import spot_utils
from cxid9114.index.sad import params as sad_index_params
from cxid9114.index.ddi import params as mad_index_params
from libtbx.utils import Sorry as Sorry
from cxid9114.spots import spot_utils
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
import dxtbx
from dxtbx.datablock import DataBlockFactory
from dials.array_family import flex
from dxtbx.model import Detector

# load a dummie hierarchy and store in template
# so we can use it to do simtbx in multi panel mode
simple_det = utils.open_flex(sim_utils.det_f)
dummie_hier = simple_det.to_dict()["hierarchy"]
#det_templ = {"hierarchy": simple_det.to_dict()["hierarchy"]}

fcalc_f = "/Users/dermen/cxid9114_gain/sim/fcalc_slim.pkl"

MULTI_PANEL = True

spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
spot_par_moder = deepcopy(spot_par)

spot_par.spotfinder.threshold.dispersion.global_threshold = 70.
spot_par.spotfinder.threshold.dispersion.gain = 28.
spot_par.spotfinder.threshold.dispersion.kernel_size = [4,4]
spot_par.spotfinder.threshold.dispersion.sigma_strong = 2.25
spot_par.spotfinder.threshold.dispersion.sigma_background =6.
spot_par.spotfinder.filter.min_spot_size = 2
spot_par.spotfinder.force_2d = True
spot_par.spotfinder.lookup.mask = "../mask/dials_mask_64panels_2.pkl"


spot_par_moder.spotfinder.threshold.dispersion.global_threshold = 56.
spot_par_moder.spotfinder.threshold.dispersion.gain = 28.
spot_par_moder.spotfinder.threshold.dispersion.kernel_size = [1,1]
spot_par_moder.spotfinder.threshold.dispersion.sigma_strong = 2.5
spot_par_moder.spotfinder.threshold.dispersion.sigma_background = 2.5
spot_par_moder.spotfinder.filter.min_spot_size = 1
spot_par_moder.spotfinder.force_2d = True
spot_par_moder.spotfinder.lookup.mask = "../mask/dials_mask_64panels_2.pkl"
#spot_par_moder.spotfinder.lookup.mask = "../mask/dials_mask2d.pickle"

try_fft1d = False
img_f = "/Users/dermen/cxid9114/multi_run62_hits_wtime.h5"
# img_f = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
loader = dxtbx.load(img_f)

detector = loader.get_detector(0)

info_f = utils.open_flex("../index/run62_idx_processed.pkl")
hit_idx = info_f.keys()
from IPython import embed
embed()
N = 20  # process 20
A_results, B_results, AB_results = [],[],[]
for idx in hit_idx[5:N+6]:

    iset = loader.get_imageset( img_f)[ idx:idx+1]
    dblock = DataBlockFactory.from_imageset(iset)[0]

    refls_strong = flex.reflection_table.from_observations(dblock, spot_par)
    # refls = info_f[idx]['refl']

    # TODO: add a new method of retrieving these values
    fracA = info_f[idx]['fracA']
    fracB = info_f[idx]['fracB']

    cryst_orig = info_f[idx]['crystals'][0]

    # load a test crystal
    #crystal = utils.open_flex( sim_utils.cryst_f )

    # fraction of two color energy
    # simulate the pattern
    #Patts = sim_utils.PatternFactory()
    en, fcalc = sim_utils.load_fcalc_file("../sim/fcalc_slim.pkl")
    flux = [fracA * 1e14, fracB * 1e14]
    #imgA, imgB = Patts.make_pattern2(crystal, flux, en, fcalc, 20, 0.1, False)

    # ==================================
    # 2 color indexer of 2 color pattern
    # ==================================
    sad_index_params.indexing.multiple_lattice_search.max_lattices = 1
    sad_index_params.indexing.stills.refine_all_candidates = False
    sad_index_params.indexing.stills.refine_candidates_with_known_symmetry = False
    sad_index_params.indexing.stills.candidate_outlier_rejection =  False
    sad_index_params.indexing.stills.rmsd_min_px = 10
    sad_index_params.indexing.refinement_protocol.mode = "ignore"
    waveA = parameters.ENERGY_CONV / en[0]
    waveB = parameters.ENERGY_CONV / en[1]

    beamA = deepcopy( iset.get_beam())
    beamB = deepcopy( iset.get_beam())

    beamA.set_wavelength(waveA)
    beamB.set_wavelength(waveB)

    isetA = deepcopy(iset)
    isetB = deepcopy(iset)
    isetA.set_beam(beamA)
    isetB.set_beam(beamB)

    #if try_fft1d:
    #    # index two color pattern using fft1d
    #    try:
    #        orientA = indexer_base.from_parameters(
    #            reflections=spot_utils.as_single_shot_reflections(refls, inplace=False),
    #            imagesets=[isetA],
    #            params=sad_index_params)
    #        orientA.index()
    #        crystalA = orientA.refined_experiments.crystals()[0]
    #        outA = sim_utils.sim_twocolors(crystalA, mos_dom=20, mos_spread=0.1, fracA=fracA, fracB=fracB, plot=False)
    #        A_results.append(outA)

    #    except:
    #        A_results.append(None)

    #    try:
    #        # try with other color
    #        orientB = indexer_base.from_parameters(
    #            reflections=spot_utils.as_single_shot_reflections(refls, inplace=False),
    #            imagesets=[isetB],
    #            params=sad_index_params)
    #        orientB.index()
    #        crystalB = orientB.refined_experiments.crystals()[0]
    #        outB = sim_utils.sim_twocolors(crystalB, mos_dom=20, mos_spread=0.1, fracA=fracA, fracB=fracB, plot=False)
    #        B_results.append(outB)
    #    except:
    #        B_results.append(None)

    # ==================================
    # 2 color indexer of 2 color pattern
    # ==================================
    try:
        orientAB = indexer_two_color(
            reflections=spot_utils.as_single_shot_reflections(refls_strong, inplace=False),
            imagesets=[iset],
            params=mad_index_params)
        orientAB.index()
    except (Sorry, RuntimeError):
        AB_results.append(None)
        continue

    if not orientAB.refined_experiments.crystals():
        continue

    refls_moder = flex.reflection_table.from_observations(dblock, spot_par_moder)

    crystalAB = orientAB.refined_experiments.crystals()[0]


    refls_strong_pp = spot_utils.refls_by_panelname(refls_strong)
    refls_moder_pp = spot_utils.refls_by_panelname(refls_moder)

    # now lets iterate over the panels, and then only simulate a panel
    # if it has more than 1 strong spot!
    sim_res = {}
    for i_pan in range(64):
        if i_pan not in refls_strong_pp:
            continue

        n_strong = len(refls_strong_pp[i_pan])
        if n_strong < 2:
            continue

        dump = sim_utils.sim_twocolors(crystalAB,
                                    detector=detector,
                                    panel_id=i_pan,
                                    mos_dom=20,
                                    fcalc_f = fcalc_f,
                                    Gauss=False,
                                    mos_spread=0.1,
                                    fracA=fracA,
                                    fracB=fracB)

        sim_res[i_pan] = dump
        from IPython import embed
        embed

        #imgA, imgB = dump['imgA'], dump['imgB']
        #spotsA = spot_utils.get_spot_data(imgA, thresh=1e-6)
        #spotsB = spot_utils.get_spot_data(imgB, thresh=1e-6)

        #yA, xA = map( np.array, zip(*spotsA["comIpos"]))  # from the simulated Low-energy (labeled by A here) pattern
        #yB, xB = map( np.array, zip(*spotsB["comIpos"]))  # from the simulated Low-energy (labeled by A here) pattern

        #crystalAB = dump["sim_param"]['crystal']

        #color_data = {'A': spot_utils.make_color_data_object(xA, yA, beamA, crystalAB, detector),
        #              'B': spot_utils.make_color_data_object(xB, yB, beamB, crystalAB, detector)}

        #indexa = spot_utils.compute_indexability(refls_strong, color_data, hkl_tol=0.15)
        #indexa_moder = spot_utils.compute_indexability(refls_moder, color_data, hkl_tol=0.1)

    AB_results.append(dump)


from IPython import embed
embed()

