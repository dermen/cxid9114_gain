"""
msisrp stands for mad --> spot --> index --> simulate --> refine --> predict

Here we load a data image that has crystal diffraction
then we spot peaks and index the crystal image
Then simulate the indexed crystal and check overlap
Then refine using simulations
Then predict using simulations

"""
import sys
import numpy as np
from cxid9114.sim import sim_utils
from cxid9114 import utils
from copy import deepcopy
from cxid9114 import parameters
from cxid9114.index.ddi import params as mad_index_params
from libtbx.utils import Sorry as Sorry
from cxid9114.spots import spot_utils
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
import dxtbx
from dxtbx.datablock import DataBlockFactory
from dxtbx.model.experiment_list import  ExperimentList, Experiment
from dials.array_family import flex
from cxid9114.refine import jitter_refine
from cxid9114.refine import metrics
import scipy.ndimage
import os

# -----------
# Parameters
# -----------
mask_file = "dials_mask_64panels_2.pkl"  # mask file for spot detection
img_f = "run62.loc"  # dxtbx format file
out_dir = "results.62"  # where to dump results
save_sim = 20  # how often to save an image file
N = int(sys.argv[1])
start = int(sys.argv[2])
jitt_ref = False
n_jobs = 1
jid = 1
DET = sys.argv[3]
BEAM = sys.argv[4]
# -----------

def load_tracker_f(fname):
    data = []
    if os.path.exists(fname):
        data = np.loadtxt(fname, str)
        if data.size and not data.shape:
            data = list(set(data[None].astype(int)))
        else:
            data = list(set(data.astype(int)))
    return data

skip_weak = True
skip_failed = True
skip_indexed = True
weak_shots_f = os.path.join(out_dir, "weak_shots.txt")
failed_idx_f = os.path.join(out_dir, "failed_shots.txt")
indexed_f = os.path.join(out_dir, "indexed_shots.txt")
weak_shots = load_tracker_f(weak_shots_f)
failed_shots = load_tracker_f(failed_idx_f)
indexed_shots = load_tracker_f(indexed_f)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
spot_par_moder = deepcopy(spot_par)

spot_par.spotfinder.threshold.dispersion.global_threshold = 70.
spot_par.spotfinder.threshold.dispersion.gain = 28.
spot_par.spotfinder.threshold.dispersion.kernel_size = [4,4]
spot_par.spotfinder.threshold.dispersion.sigma_strong = 2.25
spot_par.spotfinder.threshold.dispersion.sigma_background =6.
spot_par.spotfinder.filter.min_spot_size = 2
spot_par.spotfinder.force_2d = True
spot_par.spotfinder.lookup.mask = mask_file

#spot_par_moder.spotfinder.threshold.dispersion.global_threshold = 56.
#spot_par_moder.spotfinder.threshold.dispersion.gain = 28.
#spot_par_moder.spotfinder.threshold.dispersion.kernel_size = [1,1]
#spot_par_moder.spotfinder.threshold.dispersion.sigma_strong = 2.5
#spot_par_moder.spotfinder.threshold.dispersion.sigma_background = 2.5
#spot_par_moder.spotfinder.filter.min_spot_size = 1
#spot_par_moder.spotfinder.force_2d = True
#spot_par_moder.spotfinder.lookup.mask = mask_file
#try_fft1d = False

loader = dxtbx.load(img_f)

#info_f = utils.open_flex("../index/run62_idx_processed.pkl")
#hit_idx = info_f.keys()

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [5000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e14, 1e14]  # fluxes of the beams

#from cxid9114.sim import scattering_factors
#Fcalcs = scattering_factors.get_scattF( parameters.WAVELEN_LOW,
#                                     pdb_name="../sim/4bs7.pdb",
#                                     algo='direct',
#                                     dmin=1.5, ano_flag=True)
# FF = [Fcalcs, None]

n_idx = 0
IMGSET = loader.get_imageset( img_f)
if N == -1:
    N = len(IMGSET)

for idx in range(start,N+start):
    if idx in weak_shots and skip_weak:
        print("Skipping weak shots %d" % idx)
        continue
    if idx in failed_shots and skip_failed:
        print("Skipping failed idx shots %d" % idx)
        continue
    if idx in indexed_shots and skip_indexed:
        print("Skipping already idx shots %d" % idx)
        continue

    iset = IMGSET[idx:idx+1] 
    iset.set_detector(DET)
    iset.set_beam(BEAM)
    #detector = iset.get_detector(0)
    
    dblock = DataBlockFactory.from_imageset(iset)[0]
    refls_strong = flex.reflection_table.from_observations(dblock, spot_par)

    if len(refls_strong) < 10:
        print("Not enough spots shot %d, continuing!" % idx)
        weak_shots.append(idx)
        try:
            np.savetxt(weak_shots_f, weak_shots, fmt="%d")
        except:
            pass
        continue
 
    waveA = parameters.ENERGY_CONV / ENERGIES[0]
    waveB = parameters.ENERGY_CONV / ENERGIES[1]

    beamA = deepcopy(iset.get_beam())
    beamB = deepcopy(iset.get_beam())

    beamA.set_wavelength(waveA)
    beamB.set_wavelength(waveB)

    isetA = deepcopy(iset)
    isetB = deepcopy(iset)
    isetA.set_beam(beamA)
    isetB.set_beam(beamB)

    ##if try_fft1d:
    #    sad_index_params.indexing.multiple_lattice_search.max_lattices = 1
    #    sad_index_params.indexing.stills.refine_all_candidates = False
    #    sad_index_params.indexing.stills.refine_candidates_with_known_symmetry = False
    #    sad_index_params.indexing.stills.candidate_outlier_rejection = False
    #    sad_index_params.indexing.stills.rmsd_min_px = 10
    #    sad_index_params.indexing.refinement_protocol.mode = "ignore"
    ##    # index two color pattern using fft1d
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
        print("####\nIndexingFailed  T_T \n####")
        failed_shots.append( idx)
        try:
            np.savetxt(failed_idx_f, failed_shots, fmt="%d")
        except:
            pass
        continue
    n_idx += 1
    #if not orientAB.refined_experiments.crystals():  # this would probably never happen...
    #    continue
    print("####\n *_* IndexingWORKED *_* \n####")
    crystalAB = orientAB.refined_experiments.crystals()[0]

    # identify the panels with the strong spots
    # for these will be used in refinement step below
    reflsPP = spot_utils.refls_by_panelname(refls_strong)
    pids = [i for i in reflsPP if len(reflsPP[i]) > 0 ]  # refine on these panels only
    pan_imgs = [iset.get_raw_data(0)[pid].as_numpy_array()
                for pid in pids]

    # helper wrapper for U-matrix grid search based refinement
    # `scanZ = ...` can also be passed as an argument, to jitter rotation
    # about the Z (beam) axis
    if not jitt_ref:
        cryst_model = crystalAB
    else:
        jitt_out = jitter_refine.jitter_panels(panel_ids=pids,  # we only simulate the pids with strong spots
                             crystal=crystalAB,
                             refls=refls_strong,
                             det=detector,
                             beam=iset.get_beam(0),
                             FF = FF,
                             en = ENERGIES,
                             data_imgs = pan_imgs,
                             flux = FLUX,
                             ret_best=False,
                             Ncells_abc=(10,10,10),
                             oversample=2,
                             Gauss=False,
                             mos_dom=2,
                             mos_spread=0.0,
                             scanX=np.arange(-.3, .3, .025),  # these seemed to be sufficient ranges
                             scanY=np.arange(-.3, .3, .025))

    # select the refined matrix based on overlap superposition
    # overlap is a metric used in the JitterFactory (wrapped into jitter_panels)
    # which checks agreement between data panels and simulated panels

        overlap = np.sum([jitt_out[pid]['overlaps'] for pid in jitt_out], axis=0)
        max_pos = np.argmax(overlap)
        optA = jitt_out[jitt_out.keys()[0]]["A_seq"][max_pos]  # just grab the first A_seq cause same sequence is tested on all panels

    # Now we have an indexing solution that is somewhat refined
    # make a new refined crystal
        optCrystal = deepcopy(crystalAB)
        optCrystal.set_A(optA)

    # TODO: include a metric to verify that optCrystal
    # is indeed optimized. I guess this is already done via the refinement overlap
    # but lets introduce another metric that evaluates overall
    # indexability of the solution
    # Simplest metrix is maybe the overal
    # residual HKL given either crystalAB or optCrystal

    #resAB,colorAB = metrics.likeliest_color_and_res(
    #    refls_strong, crystalAB,
    #    iset.get_detector(0),beamA, beamB)

    #sum_res_AB = np.mean([r for r in resAB if r is not None])
    #num_idx_AB = len([r for r in resAB if r is not None])

    #res_opt,color_opt = metrics.likeliest_color_and_res(
    #    refls_strong, optCrystal,
    #    iset.get_detector(0),beamA, beamB)
    #sum_res_opt = np.mean([r for r in res_opt if r is not None])
    #num_idx_opt = len([r for r in res_opt if r is not None])

    #if sum_res_opt < sum_res_AB:
    #    #  refinement worked as expected
    #    cryst_model = optCrystal
    #else:
    #    cryst_model = crystalAB
        cryst_model = optCrystal
    # end of that HKL testing

    # Optional secondary refinement.. Search a finer grid with smaller spots
    # starting from already refined position
    #jitt_out2 = jitter_refine.jitter_panels(panel_ids=pids,  # we only simulate the pids with strong spots
    #                     crystal=cryst_model,
    #                     refls=refls_strong,
    #                     det=detector,
    #                     beam=iset.get_beam(0),
    #                     FF = FF,
    #                     en = ENERGIES,
    #                     data_imgs = pan_imgs,
    #                     flux = FLUX,
    #                     ret_best=False,
    #                     Ncells_abc=(15,15,15),
    #                     oversample=2,
    #                     Gauss=False,
    #                     mos_dom=2,
    #                     mos_spread=0.0,
    #                     scanX=np.arange(-.1, .1, .01),  # these seemed to be sufficient ranges
    #                     scanY=np.arange(-.1, .1, .01))


    # now simulate the cryst_model
    # and we can use positions of the of the simulated pattern
    # to integrate the pixels on the camera
    # and set up the two color disentangler
    simsAB = sim_utils.sim_twocolors2(
        cryst_model, detector, iset.get_beam(0), [5000, None],
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        [1e14, 1e14], pids=None, Gauss=False, oversample=4,
        Ncells_abc=(20,20,20), mos_dom=20, mos_spread=0.0)  # returns a dict of {0: 64panelsim, 1: 64panelsim }

    if n_idx % save_sim ==0:
        sim_fname = os.path.join( out_dir, "sim64_%d.h5" % idx)
        sim_utils.save_twocolor(simsAB, iset, sim_fname, force=0)

    simsAB_2 = sim_utils.sim_twocolors2(
        crystalAB, detector, iset.get_beam(0), [5000, None],
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        [1e14, 1e14], pids=None, Gauss=False, oversample=4,
        Ncells_abc=(20,20,20), mos_dom=20, mos_spread=0.0)  # returns a dict of {0: 64panelsim, 1: 64panelsim }



    # Now, few things to do here:
    # The above simulation object has 128 images
    # 1 image per each of 64 asics per each of 2 colors
    # For each color, we will sum the single-color
    # images in order to obtain a color-composited
    # image akin to the measurement.
    # This will guide where we should integrate the raw data

    # In order to find the color-composited spot
    # we will apply a filter to smear any potentially
    # partially overlapping spots merge into a
    # single detectable region

    # The profiles on the single-color images will also be integrated
    # separately and used as parameters in the spot disentanglement
    # algorithm

    # We will run simple thresholding and connected region labeling
    # in order to do spot finding on each of the images

    # spot data on each colors simulated image
    spot_dataA = spot_utils.get_spot_data_multipanel(
        simsAB[0], detector=detector,
        beam=beamA, crystal=cryst_model, thresh=0,
        filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

    spot_dataB = spot_utils.get_spot_data_multipanel(
        simsAB[1],detector=detector,
        beam=beamB , crystal=cryst_model, thresh=0,
        filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

    d, dvecs, best = metrics.indexing_residuals_twocolor(spot_dataA, spot_dataB, refls_strong, detector)

    if jitt_ref:
        spot_dataA_2 = spot_utils.get_spot_data_multipanel(
            simsAB[0], detector=detector,
            beam=beamA, crystal=crystalAB, thresh=0,
            filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

        spot_dataB_2 = spot_utils.get_spot_data_multipanel(
            simsAB[1],detector=detector,
            beam=beamB , crystal=crystalAB, thresh=0,
            filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)
        d2, dvecs2, best2 = metrics.indexing_residuals_twocolor(spot_dataA_2, spot_dataB_2, refls_strong, detector)



    #HA, HiA = spot_utils.multipanel_refls_to_hkl(
    #    refls_strong,
    #    detector=iset.get_detector(0),
    #    beam=beamA, crystal=cryst_model)
    #HB, HiB = spot_utils.multipanel_refls_to_hkl(
    #    refls_strong,
    #    detector=iset.get_detector(0),
    #    beam=beamB, crystal=cryst_model)

    #treeA = cKDTree(spot_dataA["Q"])
    #treeB = cKDTree(spot_dataB["Q"])

    #reflsQA = spot_utils.multipanel_refls_to_q(refls_strong, iset.get_detector(0), beamA)
    #reflsQB = spot_utils.multipanel_refls_to_q(refls_strong, iset.get_detector(0), beamB)

    #dists = []
    #dist_vecs = []
    #for i,c in enumerate(color_opt):
    #    qA = reflsQA[i]
    #    qB = reflsQB[i]
    #    if c =="A":
    #        dist, iA = treeA.query(qA)
    #        dist_vec = qA - treeA.data[iA]
    #    elif c == "B":
    #        dist, iB = treeB.query(qB)
    #        dist_vec = qB - treeB.data[iB]
    #    else:
    #        continue
    #    dists.append( dist)
    #    dist_vecs.append( dist_vec)

    # spot data on composited color images

    #spot_data_combo = spot_utils.get_spot_data_multipanel(
    #    np.array(simsAB[0]) + np.array(simsAB[1]),
    #    detector=iset.get_detector(0),
    #    beam=iset.get_beam(0),
    #    crystal=cryst_model,
    #    thresh=0,
    #    filter=scipy.ndimage.filters.gaussian_filter, sigma=1)

    # save the crystal models and other data for later use and debugging

    reflsA = refls_strong.select(flex.bool(
        np.logical_and(best == 0, d/.10992 < 5)))

    reflsB = refls_strong.select(flex.bool(
        np.logical_and(best == 1, d/.10992 < 5)))

    eA = Experiment()
    eA.beam = beamA
    eA.detector = detector
    eA.imageset = iset
    eA.crystal = cryst_model

    eB = Experiment()
    eB.beam = beamB
    eB.detector = detector
    eB.imageset = iset
    eB.crystal = cryst_model

    EL = ExperimentList()
    EL.append(eA)
    EL.append(eB)

    exp_json = os.path.join(out_dir, "expAB_%d.json" % idx )
    reflA_pkl = os.path.join(out_dir, "reflA_%d.pkl" % idx)
    reflB_pkl = os.path.join(out_dir, "reflB_%d.pkl" % idx)
    orientAB.export_as_json( EL, exp_json)
    utils.save_flex(reflsA, reflA_pkl)
    utils.save_flex(reflsB, reflB_pkl)
    dump_pkl = os.path.join(out_dir, "dump_%d.pkl" % idx)


    
    t = loader.times[idx]  # event time
    sec, nsec, fid = t.seconds(), t.nanoseconds(), t.fiducial()
    t_num, _ = utils.make_event_time( sec, nsec, fid)

    dump = {"crystalAB": crystalAB,
             "event_time": t_num,
             "tsec": sec,"tnsec": nsec, "tfid": fid,
             #"res_opt": res_opt,
             #"color_opt": color_opt,
             #"resAB": resAB,
             #"colorAB": colorAB,
             "beamA": beamA,
             "beamB": beamB,
             "detector": detector,
             "spot_dataA": spot_dataA,
             "spot_dataB": spot_dataB,
             "d": d,
             "dvecs": dvecs,
             "best": best,
             "rmsd": orientAB.best_rmsd,
             #"dist_vecs": dist_vecs,
             #"dists": dists,
             #"spot_data_combo": spot_data_combo,
             "refls_strong": refls_strong}

    if jitt_ref:
        dump["crystalOpt"]= optCrystal
        dump["d2"] =  d2
        dump["overlap"] =  overlap
        dump["dvecs2"] =  dvecs2
        dump["best2"]= best2
    utils.save_flex(dump,  dump_pkl)


    # finding moderately-strong spots
    #refls_mod = flex.reflection_table.from_observations(dblock, spot_par_moder)
    #refls_modPP = spot_utils.refls_by_panelname(refls_mod)  # per panel

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
    indexed_shots.append(idx)
    try:
        np.savetxt(indexed_f, indexed_shots, fmt="%d")
    except:
        pass



