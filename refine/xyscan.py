
import sys
import numpy as np
from copy import deepcopy
import os
from scitbx.matrix import col, sqr
from cxid9114.sim import sim_utils
from cxid9114.spots import spot_utils
from cxid9114 import utils
import dxtbx
from dials.array_family import flex

from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse

find_spot_params = find_spots_phil_scope.fetch(source=parse("")).extract()
find_spot_params.spotfinder.threshold.dispersion.global_threshold = 0
find_spot_params.spotfinder.threshold.dispersion.sigma_strong = 0.5
find_spot_params.spotfinder.filter.min_spot_size = 3



def xyscan(crystal, fcalcs_energies, fcalcs, fracA, fracB, strong, rotxy_series, jid,
           mos_dom=1, mos_spread=0.05, flux=1e14, use_weights=False, raw_image=None):
    """
    :param crystal:
    :param fcalcs_energies:
    :param fcalcs:
    :param fracA:
    :param fracB:
    :param strong:
    :param rotxy_series:
    :param jid:
    :param mos_dom:
    :param mos_spread:
    :param flux:
    :param use_weights:
    :return:
    """
    Patts = sim_utils.PatternFactory()
    Patts.adjust_mosaicity(mos_dom, mos_spread)  # defaults
    flux_per_en = [fracA*flux, fracB*flux]

    img_size = Patts.detector.to_dict()['panels'][0]['image_size']
    found_spot_mask = spot_utils.strong_spot_mask(strong, img_size)
    #if use_weights:
    #    if raw_image is None:
    #        raise ValueError("Cant use weights if raw image is None")
    #    spot_signals = raw_image * found_spot_mask
    #    weights = spot_signals /  spot_signals.max()

    overlaps = []
    for rots in rotxy_series:
        if len(rots)==2:
            Op = rots[0]*rots[1]
        elif len(rots)==3:
            Op = rots[0]*rots[1]*rots[2]
        elif len(rots==1):
            Op = rots[0]
        else:
            raise ValueError("rotxy_series should be list of 1,2, or 3-tuples")
        sim_patt = Patts.make_pattern2(crystal=deepcopy(crystal),
                                       flux_per_en=flux_per_en,
                                       energies_eV=fcalcs_energies,
                                       fcalcs_at_energies=fcalcs,
                                       mosaic_spread=None,
                                       mosaic_domains=None,
                                       ret_sum=True,
                                       Op=Op)


        if use_weights:
            dblock = utils.datablock_from_numpyarrays(
                image=sim_patt,
                detector=Patts.detector,
                beam=Patts.beam)
            sim_refl = flex.reflection_table.from_observations(dblock, params=find_spot_params)
            sim_spot_mask = spot_utils.strong_spot_mask(sim_refl, sim_patt.shape)
            overlaps.append( np.sum( sim_spot_mask * found_spot_mask) )
            #weights2 = sim_patt / sim_patt.max()
            #overlaps.append( np.sum(sim_sig_mask * found_spot_mask * weights * weights2))
        else:
            sim_sig_mask = sim_patt > 0
            overlaps.append( np.sum(sim_sig_mask * found_spot_mask))
        print "JOB %d" % jid
    return overlaps

def xyscan_multi(crystal, fcalcs_file, fracA, fracB,
                 strong, rot_series, n_jobs, scan_func=xyscan,
                 use_weights=False, raw_image=None):
    """
    :param crystal_file:
    :param energies:
    :param fcalcs_file:
    :param fracA:
    :param fracB:
    :param strong_file:
    :param rotxy_series:
    :param n_jobs:
    :return:
    """
    from joblib import Parallel, delayed

    nscans = len( rot_series)
    scan_idx = np.array_split(range( nscans), n_jobs)
    scan_split = []
    for idx in scan_idx:
        scan_split.append( [ rot_series[i] for i in idx ] )
    energy, fcalcs = sim_utils.load_fcalc_file(fcalcs_file)

    results = Parallel(n_jobs=n_jobs)(delayed(scan_func)\
                (crystal=crystal,
                 fcalcs_energies=energy,
                 fcalcs = fcalcs,
                 fracA=fracA, fracB=fracB,
                 strong=strong,jid=jid,
                 use_weights=use_weights, raw_image=raw_image,
                 rotxy_series=scan_split[jid]) \
                for jid in range(n_jobs))
    results = np.concatenate(results)

    return results

def fine_xyrefine(refine_xy_file, n_jobs=6, raw_image=None, use_weights=False):
    scan_data = utils.open_flex(refine_xy_file)

    crystal = deepcopy(scan_data['optCrystal'])
    refl = scan_data['refl']
    x = col((1,0,0))
    y = col((0,1,0))

    xR = x.axis_and_angle_as_r3_rotation_matrix
    yR = y.axis_and_angle_as_r3_rotation_matrix

    fine_degs = np.arange( -0.045, 0.045, 0.005)
    rotXY_series = [(xR(i,deg=True), yR(j,deg=True))
                    for i in fine_degs
                    for j in fine_degs]

    fracA = int(scan_data["fracA"])
    fracB = int(scan_data["fracB"])
    fcalc_f = scan_data["fcalc_f"]

    results = xyscan_multi(crystal,
                       fcalc_f,
                       fracA,
                       fracB,
                       refl,
                       rotXY_series, n_jobs=n_jobs,
                       scan_func=xyscan, use_weights=use_weights, raw_image=raw_image)

    optX_fine, optY_fine = rotXY_series[np.argmax(results)]
    optA_fine = optX_fine * optY_fine * sqr(crystal.get_U()) * sqr(crystal.get_B())
    return results, optX_fine, optY_fine, optA_fine


if __name__ == "__main__":
    # ================
    # PARAMETERS ENTRY
    # ================
    idx_proc_file = sys.argv[1]
    fcalc_f = sys.argv[2]
    output_prefix = sys.argv[3]
    Nprocess = int(sys.argv[4])
    n_jobs = int( sys.argv[5])
    simulate_refined = True
    degs = np.arange(-0.2, 0.2, 0.025)
    fine_scan = False
    use_weights = False
    # ======================
    # END OF PARAMETER ENTRY
    # ======================
    x = col((1., 0., 0.))
    y = col((0., 1., 0.))
    xRot = x.axis_and_angle_as_r3_rotation_matrix
    yRot = y.axis_and_angle_as_r3_rotation_matrix
    rotXY_series = [ (xRot(i, deg=True), yRot(j, deg=True))
                     for i in degs for j in degs ]

    loader = dxtbx.load("/Users/dermen/cxid9114/run62_hits_wtime.h5")

    data = utils.open_flex(idx_proc_file)

    hits_w_spec_ana = [i for i in data if data[i]['spectrum'] is not None
                 and data[i]['can_analyze']]
    if Nprocess > -1:
        hits_w_spec_ana = hits_w_spec_ana[:Nprocess]

    for hit_i in hits_w_spec_ana:
        crystal = data[hit_i]["crystals"][0]
        refl = data[hit_i]['refl']
        fracA = data[hit_i]['fracA']
        fracB = data[hit_i]['fracB']
        raw_img = loader.get_raw_data( hit_i).as_numpy_array()

        results = xyscan_multi( crystal,
                      fcalc_f,
                      fracA,
                      fracB,
                      data[hit_i]['refl'],
                      rotXY_series, n_jobs=n_jobs,
                      scan_func=xyscan, raw_image=raw_img, use_weights=use_weights)

        max_pos = np.argmax( results)
        optX, optY = rotXY_series[max_pos]

        optA = optX * optY * sqr(crystal.get_U()) * sqr(crystal.get_B())
        optCrystal = deepcopy(crystal)
        optCrystal.set_A(optA)

        output_dump = {'results': results,
                       'degs': degs,
                       'rotXY_series': rotXY_series,
                       'fracA': fracA,
                       'fracB': fracB,
                       'crystal': crystal,
                       'refl': refl,
                       'optX': optX,
                       'optY': optY,
                       'optCrystal': optCrystal,
                       'hit_idx': hit_i,
                       'fcalc_f': fcalc_f,
                       'idx_proc_file': idx_proc_file}

        output_dir = os.path.dirname(output_prefix)
        output_base = os.path.basename(output_prefix)
        output_file = os.path.join( output_dir, "good_hit_%d_%s" % (hit_i, output_base))

        output_pkl = output_file + ".pkl"
        utils.save_flex(output_dump, output_pkl)

        if fine_scan:
            results, fineX, fineY, fineA = fine_xyrefine(output_pkl, n_jobs=n_jobs, use_weights=use_weights,
                                                  raw_image=raw_img)

            #optA_fine = fineX * fineY * optX * optY * sqr(crystal.get_U()) * sqr(crystal.get_B())
            optCrystal_fine = deepcopy(crystal)
            optCrystal_fine.set_A(fineA)

            output_dump['fine_scan_results'] = results
            output_dump['optX_fine'] = fineX
            output_dump['optY_fine'] = fineY
            output_dump["optCrystal_fine"] = optCrystal_fine
            utils.save_flex(output_dump, output_pkl)

        if simulate_refined:
            sim_utils.simulate_xyscan_result(output_pkl, output_file)


