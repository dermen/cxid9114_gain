
import sys
import numpy as np
from copy import deepcopy

from scitbx.matrix import col
from cxid9114.sim import sim_utils
from cxid9114.spots import spot_utils
from cxid9114 import utils


def xyscan(crystal, fcalcs_energies, fcalcs, fracA, fracB, strong, rotxy_series, jid,
           mos_dom=2, mos_spread=0.05, flux=1e14, use_weights=False):
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
        sim_sig_mask = sim_patt > 0
        weights = sim_patt / sim_patt.max()
        if use_weights:
            overlaps.append( np.sum(sim_sig_mask * found_spot_mask * weights))
        else:
            overlaps.append( np.sum(sim_sig_mask * found_spot_mask))
        print "JOB %d" % jid
    return overlaps

def xyscan_multi(crystal, fcalcs_file, fracA, fracB,
                 strong, scan_params, n_jobs, scan_func=xyscan, prefix=None):
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

    nscans = len( scan_params)
    scan_idx = np.array_split(range( nscans), n_jobs)
    scan_split = []
    for idx in scan_idx:
        scan_split.append( [ scan_params[i] for i in idx ] )
    energy, fcalcs = sim_utils.load_fcalc_file(fcalcs_file)

    results = Parallel(n_jobs=n_jobs)(delayed(scan_func)\
                (crystal=crystal,
                 fcalcs_energies=energy,
                 fcalcs = fcalcs,
                 fracA=fracA, fracB=fracB,
                 strong=strong,jid=jid,
                 rotxy_series=scan_split[jid]) \
                for jid in range(n_jobs))
    results = np.concatenate(results)

    return results


if __name__ == "__main__":
    idx_proc_file = sys.argv[1]
    fcalc_f = sys.argv[2]
    output_prefix = sys.argv[3]
    Nprocess = int(sys.argv[4])
    n_jobs = int( sys.argv[5])
    simulate_refined = True
    degs = np.arange(-0.2, 0.25, 0.5)  # e.g.

    x = col((1., 0., 0.))
    y = col((0., 1., 0.))
    xRot = x.axis_and_angle_as_r3_rotation_matrix
    yRot = y.axis_and_angle_as_r3_rotation_matrix
    rotXY_series = [ (xRot(i, deg=True), yRot(j, deg=True))
                     for i in degs for j in degs ]

    data = utils.open_flex(idx_proc_file)
    from IPython import embed
    embed()
    hits_w_spec_ana = [i for i in data if data[i]['spectrum'] is not None
                 and data[i]['can_analyze']]
    if Nprocess > -1:
        hits_w_spec_ana = hits_w_spec_ana[:Nprocess]

    for hit_i in hits_w_spec_ana:
        crystal = data[hit_i]["crystals"][0]
        refl = data[hit_i]['refl']
        fracA = data[hit_i]['fracA']
        fracB = data[hit_i]['fracB']
        results = xyscan_multi( crystal,
                      fcalc_f,
                      fracA,
                      fracB,
                      data[hit_i]['refl'],
                      rotXY_series, n_jobs=6,
                      scan_func=xyscan)

        output_dump = {'results': results,
                       'degs': degs,
                       'rotXY_series': rotXY_series,
                       'fracA': fracA,
                       'fracB': fracB,
                       'crystal': crystal,
                       'strong_refl': refl,
                       'hit_idx': hit_i,
                       'fcalc_f': fcalc_f,
                       'idx_proc_file': idx_proc_file}

        output_file = "good_hit_%d_%s" % (output_prefix, hit_i)
        utils.save_flex(output_dump, output_file + ".pkl")

        if simulate_refined:
            sim_utils.simulate_xyscan_result(output_file, output_file)

