"""
msi stands for mad --> spot --> index

Here we load a data image that has crystal diffraction
then we spot peaks and index the crystal image

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
from cxi_xdr_xes.two_color import two_color_indexer 
indexer_two_color = two_color_indexer.indexer_two_color
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
img_f = sys.argv[1]  # dxtbx format image file
out_dir = sys.argv[2]  # where to store outpur
start = int(sys.argv[3])  # first shot to process, then proceed 
N = int(sys.argv[4])  # number of shots to process
DET = utils.open_flex(sys.argv[5])  # path to pickled detector model
BEAM = utils.open_flex(sys.argv[6])  # '' '' beam model
tag = sys.argv[7]  # tag to attache to output files
# -----------

# track shots that indexed, or shots that 
# had too few spots to index, so can change parameters and try again
def load_tracker_f(fname):
    data = []
    if os.path.exists(fname):
        data = np.loadtxt(fname, str)
        if data.size and not data.shape:
            data = list(set(data[None].astype(int)))
        else:
            data = list(set(data.astype(int)))
    return data

skip_weak = False
skip_failed = False
skip_indexed = False
weak_shots_f = os.path.join(out_dir, "weak_shots.txt")
failed_idx_f = os.path.join(out_dir, "failed_shots.txt")
indexed_f = os.path.join(out_dir, "indexed_shots.txt")
weak_shots = load_tracker_f(weak_shots_f)
failed_shots = load_tracker_f(failed_idx_f)
indexed_shots = load_tracker_f(indexed_f)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# --- spotting parameters
spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
spot_par.spotfinder.threshold.dispersion.global_threshold = 70.
spot_par.spotfinder.threshold.dispersion.gain = 28.
spot_par.spotfinder.threshold.dispersion.kernel_size = [4,4]
spot_par.spotfinder.threshold.dispersion.sigma_strong = 2.25
spot_par.spotfinder.threshold.dispersion.sigma_background =6.
spot_par.spotfinder.filter.min_spot_size = 2
spot_par.spotfinder.force_2d = True
spot_par.spotfinder.lookup.mask = mask_file

# ------ indexing parameters
mad_index_params.indexing.two_color.spiral_method = (1.25, 1000000)
mad_index_params.indexing.two_color.n_unique_v = 22
mad_index_params.indexing.two_color.block_size = 25
mad_index_params.indexing.two_color.filter_by_mag = (10,3)
# ------

loader = dxtbx.load(img_f)

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
    print("####\n *_* IndexingWORKED %d *_* \n####" % idx)
    n_idx += 1
    indexed_shots.append(idx)
    try:
        np.savetxt(indexed_f, indexed_shots, fmt="%d")
    except:
        pass
    crystalAB = orientAB.refined_experiments.crystals()[0]
    
    E = Experiment()
    E.beam = BEAM
    E.detector = DET
    E.imageset = iset
    E.crystal = crystalAB
    EL = ExperimentList()
    EL.append(E)

    exp_json = os.path.join(out_dir, "exp_%d_%s.json" % (idx, tag) )
    refl_pkl = os.path.join(out_dir, "refl_%d_%s.pkl" % (idx, tag))
    orientAB.export_as_json( EL, exp_json)
    utils.save_flex(refls_strong, refl_pkl)

    t = loader.times[idx]  # event time
    sec, nsec, fid = t.seconds(), t.nanoseconds(), t.fiducial()
    t_num, _ = utils.make_event_time( sec, nsec, fid)
    
    beamA = deepcopy(iset.get_beam())
    beamB = deepcopy(iset.get_beam())
    waveA = parameters.ENERGY_CONV / mad_index_params.indexing.two_color.low_energy
    waveB = parameters.ENERGY_CONV / mad_index_params.indexing.two_color.high_energy
    beamA.set_wavelength(waveA)
    beamB.set_wavelength(waveB)

    dump = {"crystalAB": crystalAB,
            "refined_refls_v1": orientAB.refined_reflections,
             "event_time": t_num,
             "tsec": sec,"tnsec": nsec, "tfid": fid,
             "beamA": beamA,
             "beamB": beamB,
             "detector": DET,
             "rmsd": orientAB.best_rmsd,
             "refls_strong": refls_strong}

    dump_pkl = os.path.join(out_dir, "dump_%d_%s.pkl" % (idx, tag))
    utils.save_flex(dump,  dump_pkl)


