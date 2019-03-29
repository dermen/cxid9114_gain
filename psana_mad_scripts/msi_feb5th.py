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
img_f = "run120.loc"  # dxtbx format file
out_dir = "results.120"  # where to dump results
start = int(sys.argv[1])
N = int(sys.argv[2])
DET = utils.open_flex(sys.argv[3])
BEAM = utils.open_flex(sys.argv[4])
tag = sys.argv[5]
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

# ------
mad_index_params.indexing.two_color.spiral_method = (1.25, 1000000)
mad_index_params.indexing.two_color.n_unique_v = 20
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
    detector = iset.get_detector(0)
    
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
 
    waveA = parameters.ENERGY_CONV / 8944
    waveB = parameters.ENERGY_CONV / 9034.7

    beamA = deepcopy(iset.get_beam())
    beamB = deepcopy(iset.get_beam())

    beamA.set_wavelength(waveA)
    beamB.set_wavelength(waveB)

    isetA = deepcopy(iset)
    isetB = deepcopy(iset)
    isetA.set_beam(beamA)
    isetB.set_beam(beamB)

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

    dump = {"crystalAB": crystalAB,
             "event_time": t_num,
             "tsec": sec,"tnsec": nsec, "tfid": fid,
             "beamA": beamA,
             "beamB": beamB,
             "detector": detector,
             "rmsd": orientAB.best_rmsd,
             "refls_strong": refls_strong}

    dump_pkl= os.path.join(out_dir, "dump_%d_%s.pkl" % (idx, tag))
    utils.save_flex(dump,  dump_pkl)


