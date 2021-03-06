"""
ssirp stands for sad spot index refine predict

basically a lightweight stills process to compare my
ultimate two color model to stills process

we will find spots and index with fft1d/stills
we will refine lattice with stills
then we will refine orientation with simtbx
then we will predict spots with simtbx
then we will integrate with custom
then we will do all in stills process and compare the results..

We do this for many shots

"""
import numpy as np
from cxid9114.sim import sim_utils
from cxid9114 import utils
from copy import deepcopy
from cxid9114 import parameters
from cxid9114.index.sad import params as sad_index_params
from libtbx.utils import Sorry
from cxid9114.spots import spot_utils
from dials.algorithms.indexing.indexer import indexer_base
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
import dxtbx
from dxtbx.datablock import DataBlockFactory
from dials.array_family import flex
import os
import sys

fcalc_f = "/Users/dermen/cxid9114_gain/sim/fcalc_slim.pkl"
#outdir = "ssirp_cell.beam"
#outdir = "

img_f = sys.argv[1]#  "xtc_102.loc"
outdir = os.path.join(sys.argv[2], os.path.basename(img_f).replace(".", "_"))
print outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)
BEAM = utils.open_flex(sim_utils.beam_f)
sad_wave = parameters.ENERGY_CONV / 8950
BEAM.set_wavelength(sad_wave)
MULTI_PANEL = True

spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
spot_par_moder = deepcopy(spot_par)

par1 = 0
mask_f = "../mask/dials_mask_64panels_2.pkl"

if par1:
    spot_par.spotfinder.threshold.dispersion.global_threshold = 60.
    spot_par.spotfinder.threshold.dispersion.gain = 28.
    spot_par.spotfinder.threshold.dispersion.kernel_size = [3,3]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 1.5
    spot_par.spotfinder.threshold.dispersion.sigma_background = 6.
    spot_par.spotfinder.filter.min_spot_size = 3
    spot_par.spotfinder.force_2d = True
    spot_par.spotfinder.lookup.mask = mask_f 

else:
    spot_par.spotfinder.threshold.dispersion.global_threshold = 70.
    spot_par.spotfinder.threshold.dispersion.gain = 28.
    spot_par.spotfinder.threshold.dispersion.kernel_size = [4,4]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 2.25
    spot_par.spotfinder.threshold.dispersion.sigma_background =6.
    spot_par.spotfinder.filter.min_spot_size = 2
    spot_par.spotfinder.force_2d = True
    spot_par.spotfinder.lookup.mask = mask_f

spot_par_moder.spotfinder.threshold.dispersion.global_threshold = 56.
spot_par_moder.spotfinder.threshold.dispersion.gain = 28.
spot_par_moder.spotfinder.threshold.dispersion.kernel_size = [1, 1]
spot_par_moder.spotfinder.threshold.dispersion.sigma_strong = 2.5
spot_par_moder.spotfinder.threshold.dispersion.sigma_background = 2.5
spot_par_moder.spotfinder.filter.min_spot_size = 1
spot_par_moder.spotfinder.force_2d = True
spot_par_moder.spotfinder.lookup.mask = mask_f
#spot_par_moder.spotfinder.lookup.mask = "../mask/dials_mask2d.pickle"

loader = dxtbx.load(img_f)


def load_tracker_f(fname):
    data = []
    if os.path.exists(fname):
        data = np.loadtxt(fname, str)
        if data.size and not data.shape:
            data = list(set(data[None].astype(int)))
        else:
            data = list(set(data.astype(int)))
    return data

skip_weak = False #True
skip_failed = False 
skip_indexed = False #True
weak_shots_f = os.path.join(outdir, "weak_shots.txt")
failed_idx_f = os.path.join(outdir, "failed_shots.txt")
indexed_f = os.path.join(outdir, "indexed_shots.txt")
weak_shots = load_tracker_f(weak_shots_f)
failed_shots = load_tracker_f(failed_idx_f)
indexed_shots = load_tracker_f(indexed_f)

IMGSET = loader.get_imageset(img_f)
DETECTOR = IMGSET.get_detector(0)

Nprocessed = 0
crystals = {}
N = len(IMGSET)  # number to process
for idx in range(N):
    if idx in weak_shots and skip_weak:
        print("Skipping weak shots %d" % idx)
        continue
    if idx in failed_shots and skip_failed:
        print("Skipping failed idx shots %d" % idx)
        continue
    if idx in indexed_shots and skip_indexed:
        print("Skipping already idx shots %d" % idx)
        continue

    iset = IMGSET[ idx:idx+1]
    iset.set_beam(BEAM)
    
    dblock = DataBlockFactory.from_imageset(iset)[0]
    refls_strong = flex.reflection_table.from_observations(dblock, spot_par)

    refls_moder = flex.reflection_table.from_observations(dblock, spot_par_moder)

    if len(refls_strong) < 10:
        print("Not enough spots shot %d, continuing!" % idx)
        weak_shots.append(idx)
        np.savetxt(weak_shots_f, weak_shots, fmt="%d")
        continue

    # Indexing parameters
    # ===================
    sad_index_params.indexing.multiple_lattice_search.max_lattices = 1
    sad_index_params.indexing.stills.refine_all_candidates = True
    sad_index_params.indexing.method = "fft1d"
    #sad_index_params.refinement.parameterisation.crystal.fix = "orientation"
    #sad_index_params.refinement.parameterisation.beam.fix = "wavelength"
    #sad_index_params.refinement.parameterisation.detector.fix = None
    #sad_index_params.refinement.parameterisation.detector.panels = "hierarchical"
    #sad_index_params.refinement.parameterisation.detector.hierarchy_level = 1
    sad_index_params.indexing.stills.refine_candidates_with_known_symmetry = True
    sad_index_params.indexing.stills.candidate_outlier_rejection = False
    sad_index_params.indexing.stills.rmsd_min_px = 8
    sad_index_params.indexing.debug = False
    sad_index_params.indexing.fft1d.characteristic_grid = 0.029
    #sad_index_params.indexing.refinement_protocol.mode = ""
    
    # index two color pattern using fft1d
    orient = indexer_base.from_parameters(
        reflections=spot_utils.as_single_shot_reflections(refls_strong, inplace=False),
        imagesets=[iset],
        params=sad_index_params)

    try:
        orient.index()
    except (Sorry, RuntimeError):
        print("\n\n\t INDEXING FAILED %d\n" % idx)
        failed_shots.append( idx)
        np.savetxt(failed_idx_f, failed_shots, fmt="%d")
        continue
    else:
        print("\n\n\t INDEXING  %d !!!\n" % idx)
        indexed_shots.append(idx)
        np.savetxt(indexed_f, indexed_shots, fmt="%d")



    exp_name = os.path.join(outdir, "exp_%d.json" % idx )
    refl_name = os.path.join(outdir, "refl_%d.pkl" % idx)
    orient.export_as_json(orient.refined_experiments, file_name=exp_name)
    utils.save_flex(orient.refined_reflections, refl_name)

    crystals[idx] = orient.refined_experiments.crystals()[0]
    Nprocessed += 1
    from IPython import embed
    embed()
utils.save_flex(crystals, os.path.join(outdir, "ssirp_cryst_r102.pkl"))

