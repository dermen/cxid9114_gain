"""
dis stands for Data --> Index --> Simulate

Here we load a data image that has crystal diffraction
then we index the crystal image using different methods
Then simulate the indexed crystal

"""
from cxid9114.sim import sim_utils
from cxid9114 import utils
from dials.array_family import flex
from dials.algorithms.indexing.indexer import indexer_base
from copy import deepcopy
from cxid9114 import parameters

from cxid9114.index.sad import params as sad_index_params
from cxid9114.index.ddi import params as mad_index_params

from cxid9114.spots import spot_utils
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
from cxid9114.refine.test_sim_overlapper import  plot_overlap

find_spot_params = find_spots_phil_scope.fetch(source=parse("")).extract()
find_spot_params.spotfinder.threshold.dispersion.global_threshold = 0
find_spot_params.spotfinder.threshold.dispersion.sigma_strong = 0
find_spot_params.spotfinder.filter.min_spot_size = 1

import dxtbx
from dxtbx.datablock import DataBlockFactory

img_f = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
loader = dxtbx.load(img_f)

info_f = utils.open_flex("../index/run62_idx_processed.pkl")
hit_idx = info_f.keys()
idx = hit_idx[0]
iset = loader.get_imageset( img_f)[ idx:idx+1]
dblock = DataBlockFactory.from_imageset(iset) [0]

refls = info_f[idx]['refl']
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

# here we can index each 1 color pattern as well as the two color pattern
# for fun

# ==================================
# 2 color indexer of 2 color pattern
# ==================================
sad_index_params.indexing.multiple_lattice_search.max_lattices = 1
sad_index_params.indexing.stills.refine_all_candidates = True #False
sad_index_params.indexing.stills.refine_candidates_with_known_symmetry = True #False
sad_index_params.indexing.stills.candidate_outlier_rejection = True # False
sad_index_params.indexing.stills.rmsd_min_px = 10
#sad_index_params.indexing.refinement_protocol.mode = "ignore"
#sad_index_params2.indexing.stills.refine_all_candidates = False
#sad_index_params2.indexing.stills.refine_candidates_with_known_symmetry = False
#sad_index_params2.indexing.stills.candidate_outlier_rejection = False
#sad_index_params2.indexing.refinement_protocol.mode = "ignore"
waveA = parameters.ENERGY_CONV / en[0]
waveB = parameters.ENERGY_CONV / en[1]

BeamA = deepcopy( iset.get_beam())
BeamB = deepcopy( iset.get_beam())

BeamA.set_wavelength(waveA)
BeamB.set_wavelength(waveB)

isetA = deepcopy(iset)
isetB = deepcopy( iset)
isetA.set_beam(BeamA)
isetB.set_beam(BeamB)

# index two color pattern using fft1d
orientA = indexer_base.from_parameters(
    reflections=spot_utils.as_single_shot_reflections(refls, inplace=False),
    imagesets=[isetA],
    params=sad_index_params)
orientA.index()
crystalA = orientA.refined_experiments.crystals()[0]

# try with other color
orientB = indexer_base.from_parameters(
    reflections=spot_utils.as_single_shot_reflections(refls, inplace=False),
    imagesets=[isetB],
    params=sad_index_params)
orientB.index()
crystalB = orientB.refined_experiments.crystals()[0]

# ==================================
# 2 color indexer of 2 color pattern
# ==================================
orientAB = indexer_two_color(
    reflections=spot_utils.as_single_shot_reflections(refls, inplace=False),
    imagesets=[iset],
    params=mad_index_params)
orientAB.index()
crystalAB = orientAB.refined_experiments.crystals()[0]

from IPython import embed
embed()

