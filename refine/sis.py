"""
sis stands for Simulate --> Index --> Simulate

Here we simulate a crystal with simtbx
then we index the crystal with two color code
Then re-simulate the indexed crystal

"""
from cxid9114.sim import sim_utils
from cxid9114 import utils
from dials.array_family import flex
from dials.algorithms.indexing.indexer import indexer_base
from copy import deepcopy
from cxid9114 import parameters

from cxid9114.index.sad import params as sad_index_params
from cxid9114.index.ddi import params as mad_index_params

from cxid9114.spots import count_spots
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
from cxid9114.refine.test_sim_overlapper import  plot_overlap

find_spot_params = find_spots_phil_scope.fetch(source=parse("")).extract()
find_spot_params.spotfinder.threshold.dispersion.global_threshold = 0
find_spot_params.spotfinder.threshold.dispersion.sigma_strong = 0
find_spot_params.spotfinder.filter.min_spot_size = 1

# load a test crystal
crystal = utils.open_flex( sim_utils.cryst_f )

# fraction of two color energy
fracA = 0.5
fracB = 0.5

# simulate the pattern
Patts = sim_utils.PatternFactory()
en, fcalc = sim_utils.load_fcalc_file("../sim/fcalc_slim.pkl")
flux = [fracA * 1e14, fracB * 1e14]
imgA, imgB = Patts.make_pattern2(crystal, flux, en, fcalc, 20, 0.1, False)

# here we can index each 1 color pattern as well as the two color pattern
# for fun

waveA = parameters.ENERGY_CONV / en[0]
waveB = parameters.ENERGY_CONV / en[1]
BeamA = Patts.beam
BeamA.set_wavelength(waveA)
BeamB = deepcopy(BeamA)
BeamB.set_wavelength(waveB)

dblockA = utils.datablock_from_numpyarrays(image=imgA, detector=Patts.detector, beam=BeamA)
dblockB = utils.datablock_from_numpyarrays(image=imgB, detector=Patts.detector, beam=BeamB)
dblockAB = utils.datablock_from_numpyarrays(image=fracB*imgB+fracA*imgA, detector=Patts.detector, beam=BeamA)
dblockAB_2 = utils.datablock_from_numpyarrays(image=fracB*imgB+fracA*imgA, detector=Patts.detector, beam=BeamB)

reflA = flex.reflection_table.from_observations(dblockA, find_spot_params)
reflB = flex.reflection_table.from_observations(dblockB, find_spot_params)
reflAB = flex.reflection_table.from_observations(dblockAB, find_spot_params)

from IPython import embed
embed()

isetsA = dblockA.extract_imagesets()
isetsB = dblockB.extract_imagesets()
isetsAB = dblockAB.extract_imagesets()
isetsAB_2 = dblockAB_2.extract_imagesets()


# ==================================
# 2 color indexer of 2 color pattern
# ==================================
sad_index_params.indexing.multiple_lattice_search.max_lattices = 1
sad_index_params.indexing.stills.refine_all_candidates = False
sad_index_params.indexing.stills.refine_candidates_with_known_symmetry = False
sad_index_params.indexing.stills.candidate_outlier_rejection = False
sad_index_params.indexing.refinement_protocol.mode = "ignore"
sad_index_params2 = deepcopy(sad_index_params)
#sad_index_params2.indexing.stills.refine_all_candidates = False
#sad_index_params2.indexing.stills.refine_candidates_with_known_symmetry = False
#sad_index_params2.indexing.stills.candidate_outlier_rejection = False
#sad_index_params2.indexing.refinement_protocol.mode = "ignore"

orientAB_fft1d = indexer_base.from_parameters(
    reflections=count_spots.as_single_shot_reflections(reflAB, inplace=False),
    imagesets=isetsAB,
    params=sad_index_params2)
orientAB_fft1d.index()
crystal_AB_fft1d = orientAB_fft1d.refined_experiments.crystals()[0]

orientAB_2_fft1d = indexer_base.from_parameters(
    reflections=count_spots.as_single_shot_reflections(reflAB, inplace=False),
    imagesets=isetsAB_2,
    params=sad_index_params2)
orientAB_2_fft1d.index()
crystalAB_2_fft1d = orientAB_2_fft1d.refined_experiments.crystals()[0]



# index the 1 color patterns:
orientA = indexer_base.from_parameters(
    reflections=count_spots.as_single_shot_reflections(reflA, inplace=False),
    imagesets=isetsA,
    params=sad_index_params)
orientA.index()
crystal_A = orientA.refined_experiments.crystals()[0]




orientB = indexer_base.from_parameters(
    reflections=count_spots.as_single_shot_reflections(reflB, inplace=False),
    imagesets=isetsB,
    params=sad_index_params)
orientB.index()
crystal_B = orientB.refined_experiments.crystals()[0]


# ==================================
# 2 color indexer of 2 color pattern
# ==================================
orientAB = indexer_two_color(
    reflections=count_spots.as_single_shot_reflections(reflAB, inplace=False),
    imagesets=isetsAB,
    params=mad_index_params)
orientAB.index()
crystal_AB = orientAB.refined_experiments.crystals()[0]

from IPython import embed
embed()
