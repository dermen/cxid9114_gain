# DO.DATA.INDEXING.

from dials.algorithms.indexing.indexer import master_phil_scope\
    as indexer_phil_scope
from cxi_xdr_xes.two_color import two_color_indexer
from cxi_xdr_xes.two_color
from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope
from cctbx import crystal

indexer_phil_scope.adopt_scope(two_color_phil_scope)
params = indexer_phil_scope.extract()

ENERGY_LOW = 8944.
ENERGY_HIGH = 9034.7
INDEXER = two_color_indexer.indexer_two_color
KNOWN_SYMMETRY = crystal.symmetry("78,78,37,90,90,90","P43212")
two_color_indexer.N_UNIQUE_V = 20

params.refinement.parameterisation.beam.fix = "all"
params.refinement.parameterisation.detector.fix = "all"
params.indexing.KNOWN_SYMMETRY.space_group = KNOWN_SYMMETRY.space_group_info()
params.refinement.verbosity = 3
# params.indexing.refinement_protocol.d_min_start
params.indexing.refinement_protocol.n_macro_cycles = 1
params.indexing.KNOWN_SYMMETRY.unit_cell = KNOWN_SYMMETRY.unit_cell()
params.indexing.debug = True
params.indexing.real_space_grid_search.characteristic_grid = 0.015
params.indexing.stills.refine_all_candidates = False
params.indexing.KNOWN_SYMMETRY.absolute_angle_tolerance = 5.0
params.indexing.KNOWN_SYMMETRY.relative_length_tolerance = 0.3
params.indexing.two_color.high_energy = ENERGY_HIGH
params.indexing.two_color.low_energy = ENERGY_LOW
params.indexing.two_color.avg_energy = ENERGY_LOW * .5 + ENERGY_HIGH * .5
params.indexing.stills.rmsd_min_px = 3.5

# iset_data = ImageSetData(reader, masker)
# imgset = ImageSet(iset_data)
# imgset.set_beam(beams[0])
# imgset.set_detector(detector)
# imagesets = [imgset]

#orient = indexer_two_color(reflections, imagesets, params)
#orient.index()
