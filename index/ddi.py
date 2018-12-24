# DO.DATA.INDEXING.

# from dials.array_family import flex
import sys
import numpy as np
import cPickle

from dials.algorithms.indexing.indexer import master_phil_scope\
    as indexer_phil_scope
from dxtbx.model.beam import BeamFactory
try:
    from cxi_xdr_xes.two_color import two_color_indexer
    from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope
    HAS_TWO_COLOR = True
except ImportError:
    print "No two color indexer available"
    HAS_TWO_COLOR = False

from cctbx import crystal
import dxtbx
from cxid9114.spots import count_spots
from cxid9114 import parameters
from libtbx.utils import Sorry

if HAS_TWO_COLOR:
    indexer_phil_scope.adopt_scope(two_color_phil_scope)
    params = indexer_phil_scope.extract()
    MIN_SPOT_PER_HIT = 30

#   =====================================
#   Parameters
    show_hits = False
    INDEXER = two_color_indexer.indexer_two_color
    KNOWN_SYMMETRY = crystal.symmetry("78,78,37,90,90,90","P43212")
    two_color_indexer.N_UNIQUE_V = 20

    params.refinement.parameterisation.beam.fix = "all"
    params.refinement.parameterisation.detector.fix = "all"
    params.indexing.known_symmetry.space_group = KNOWN_SYMMETRY.space_group_info()
    params.refinement.verbosity = 3
# params.indexing.refinement_protocol.d_min_start
    params.indexing.known_symmetry.unit_cell = KNOWN_SYMMETRY.unit_cell()
    params.indexing.debug = True
    params.indexing.real_space_grid_search.characteristic_grid = 0.029
# params.indexing.stills.refine_all_candidates = False
    params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
    params.indexing.known_symmetry.relative_length_tolerance = 0.3
    params.indexing.two_color.high_energy = parameters.ENERGY_HIGH
    params.indexing.two_color.low_energy = parameters.ENERGY_LOW
    params.indexing.two_color.avg_energy = parameters.ENERGY_LOW * .5 + parameters.ENERGY_HIGH * .5
    params.indexing.stills.rmsd_min_px = 3.5
    params.indexing.refinement_protocol.n_macro_cycles = 1
    params.indexing.multiple_lattice_search.max_lattices = 100
# ====================================================================

    BEAM_LOW = BeamFactory.simple_directional((0, 0, 1), parameters.WAVELEN_LOW)
    BEAM_HIGH = BeamFactory.simple_directional((0, 0, 1), parameters.WAVELEN_HIGH)

if __name__ == "__main__":
    if not HAS_TWO_COLOR:
        print("Need to install the module cxi_xdr_xes")
        sys.exit()
    pickle_fname = sys.argv[1]
    image_fname = sys.argv[2]

    print('Loading reflections')
    with open(pickle_fname, 'r') as f:
        found_refl = cPickle.load(f)
    refl_select = count_spots.ReflectionSelect(found_refl)

    print('Loading format')
    loader = dxtbx.load(image_fname)
    imgset = loader.get_imageset(loader.get_image_file())

    print('Counting spots')
    idx, Nspot_at_idx = count_spots.count_spots(pickle_fname)
    where_hits = np.where( Nspot_at_idx > MIN_SPOT_PER_HIT )[0]
    Nhits = where_hits.shape[0]

    n_indexed = 0
    idx_indexed = []
    print('Iterating over {:d} hits'.format(Nhits))
    for i_hit in range( Nhits):
        shot_idx = idx[where_hits[i_hit]]
        if show_hits:
            loader.show_data(shot_idx)
        hit_imgset = imgset[shot_idx:shot_idx+1]
        hit_imgset.set_beam(BEAM_LOW)
        hit_refl = refl_select.select(shot_idx)

        print '\rIndexing shot {:d} (Hit {:d}/{:d}) using {:d} spots' \
            .format(shot_idx, i_hit+1, Nhits, len(hit_refl)),
        sys.stdout.flush()
        orient = INDEXER(reflections=hit_refl, imagesets=[hit_imgset], params=params)
        try:
            orient.index()
            n_indexed += 1
            idx_indexed.append( i_hit)
        except Sorry, RunTimeError:
            print("Could not index")
            pass
        print ("Indexed %d / %d hits"%(n_indexed, Nhits))
    print ("Indexed %d / %d hits"%(n_indexed, Nhits))

    print  idx_indexed
