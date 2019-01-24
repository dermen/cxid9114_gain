# DO.DATA.INDEXING.

# from dials.array_family import flex
import sys
import numpy as np
import cPickle

from dials.algorithms.indexing.indexer import master_phil_scope\
    as indexer_phil_scope


from dials.algorithms.indexing.fft1d import indexer_fft1d
from dials.algorithms.indexing import stills_indexer
from dials.algorithms.indexing.indexer import indexer_base
from cctbx import crystal
import dxtbx
from cxid9114.spots import count_spots
from libtbx.utils import Sorry

params = indexer_phil_scope.extract()
MIN_SPOT_PER_HIT = 3

#   =====================================
#   Parameters
show_hits = False
INDEXER = stills_indexer.stills_indexer
KNOWN_SYMMETRY = crystal.symmetry("79,79,38,90,90,90","P43212")
params.refinement.parameterisation.beam.fix = "all"
params.refinement.parameterisation.detector.fix = "all"
params.indexing.method = "fft1d"
params.indexing.known_symmetry.space_group = KNOWN_SYMMETRY.space_group_info()
params.refinement.verbosity = 3
# params.indexing.refinement_protocol.d_min_start
params.indexing.known_symmetry.unit_cell = KNOWN_SYMMETRY.unit_cell()
params.indexing.debug = True
params.indexing.real_space_grid_search.characteristic_grid = 0.029
# params.indexing.stills.refine_all_candidates = False
params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
params.indexing.known_symmetry.relative_length_tolerance = 0.3
params.indexing.stills.rmsd_min_px = 2
params.indexing.refinement_protocol.n_macro_cycles = 1
params.indexing.multiple_lattice_search.max_lattices = 20
params.indexing.basis_vector_combinations.max_refine = 5
params.indexing.stills.indexer = 'stills'
# ====================================================================

if __name__ == "__main__":
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
    where_hits = np.where(Nspot_at_idx > MIN_SPOT_PER_HIT)[0]
    Nhits = where_hits.shape[0]

    n_indexed = 0
    idx_indexed = []
    print('Iterating over {:d} hits'.format(Nhits))
    for i_hit in range(Nhits):
        shot_idx = idx[where_hits[i_hit]]
        if show_hits:
            loader.show_data(shot_idx)
        hit_imgset = imgset[shot_idx:shot_idx + 1]
        hit_refl = refl_select.select(shot_idx)

        print '\rIndexing shot {:d} (Hit {:d}/{:d}) using {:d} spots' \
            .format(shot_idx, i_hit + 1, Nhits, len(hit_refl)),
        sys.stdout.flush()
        # orient = INDEXER(reflections=hit_refl, imagesets=[hit_imgset], params=params)
        #orient = indexer_base.from_parameters(reflections=hit_refl,
        #                                      imagesets=[hit_imgset],
        #                                      params=params)
        orient = INDEXER(reflections=hit_refl,
                          imagesets=[hit_imgset],
                          params=params)
        try:
            orient.index()
            n_indexed += 1
            idx_indexed.append(i_hit)
        except (Sorry, RuntimeError, AssertionError):
            print("Could not index")
            pass
        print ("Indexed %d / %d hits, out of %d total hits" % (n_indexed, i_hit +1, Nhits))
    print ("Indexed %d / %d hits" % (n_indexed, Nhits))

    print idx_indexed
