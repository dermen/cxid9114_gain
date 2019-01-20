"""
Index Simulated Results 1-color!
"""

import dxtbx
from cxid9114.utils import images_and_refls_to_simview
from dials.array_family import flex
from dxtbx.datablock import DataBlockFactory
from dials.algorithms.indexing.indexer import indexer_base

from cxid9114.index.sad import params as sad_index_params
from cxid9114 import utils
from cxid9114.spots import count_spots
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse

# spot params:
find_spot_params = find_spots_phil_scope.fetch(source=parse("")).extract()
find_spot_params.spotfinder.threshold.dispersion.global_threshold = 0
find_spot_params.spotfinder.threshold.dispersion.sigma_strong = 0
find_spot_params.spotfinder.filter.min_spot_size = 1

import sys
output_pref = "1col_sims_5shots"
image_fname = sys.argv[1]
out_tag = sys.argv[2]

idx = 0
imgs = []
refls = []

loader = dxtbx.load(image_fname)
img = loader.get_raw_data(idx).as_numpy_array()

iset = loader.get_imageset(loader.get_image_file())
dblock = DataBlockFactory.from_imageset(iset[idx:idx+1])[0]
refl = flex.reflection_table.from_observations(dblock, find_spot_params)

imgs.append(img)
refls.append( refl)

info_fname = image_fname.replace(".h5", ".pkl")
sim_data = utils.open_flex(info_fname)

orient = indexer_base.from_parameters(
    reflections=count_spots.as_single_shot_reflections(refl, inplace=False),
    imagesets=[iset],
    params=sad_index_params)

try:
    orient.index()
    crystals = [o.crystal for o in orient.refined_experiments]
    rmsd =  orient.best_rmsd
    sim_data["sim_indexed_rmsd"] = rmsd
    sim_data["sim_indexed_crystals"] = crystals
    sim_data["sim_indexed_refls"] = refl
except:
    print("FAILED!")

utils.save_flex( sim_data, info_fname+out_tag)

images_and_refls_to_simview(output_pref, imgs, refls)
