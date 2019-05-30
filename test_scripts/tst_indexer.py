
import os

from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model.experiment_list import ExperimentList
from cxid9114 import utils
from dials.array_family import flex
import dxtbx

# Write a temporary image file
img_filename ="_autogen.tst_indexer.txt"
img_file = \
    ("\n"
     "experiment = cxid9114\n"
     "run = 62\n"
     "mode = idx\n"
     "detector_address = \"CxiDs2.0:Cspad.0\"\n"
     "cspad  {\n"
     "    dark_correction = True\n"
     "    apply_gain_mask = True\n"
     "    detz_offset = 572.3938  # 572.9922 is the default\n"
     "    common_mode = default\n"
     "}\n"
     "d9114 {\n"
     "    common_mode_algo = pppg\n"
     "    savgol_polyorder = 3\n"
     "    mask = d9114_32pan_mask.npy\n"
     "}")
with open(img_filename,"w") as oid:
    oid.write(img_file)

loader = dxtbx.load(img_filename)
IMGSET = loader.get_imageset(img_filename)
from cxid9114 import mask
mask_file = os.path.dirname(mask.__file__) + "/dials_mask_64panels_2.pkl"
from cxid9114 import geom
geom_folder = os.path.dirname(geom.__file__)
DETECTOR = utils.open_flex(geom_folder + "/ref1_det.pkl")
BEAM = utils.open_flex(geom_folder + "/ref3_beam.pkl")

# --------- spot finding parameter
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from libtbx.phil import parse
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
from cxid9114.index.ddi import params as mad_index_params
mad_index_params.indexing.two_color.spiral_seed = 455
mad_index_params.indexing.two_color.spiral_method = (0.75, 250000)
mad_index_params.indexing.two_color.n_unique_v = 22
mad_index_params.indexing.two_color.block_size = 25
mad_index_params.indexing.two_color.filter_by_mag = (10, 3)
from cctbx import crystal
KNOWN_SYMMETRY = crystal.symmetry("79.1,79.1,38.5,90,90,90", "P43212")
#mad_index_params.indexing.refinement_protocol.mode = "repredict_only"
mad_index_params.indexing.refinement_protocol.mode = "ignore"

EXP_LIST = ExperimentList()
shot_indices = [0,1,4,8,17,19]  # shots to search for reflections
expected_Nref = [0,3,85,14,172,189]  # expected number of reflections

for idx in shot_indices:
    iset = IMGSET[idx:idx + 1]
    iset.set_detector(DETECTOR)
    iset.set_beam(BEAM)

    sub_EXP_LIST = ExperimentListFactory.from_stills_and_crystal(
        iset, crystal=None, load_models=True)

    EXP_LIST.extend(sub_EXP_LIST)

REFLS = []
CRYSTALS = []
RMSD = []

RMSD_MAXs = []

def tst_find_spots():

    global REFLS
    for i in range(len(EXP_LIST)):
        refls_strong = flex.reflection_table.from_observations(EXP_LIST[i:i+1], spot_par)
        Nrefls = len(refls_strong)
        print ("Found %d reflections on image %d; expected %d refls" % (Nrefls, i, expected_Nref[i]))

        assert(Nrefls == expected_Nref[i])
        REFLS.append( refls_strong)


def tst_index_spots():
    assert(REFLS)

    global CRYSTALS
    global RMSD
    from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
    from cxid9114.spots import spot_utils
    from libtbx.utils import Sorry

    for i in [2, 5]:  # NOTE: these are the experiments that should index
        refls_strong = REFLS[i]
        try:
            orientAB = indexer_two_color(
                reflections=spot_utils.as_single_shot_reflections(refls_strong, inplace=False),
                experiments=EXP_LIST[i:i+1],
                params=mad_index_params)
            orientAB.index()
        except (Sorry, RuntimeError) as error:
            print("Failed to index experiment %d:" % i)
            print(error)
            continue
        RMSD.append(orientAB.best_Amat_rmsd)
        CRYSTALS.append(orientAB.refined_experiments.crystals()[0])

    assert(len(RMSD) == 2)
    assert(all(
        [r < 3.75 for r in RMSD]))


def tst_make_param_list():

    from cxid9114.refine import jitter_refine

    param_list = jitter_refine.make_param_list(
        crystal=crystal,
        detector=DETECTOR,
        beam=BEAM,
        Nparam=20)

def tst_reindex():
    refls_strong = REFLS[2]
    from cxid9114.sim import sim_utils



if __name__ == "__main__":
    tst_find_spots()
    tst_index_spots()
    print("OK")
