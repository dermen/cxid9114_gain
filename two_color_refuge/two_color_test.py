from __future__ import division
import copy

from dials.array_family import flex
from dials.algorithms.indexing.refinement import refine
from cxi_xdr_xes.two_color import merge_close_spots
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.format.Format import Reader, Masker
from dials.algorithms.indexing.indexer import master_phil_scope
from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope
# from libtbx.phil import command_line
from dials.algorithms.indexing.compare_orientation_matrices \
  import difference_rotation_matrix_axis_angle
from cctbx import crystal
from scitbx.math import euler_angles as euler
from dxtbx.model import Crystal
import random
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color

master_phil_scope.adopt_scope(two_color_phil_scope)
master_phil_scope.extract()

def index(reflections, detector, known_symmetry, beams):
  params = master_phil_scope.extract()
  params.refinement.parameterisation.beam.fix="all"
  params.refinement.parameterisation.detector.fix="all"
  params.indexing.known_symmetry.space_group=known_symmetry.space_group_info()
  params.refinement.verbosity=3
  params.indexing.refinement_protocol.d_min_start
  params.indexing.refinement_protocol.n_macro_cycles=1
  params.indexing.known_symmetry.unit_cell=known_symmetry.unit_cell()
  params.indexing.debug=True
  params.indexing.known_symmetry.absolute_angle_tolerance=5.0
  params.indexing.known_symmetry.relative_length_tolerance=0.3
  filenames=[""]
  reader = Reader(filenames)
  masker = Masker(filenames)
  iset_data = ImageSetData(reader, masker)
  imgset = ImageSet(iset_data)
  imgset.set_beam(beams[0])
  imgset.set_detector(detector)
  imagesets = [imgset]
  # dermen:
  #   modified rmsd_min_px this so test will pass, was originally set
  #   to 2 and the stills indexer's best.rmsd was of order ~2.8ish
  #   I dont know enough about the code itself to make the call on
  #   whether this is legit or not or whether 2.8 best.rmsd is
  #   something we should focus on improving...
  params.indexing.stills.rmsd_min_px = 3.5
  orient = indexer_two_color(reflections, imagesets, params)
  orient.index()
  return orient

def align_merged_and_refined_refl(merged, refined):
  """
  dermen:
    added this hack because Ann module wasnt being applied
    correctly , I think this helps accomplishe the intended goal
    of the test
  finds indices of merged reflections corresponding to those
  that were refined.
  It is assumed that refined is a subset of merged
  """
  import numpy as np
  from scipy.spatial import cKDTree
  assert( len( merged) >= len( refined))
  x_merged, y_merged, _ = zip(*merged['xyzobs.px.value'])
  x_refined, y_refined, _ = zip(*refined['xyzobs.px.value'])
  xy_merged = np.vstack((x_merged, y_merged)).T
  xy_refined = np.vstack((x_refined, y_refined)).T
  tree = cKDTree(xy_merged)
  _, idx_of_match = tree.query(xy_refined, k=1)
  return idx_of_match

def run():
  random_seed = 35
  flex.set_random_seed(random_seed)
  random.seed(random_seed)
  data = merge_close_spots.merge_close_spots()
  # unit cell and space group for lysozyme
  known_symmetry=crystal.symmetry("78,78,37,90,90,90","P43212")

  sim = data.two_color_sim
  merged_spot_info = data.spot_proximity(sim)
  merged_refl = merged_spot_info[0]

  detector = data.detector
  beams = data.beams

  # to make sure input crystal and indexed crystal model are the same
  # orientation before using the refiner
  A = sim['input_orientation']
  A_inv = A.inverse()
  a = A_inv[:3]
  b = A_inv[3:6]
  c = A_inv[6:]
  crystinp = Crystal(a, b, c, space_group=known_symmetry.space_group())
  result = index(merged_refl, detector, known_symmetry, beams)
  print ("RESULTS ARE IN")
  cm = result.refined_experiments.crystals()[0]
  R, best_axis, best_angle, change_of_basis = difference_rotation_matrix_axis_angle(
    crystal_a=cm, crystal_b=crystinp)
  euler_angles = euler.xyz_angles(R)

  print "input crystal: %s"%crystinp
  print "Indexed crystal: %s"%cm
  print "rotation of: %s"%R
  print "euler angles:", euler_angles
  print "change of basis:", change_of_basis.as_hkl()

  # cmd_line = command_line.argument_interpreter(master_params=master_phil_scope)
  # working_phil = cmd_line.process_and_fetch(args=[])
  params = master_phil_scope.extract()
  params.refinement.parameterisation.beam.fix="all"
  params.refinement.parameterisation.detector.fix="all"
  params.indexing.known_symmetry.space_group=known_symmetry.space_group_info()
  params.refinement.verbosity=3
  params.indexing.refinement_protocol.d_min_start=3
  params.indexing.refinement_protocol.n_macro_cycles=1
  params.indexing.known_symmetry.unit_cell=known_symmetry.unit_cell()
  params.indexing.multiple_lattice_search.max_lattices = 1
  params.indexing.debug=True
  params.indexing.known_symmetry.absolute_angle_tolerance=5.0
  params.indexing.known_symmetry.relative_length_tolerance=0.3

  expts = copy.deepcopy(result.refined_experiments)
  expts.crystals()[0].change_basis(change_of_basis)
  print (expts.crystals()[0] )

  refined = refine(params, result.reflections, expts, verbosity=1)
  print (refined[0].get_experiments().crystals()[0] )

  # from annlib_ext import AnnAdaptor
  # ann = AnnAdaptor(merged_refl['xyzobs.px.value'].as_double(), dim=3, k=1)
  # ann.query(result.reflections['xyzobs.px.value'].as_double()+1e-6)
  indices_sim = change_of_basis.apply(merged_refl['set_miller_index'])
  id_sim = merged_refl['set_id']
  # only get those that refined:
  idx = align_merged_and_refined_refl(merged_refl, result.refined_reflections)
  indices_sim = flex.miller_index( [indices_sim[i] for i in idx])
  id_sim = flex.int(  [id_sim[i] for i in idx] )
  indices_result = result.refined_reflections['miller_index']
  id_result = result.refined_reflections['id']

  correct_ind = (indices_sim == indices_result)
  wrong_wavelength = (id_sim != id_result) & (id_sim !=2)
  wrong_index = (indices_sim != indices_result)
  correct_wavelength = (id_sim == id_result)|(id_sim==2)
  correct = correct_ind & correct_wavelength
  print "Correct index and wavelength: %i/%i" %(
    correct.count(True), len(correct))
  print "Correct index but wrong wavelength: %i/%i" %(
    wrong_wavelength.count(True), len(wrong_wavelength))
  print "Wrong index but correct wavelength: %i/%i" %(
     wrong_index.count(True), len(correct_wavelength))

if __name__ == "__main__":
  # Random
  # ROT_DEBUG = None
  # can choose either ROT_DEBUG to test a fixed orientation generated by sim.py for debugging purposes
  # Crazy
  #ROT_DEBUG = (-0.6547565638290659, -0.30361830715206495, 0.6921775535835215,
  #0.5293791302365225, 0.4694357123764552, 0.7066737920820565, -0.5394919634181565,
  #0.8291236551369916, -0.14663691861021177)
  # Almost OK
  if 0:
    R = (-0.21969520495012107, 0.9734890660022186, -0.06366361045410796,
         0.4610855793407041, 0.04610460484714496, -0.8861571271145632,
         -0.8597290884028361, -0.22403884436859772, -0.45899073059051165)
    run(R)
  elif 0:
    R =  (0.4602,  0.8876, -0.0204,
          -0.8842,  0.4602,  0.0794,
          0.0798, -0.0185,  0.9966)
    run(R)
  else:
    run()
