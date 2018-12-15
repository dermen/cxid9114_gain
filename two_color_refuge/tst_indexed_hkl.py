from __future__ import division

import copy
from six.moves import range
from dials.array_family import flex
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
from cxi_xdr_xes.two_color import merge_close_spots
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.format.Format import Reader, Masker
from dxtbx.model.beam import BeamFactory as beam_factory
from dials.algorithms.indexing.indexer import master_phil_scope
# from libtbx.phil import command_line
from cctbx import crystal
from scitbx.matrix import col
from dxtbx.model import Crystal
from dxtbx.model.detector import DetectorFactory as detector_factory
from dials.algorithms.indexing.compare_orientation_matrices\
      import difference_rotation_matrix_axis_angle
from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope

master_phil_scope.adopt_scope(two_color_phil_scope)
master_phil_scope.extract()

'''This program is designed to test that the final indexed image hkl values
for experiment list are unigue. This is done by comparing the list of hkl values
for each associated experiment to the set of unique hkl values; the number of
elements in each list should be the same. This is done for 100 simulated diffraction
images of thermolysin. '''

def index(reflections, detector, known_symmetry, beams):
  '''calls the two color indexer class after setting up the phil parameters
  and returns a class object'''

  # cmd_line = command_line.argument_interpreter(master_params=master_phil_scope)
  # working_phil = cmd_line.process_and_fetch(args=[])
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
  params.indexing.stills.rmsd_min_px = 3.5
  filenames=[""]
  reader = Reader(filenames)
  masker = Masker(filenames)
  imgsetdata = ImageSetData( reader, masker)
  imgset = ImageSet(imgsetdata)
  imgset.set_beam(beams[0])
  imgset.set_detector(detector)
  imagesets = [imgset]

  orient = indexer_two_color(reflections, imagesets, params)
  orient.index()
  return orient

def test_indexed_hkl():
  '''tests the uniqueness of hkl values associated with each experiment for
  100 simulated randomly oriented thermolysin diffraction images indexed using
  two color indexer'''
  flex.set_random_seed(42)
  known_symmetry = crystal.symmetry("78,78,37,90,90,90","P43212")

  detector = detector_factory.simple('SENSOR_UNKNOWN',125,(97.075,97.075),
                                     '+x','-y',(0.11,0.11),(1765,1765))
  wavelength1=12398/7400 #wavelength for 2 color experiment in Angstroms
  wavelength2=12398/7500 #wavelength for 2 color experiment in Angstroms

  beam1 = beam_factory.simple_directional((0,0,1), wavelength1)
  beam2 = beam_factory.simple_directional((0,0,1), wavelength2)

  a_basis = []
  b_basis = []
  c_basis = []

  # refiner resets random number seed so in order to get the same 100 images
  #generated each time the random seed is set
  # the implementation is as follows
  # gets simulated images
  sims = [merge_close_spots.merge_close_spots() for i in range(2)]

  for data in sims:
    A = data.A
    A_inv = A.inverse()

    a = col(A_inv[:3])
    b = col(A_inv[3:6])
    c = col(A_inv[6:])
    crystinp = Crystal(a, b, c, space_group=known_symmetry.space_group())
    a_basis.append(a)
    b_basis.append(b)
    c_basis.append(c)

    res = data.two_color_sim
    info = data.spot_proximity(res)

    refl = info[0]
    result = index(refl, detector, known_symmetry, [beam1, beam2])
    cm = result.refined_experiments.crystals()[0]
    R, best_axis, best_angle, change_of_basis = difference_rotation_matrix_axis_angle(
        crystal_a=cm, crystal_b=crystinp)

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
    params.indexing.stills.rmsd_min_px = 3.5

    expts = copy.deepcopy(result.refined_experiments)
    expts.crystals()[0].change_basis(change_of_basis)

    reflections_exp0 = result.refined_reflections.select(result.refined_reflections['id']==0)
    reflections_exp1 = result.refined_reflections.select(result.refined_reflections['id']==1)

    assert len(reflections_exp0['miller_index'])==len(set(reflections_exp0['miller_index']))
    assert len(reflections_exp1['miller_index'])==len(set(reflections_exp1['miller_index']))
  print "OK"

if __name__=='__main__':
  test_indexed_hkl()
