from __future__ import division
from six.moves import range

from dials.array_family import flex
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
from cxi_xdr_xes.two_color import merge_close_spots
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.format.Format import Reader, Masker

from dxtbx.model.beam import BeamFactory as beam_factory
from dials.algorithms.indexing.indexer import master_phil_scope
from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope
# from libtbx.phil import command_line
from cctbx import crystal
from scitbx.matrix import col

from dxtbx.model.detector import DetectorFactory as detector_factory
import math

'''
This program is designed to examine the candidate basis vectors calculated from
the two color 2-D grid search. The angles between the simulated basis vectors
and those found by the grid search should be small (< 1 degree). This assertion
is implemented and tested.

'''

master_phil_scope.adopt_scope(two_color_phil_scope)
master_phil_scope.extract()

def index(reflections, detector, known_symmetry, beams):
  '''sets up two color indexer parameters and calls the indexer; returns the
  candidate basis vectors from the 2-D grid search in two color indexer'''

  # cmd_line = command_line.argument_interpreter(master_params=master_phil_scope)
  # working_phil = cmd_line.process_and_fetch(args=[])
  params = master_phil_scope.extract()
  params.refinement.parameterisation.beam.fix="all"
  params.refinement.parameterisation.detector.fix="all"
  params.indexing.known_symmetry.space_group=known_symmetry.space_group_info()
  params.refinement.verbosity=3
  params.indexing.refinement_protocol.d_min_start
  params.indexing.refinement_protocol.n_macro_cycles = 1 # 5
  params.indexing.known_symmetry.unit_cell=known_symmetry.unit_cell()
  params.indexing.debug=True
  params.indexing.known_symmetry.absolute_angle_tolerance=5.0
  params.indexing.known_symmetry.relative_length_tolerance=0.3
  params.indexing.stills.rmsd_min_px = 3.5
  filenames=[""]
  reader = Reader(filenames)
  masker = Masker(filenames)
  iset_data = ImageSetData( reader, masker)
  imgset = ImageSet( iset_data)
  imgset.set_beam(beams[0])
  imgset.set_detector(detector)
  imagesets = [imgset]

  orient = indexer_two_color(reflections, imagesets, params) #, beams=beams)
  orient.index()
  cvecs = orient.candidate_basis_vectors
  return cvecs

def get_basis():
  '''gets the input basis vectors of 100 simulated diffraction images'''

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

  unique_vectors = []

  # refiner resets random number seed so in order to get the same 100 images
  #generated each time the random seed is set
  # the implementation is as follows
  sims = [merge_close_spots.merge_close_spots() for i in range(2)]

  for data in sims:
    A = data.A
    A_inv = A.inverse()
    a = col(A_inv[:3])
    b = col(A_inv[3:6])
    c = col(A_inv[6:])
    a_basis.append(a)
    b_basis.append(b)
    c_basis.append(c)

    res = data.two_color_sim
    info = data.spot_proximity(res)
    refl = info[0]
    candidate_basis_vectors=index(refl, detector, known_symmetry, [beam1, beam2])
    unique_vectors.append(candidate_basis_vectors)
  return a_basis, b_basis, c_basis, unique_vectors

def calc_angles():
  '''angles between input basis and candidate basis are calculated'''
  inp_a, inp_b, inp_c, cand = get_basis()
  res_a = flex.double()
  res_b = flex.double()
  res_c = flex.double()

  for i in range(len(cand)):
    for k in range(len(cand[i])):
      res_a.append(inp_a[i].dot(cand[i][k])/(inp_a[i].length()*cand[i][k].length()))
      res_b.append(inp_b[i].dot(cand[i][k])/(inp_b[i].length()*cand[i][k].length()))
      res_c.append(inp_c[i].dot(cand[i][k])/(inp_c[i].length()*cand[i][k].length()))

  #result is in radians convert to degrees
  a_angles = flex.acos(res_a)*180/math.pi
  b_angles = flex.acos(res_b)*180/math.pi
  c_angles = flex.acos(res_c)*180/math.pi

  #account for parallel or antiparallel candidate basis vectors
  for j in range(len(a_angles)):
    if a_angles[j]>=90:
      a_angles[j]=180-a_angles[j]

  for j in range(len(b_angles)):
    if b_angles[j]>=90:
      b_angles[j]=180-b_angles[j]

  for j in range(len(c_angles)):
    if c_angles[j]>=90:
      c_angles[j]=180-c_angles[j]

  return a_angles, b_angles, c_angles

def test_candidate_basis_vectors(verbose=True):
  '''angles between input basis and candidate basis are compared and asserted
  to be less than 1 degree'''
  angles = calc_angles()
  a_angles = angles[0]
  b_angles = angles[1]
  c_angles = angles[2]
  n_iter = int( len( angles) / 30.) # 30 is the magic number from the paper, they try 30 directions
  for i in range( n_iter):
    #if you want these values you can uncomment the print statements for debugging
    #print list(a_angles[i*30:30*(i+1)])
    #print list(b_angles[i*30:30*(i+1)])
    #print list(c_angles[i*30:30*(i+1)])

    #adds a pause after each 3 basis vector comparision to candidates if uncommented
    #raw_input()

    #this gives the angle between the input basis (a, b and c) with the
    #best candidate basis vector
    #print min(min(angle[i*30:30*(i+1)]) for angle in angles[0:3])
    test = min(
      min(list(a_angles[i*30:30*(i+1)])),
      min(list(b_angles[i*30:30*(i+1)])),
      min(list(c_angles[i*30:30*(i+1)]))
      )
    # assertion that the angle between the input basis vectors and best
    #candidate basis vector is less than 1.0 degree
    assert test < 1.0, test
  print "OK"

if __name__=='__main__':
  test_candidate_basis_vectors()
