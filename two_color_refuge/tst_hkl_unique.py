from __future__ import division
from six.moves import range
from dials.array_family import flex
from cxi_xdr_xes.two_color import two_color_still_sim
from dxtbx.model.beam import BeamFactory as beam_factory
from cctbx import crystal
from dxtbx.model.detector import DetectorFactory as detector_factory

'''This program provides a sanity check for the simulated diffraction data and
checks that each associate experiment has a set of unique hkls.'''

def test_unique_hkl():
  '''tests the uniqueness of hkl values associated with each experiment for
  100 simulated randomly oriented thermolysin diffraction images prior to indexing.'''
  flex.set_random_seed(42) # the meaning of life
  known_symmetry = crystal.symmetry("78,78,37,90,90,90","P43212")

  detector = detector_factory.simple('SENSOR_UNKNOWN',125,(97.075,97.075),
                                     '+x','-y',(0.11,0.11),(1765,1765))
  wavelength1=12398/7400 #wavelength for 2 color experiment in Angstroms
  wavelength2=12398/7500 #wavelength for 2 color experiment in Angstroms

  beam1 = beam_factory.simple_directional((0,0,1), wavelength1)
  beam2 = beam_factory.simple_directional((0,0,1), wavelength2)
  beams = [beam1, beam2]

  for i in range(100):
    data = two_color_still_sim.two_color_still_sim(known_symmetry,rot=None)
    hkl_list = data.hkl_list
    A = data.A

    sim_data = data.ewald_proximity_test(beams, hkl_list, A, detector)
    refl = sim_data.get('reflection_table')
    refl1 = refl.select(refl['set_id']==0)
    refl2 = refl.select(refl['set_id']==1)

    # unit test to make sure all miller indices are unique for each experiment id
    assert len(refl1['miller_index'])==len(set(refl1['miller_index']))
    assert len(refl2['miller_index'])==len(set(refl2['miller_index']))
  print "ok"

if __name__=='__main__':
  test_unique_hkl()
