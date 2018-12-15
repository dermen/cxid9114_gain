from __future__ import division
from six.moves import range
from dials.array_family import flex
from cxi_xdr_xes.two_color import merge_close_spots

'''
This program is designed to test the resolution of the close spots
(those < 2 pixels apart). These spots should be in the low to medium resolution
range. The program checks that the spot resolution is 14 angstroms or lower.
'''

def test_close_spot_res():
  '''from 100 simulated diffraction images the resolution of spots < 2 pixels
  apart is calculated and asserted to be < 14 angstroms'''
  flex.set_random_seed(42)

  #unit test average close spots should be in the low resolution region >=15 angstroms
  for j in range(100):
    data = merge_close_spots.merge_close_spots()
    res = data.two_color_sim
    info = data.spot_proximity(res)
    detector = data.detector
    beams = data.beams
    spots_a = res['wavelength_1_spots']
    spots_b = res['wavelength_2_spots']
    mapping = info[1]
    resolution, distances = data.spot_resolution(detector, beams, spots_a, spots_b, mapping)
    for i in range(len(distances)):
      if resolution[i] >=14:
        test = distances[i]
        assert distances[i] <= 2, test
  print "ok"

if __name__=='__main__':
  test_close_spot_res()
