from __future__ import absolute_import, division

import h5py
import numpy as np
try:
    import pylab as plt
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False
from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill
from scitbx import matrix
from dials.array_family import flex
from dxtbx.model.detector import DetectorFactory
from dxtbx.model.beam import BeamFactory
try:
    import cxid9114
    HAS_D91 = True
except ImportError:
    HAS_D91 = False
    
# required HDF5 keys
REQUIRED_KEYS = ['bigsim_d9114']

detDict={'hierarchy': {'children': [{'panel': 0}],
  'fast_axis': (1.0, 0.0, 0.0),
  'gain': 1.0,
  'identifier': '',
  'image_size': (0, 0),
  'mask': [],
  'material': '',
  'mu': 0.0,
  'name': '',
  'origin': (0.0, 0.0, 0.0),
  'pedestal': 0.0,
  'pixel_size': (0.0, 0.0),
  'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
  'raw_image_offset': (0, 0),
  'slow_axis': (0.0, 1.0, 0.0),
  'thickness': 0.0,
  'trusted_range': (0.0, 0.0),
  'type': ''},
 'panels': [{'fast_axis': (1.0, 0.0, 0.0),
   'gain': 1.0,
   'identifier': '',
   'image_size': (1800, 1800),
   'mask': [],
   'material': '',
   'mu': 0.0,
   'name': 'Panel',
   'origin': (-99.165, 99.05499999999999, -125.0),
   'pedestal': 0.0,
   'pixel_size': (0.11, 0.11),
   'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
   'raw_image_offset': (0, 0),
   'slow_axis': (0.0, -1.0, 0.0),
   'thickness': 0.0,
   'trusted_range': (-29.0, 65505.0),
   'type': 'SENSOR_CCD'}]}

beamDict ={'direction': (0.0, 0.0, 1.0),
 'divergence': 0.0,
 'flux': 0.0,
 'polarization_fraction': 0.999,
 'polarization_normal': (0.0, 1.0, 0.0),
 'sigma_divergence': 0.0,
 'transmission': 1.0,
 'wavelength': 1.37095}

BEAM = BeamFactory.from_dict(beamDict)
DET = DetectorFactory.from_dict(detDict)
IMG_SIZE = (1800, 1800)

class FormatBigSimD9114(FormatHDF5, FormatStill):
    """
    Class for reading D9114 simulated monolithic cspad data
    """
    @staticmethod
    def understand(image_file):
        if not HAS_D91:
            return False
        h5_handle = h5py.File(image_file, 'r')
        h5_keys = h5_handle.keys()
        understood = all([k in h5_keys for k in REQUIRED_KEYS])
        return understood

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatHDF5.__init__(self, image_file, **kwargs)

        self._h5_handle = h5py.File(self.get_image_file(), 'r')
        self._geometry_define()

    def _geometry_define(self):
        self._cctbx_detector = DET
        self._cctbx_beam = BEAM

    def get_num_images(self):
        return 1

    def show_image(self, index, **kwargs):
        self.load_panel_img(index)
        if CAN_PLOT:
            plt.figure()
            plt.imshow(self.panel_img, **kwargs)
            plt.show()
        else:
            print("Cannot plot")

    def load_panel_img(self):
        self.panel_img = self._h5_handle["bigsim_d9114"][()]
        if not self.panel_img.dtype == np.float64:
            self.panel_img = self.panel_img.astype(np.float64)

    def get_raw_data(self, index=0):
        self.load_panel_img()
        return flex.double(self.panel_img)

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_detector(self, index=None):
        return self._cctbx_detector

    def get_beam(self, index=None):
        return self._cctbx_beam

if __name__ == '__main__':
    import sys

    for arg in sys.argv[1:]:
        print(FormatBigSimD9114.understand(arg))
