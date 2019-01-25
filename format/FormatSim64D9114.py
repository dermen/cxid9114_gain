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
from dials.array_family import flex
from cxid9114.geom import multi_det_dict
from cxid9114.parameters import WAVELEN_LOW
from dxtbx.model.detector import DetectorFactory
DET64 = DetectorFactory.from_dict(multi_det_dict.D)

# required HDF5 keys
REQUIRED_KEYS = ['sim64_d9114_images']

class FormatSim64D9114(FormatHDF5, FormatStill):
    """
    Class for reading D9114 simulated 64 panel files
    """
    @staticmethod
    def understand(image_file):
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
        self._cctbx_detector = DET64
        self._cctbx_beam = self._beam_factory.simple(WAVELEN_LOW)

    def get_num_images(self):
        return self._h5_handle["sim64_d9114_images"].shape[0]

    def load_panel_img(self, index):
        self.panel_img = self._h5_handle["sim64_d9114_images"][index]
        if not self.panel_img.dtype == np.float64:
            self.panel_img = self.panel_img.astype(np.float64)

    def get_raw_data(self, index=0):
        self.load_panel_img(index)
        return tuple( map( lambda x: flex.double(x), self.panel_img))

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
        print(FormatSim64D9114.understand(arg))
