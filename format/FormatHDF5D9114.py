from __future__ import absolute_import, division

import h5py
import numpy as np

from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill
from scitbx import matrix
from dials.array_family import flex

from cxid9114.parameters import WAVELEN_LOW
from cxid9114.common_mode.pppg import pppg

# required HDF5 keys
REQUIRED_KEYS = ['gain_val',
                 'panel_gainmasks',
                 'panel_masks',
                 'panel_x',
                 'panel_y',
                 'panel_z',
                 'panels',
                 'pedestal']
PIXEL_SIZE = 0.10992  # CSPAD pixel size in mm
CAMERA_LENGTH = 125  # CSPAD sample-to-detector distance
IMG_SIZE = (1800, 1800)
X_OFFSET = 0
Y_OFFSET = 0
BITMAX = 2 ** 14
PPPG_ARGS = {'Nhigh': 100.0,
             'Nlow': 100.0,
             'high_x1': -5.0,
             'high_x2': 5.0,
             'inplace': True,
             'low_x1': -5.0,
             'low_x2': 5.0,
             'plot_details': False,
             'plot_metric': False,
             'polyorder': 3,
             'verbose': False,
             'window_length': 51}


class FormatHDF5D9114(FormatHDF5, FormatStill):
    """
    Class for reading D9114 HDF5 hit files
    script (this script lives on the SACLA hpc).
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

        #def _start(self):
        self._h5_handle = h5py.File(self.get_image_file(), 'r')
        self.load_dark()
        self.load_gain()
        self.load_mask()
        self.load_xyz()
        self._geometry_define()
        self._assembler_define()

    def load_dark(self):
        self.dark = self._h5_handle["pedestal"].value
        assert (self.dark.dtype == np.float64)

    def load_gain(self):
        self.gain = self._h5_handle["panel_gainmasks"].value
        assert (self.gain.dtype == np.bool)
        self.gain_val = self._h5_handle["gain_val"].value

    def load_mask(self):
        self.mask = self._h5_handle["panel_masks"].value
        assert (self.mask.dtype == np.bool)

    def load_xyz(self):
        self.panel_X = self._h5_handle["panel_x"].value
        self.panel_Y = self._h5_handle["panel_y"].value
        self.panel_Z = self._h5_handle["panel_z"].value

    def _assembler_define(self):
        bins0 = np.arange(-IMG_SIZE[0]/2, IMG_SIZE[0]/2+1)
        bins1 = np.arange(-IMG_SIZE[1]/2, IMG_SIZE[1]/2+1)
        assert(len(bins0) == IMG_SIZE[0]+1)
        assert(len(bins1) == IMG_SIZE[1]+1)
        self.hist_args = {"x": self.panel_X.ravel(),
                          "y": self.panel_Y.ravel(),
                          "bins": [bins0, bins1]}

    def _geometry_define(self):
        orig_x = -IMG_SIZE[0]*PIXEL_SIZE*.5 + X_OFFSET
        orig_y = IMG_SIZE[1]*PIXEL_SIZE*.5 + X_OFFSET
        orig_z = -CAMERA_LENGTH
        orig = (orig_x, orig_y, orig_z)

        fast_scan = matrix.col((1.0, 0.0, 0.0))
        slow_scan = matrix.col((0.0, -1.0, 0.0))

        trusted_range = (-200, 2**16)
        name = ""
        self._cctbx_detector = \
            self._detector_factory.make_detector(
                name,
                fast_scan,
                slow_scan,
                orig,
                (PIXEL_SIZE, PIXEL_SIZE),
                IMG_SIZE,
                trusted_range)

        self._cctbx_beam = self._beam_factory.simple(WAVELEN_LOW)

    # def _detector(self, index=None):
    #    return self._cctbx_detector

    # def _beam(self, index=None):
    #    return self._cctbx_beam

    def get_num_images(self):
        return self._h5_handle["panels"].shape[0]

    def assemble(self, panels):
        return np.histogram2d(
            x=self.hist_args["x"],
            y=self.hist_args["y"],
            bins=self.hist_args["bins"],
            weights=panels.ravel())[0]

    def _assemble_panels(self):
        self.panel_img = np.histogram2d(
            x=self.hist_args["x"],
            y=self.hist_args["y"],
            bins=self.hist_args["bins"],
            weights=self.panels.ravel())[0]
        self.panel_img = np.ascontiguousarray(self.panel_img)

    def get_raw_data(self, index=0):
        self.panels = self._h5_handle['panels'][index].astype(np.float64)
        self._correct_panels()
        self._assemble_panels()
        return flex.double(self.panel_img)

    def _apply_mask(self):
        self.panels *= self.mask

    def _correct_panels(self):
        self.panels = self.panels.astype(np.float64)
        self.panels -= self.dark
        self._apply_mask()
        pppg(self.panels,
             self.gain,
             self.mask,
             **PPPG_ARGS)
        self.panels[self.gain] = self.panels[self.gain]*self.gain_val

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
        print(FormatHDF5D9114.understand(arg))
