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

try:
    from cxid9114.geom import geom_utils
    from cxid9114.parameters import WAVELEN_LOW
    from cxid9114.common_mode.pppg import pppg
    HAS_D91 = True

except ImportError:
    HAS_D91 = False

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
CAMERA_LENGTH = 124.5  # CSPAD sample-to-detector distance
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
    """
    @staticmethod
    def understand(image_file):
        if not HAS_D91:
            print("FAILED D91")
            return False
        h5_handle = h5py.File(image_file, 'r')
        h5_keys = h5_handle.keys()
        understood = all([k in h5_keys for k in REQUIRED_KEYS])
        if not understood:
            print("FAIED D91")
        return understood

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatHDF5.__init__(self, image_file, **kwargs)

        #def _start(self):
        self._h5_handle = h5py.File(self.get_image_file(), 'r')
        self._decide_multi_panel()
        self.load_dark()
        self.load_gain()
        self.load_mask()
        self.load_xyz()
        self._geometry_define()
        self._assembler_define()

    def _decide_multi_panel(self):
        """
        decides whether to use single or multi-panel cspad modes...
        """
        if "multi_panel" in self._h5_handle.keys():
            self.as_multi_panel = self._h5_handle["multi_panel"][()]
            if self.as_multi_panel:
                if "psf" not in self._h5_handle.keys():
                    raise ValueError("Need the TJ Lane PSF vectors! psana.Detector(cspad).geometry().get_psf()")
        else:
            self.as_multi_panel = False


    def load_dark(self):
        self.dark = self._h5_handle["pedestal"][()]
        assert (self.dark.dtype == np.float64)

    def load_gain(self):
        self.gain = self._h5_handle["panel_gainmasks"][()]
        assert (self.gain.dtype == np.bool)
        self.gain_val = self._h5_handle["gain_val"][()]

    def load_mask(self):
        self.mask = self._h5_handle["panel_masks"][()]
        assert (self.mask.dtype == np.bool)

    def load_xyz(self):
        self.panel_X = self._h5_handle["panel_x"][()]
        self.panel_Y = self._h5_handle["panel_y"][()]
        self.panel_Z = self._h5_handle["panel_z"][()]

    def _assembler_define(self):
        if not self.as_multi_panel:
            bins0 = np.arange(-IMG_SIZE[0]/2, IMG_SIZE[0]/2+1)
            bins1 = np.arange(-IMG_SIZE[1]/2, IMG_SIZE[1]/2+1)
            assert(len(bins0) == IMG_SIZE[0]+1)
            assert(len(bins1) == IMG_SIZE[1]+1)
            self.hist_args = {"x": self.panel_X.ravel(),
                              "y": self.panel_Y.ravel(),
                              "bins": [bins0, bins1]}

    def _geometry_define(self):
        if not self.as_multi_panel:
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

        else:
            psf = self._h5_handle["psf"][()]
            psf[0, :, 2] = -1*CAMERA_LENGTH*1000
            self._cctbx_detector = geom_utils.make_dials_cspad(psf)

        self._cctbx_beam = self._beam_factory.simple(WAVELEN_LOW)

    def get_num_images(self):
        return self._h5_handle["panels"].shape[0]

    def assemble(self, panels):
        if not self.as_multi_panel:
            return np.histogram2d(
                x=self.hist_args["x"],
                y=self.hist_args["y"],
                bins=self.hist_args["bins"],
                weights=panels.ravel())[0]

    def show_image(self, index, **kwargs):
        if self.as_multi_panel:
            self._correct_raw_data(index)
            img2d = self.assemble(self.panels)
            if CAN_PLOT:
                plt.figure()
                plt.imshow(img2d, **kwargs)
                plt.show()
            else:
                print("Cannot plot")
        else:
            print ("not implemented for multi panel")

    def _assemble_panels(self):
        if not self.as_multi_panel:
            self.panel_img = np.histogram2d(
                x=self.hist_args["x"],
                y=self.hist_args["y"],
                bins=self.hist_args["bins"],
                weights=self.panels.ravel())[0]
            self.panel_img = np.ascontiguousarray(self.panel_img)

    def _correct_raw_data(self, index):
        self.panels = self._h5_handle['panels'][index].astype(np.float64)  # 32x185x388 psana-style cspad array
        self._correct_panels()  # applies dark cal, common mode, and gain, in that order..

    def get_raw_data(self, index=0):
        self._correct_raw_data(index)
        if not self.as_multi_panel:  # is single slab detector
            self._assemble_panels()
            return flex.double(self.panel_img)
        else:  # if multi-panel detector
            return geom_utils.psana_data_to_aaron64_data(self.panels, as_flex=True)

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
