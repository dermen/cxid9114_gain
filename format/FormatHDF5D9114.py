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
        self._decide_multi_panel()
        self.load_dark()
        self.load_gain()
        self.load_mask()
        self.load_xyz()
        self._geometry_define()
        self._assembler_define()

    def _decide_multi_panel(self):
        if "multi_panel" in self._h5_handle.keys():
            self.as_multi_panel = self._h5_handle["multi_panel"][()]
        else:
            self.as_multi_panel = False

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

    def _assemble_panels(self):
        if not self.as_multi_panel:
            self.panel_img = np.histogram2d(
                x=self.hist_args["x"],
                y=self.hist_args["y"],
                bins=self.hist_args["bins"],
                weights=self.panels.ravel())[0]
            self.panel_img = np.ascontiguousarray(self.panel_img)

    def get_raw_data(self, index=0):
        self.panels = self._h5_handle['panels'][index].astype(np.float64)
        self._correct_panels()
        if not self.as_multi_panel:
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

    #def _detector(self, index=None):
    #    from xfel.cftbx.detector.cspad_cbf_tbx import read_slac_metrology
    #    from dxtbx.model import Detector
    #    from scitbx.matrix import col
    #    from dxtbx.model import ParallaxCorrectedPxMmStrategy
    #    from xfel.cxi.cspad_ana.cspad_tbx import env_distance
    #    if index is None: index = 0

    #    ev = self._get_event(index)
    #    run_number = ev.run()
    #    run = self._psana_runs[run_number]
    #    det = self._psana_det[ run_number]
    #    geom= det.pyda.geoaccess(run_number)
    #    cob = read_slac_metrology(geometry=geom, include_asic_offset=True)
    #    distance = env_distance(self.params.detector_address[0], run.env(), self.params.cspad.detz_offset)
    #    d = Detector()
    #    pg0 = d.hierarchy()
    #    # first deal with D0
    #    det_num = 0
    #    origin = col((cob[(0,)] * col((0,0,0,1)))[0:3])
    #    fast   = col((cob[(0,)] * col((1,0,0,1)))[0:3]) - origin
    #    slow   = col((cob[(0,)] * col((0,1,0,1)))[0:3]) - origin
    #    origin += col((0., 0., -distance))
    #    pg0.set_local_frame(fast.elems,slow.elems,origin.elems)
    #    pg0.set_name('D%d'%(det_num))
    #    for quad_num in xrange(4):
    #      # Now deal with Qx
    #      pg1 = pg0.add_group()
    #      origin = col((cob[(0,quad_num)] * col((0,0,0,1)))[0:3])
    #      fast   = col((cob[(0,quad_num)] * col((1,0,0,1)))[0:3]) - origin
    #      slow   = col((cob[(0,quad_num)] * col((0,1,0,1)))[0:3]) - origin
    #      pg1.set_local_frame(fast.elems,slow.elems,origin.elems)
    #      pg1.set_name('D%dQ%d'%(det_num, quad_num))
    #      for sensor_num in xrange(8):
    #      # Now deal with Sy
    #        pg2=pg1.add_group()
    #        origin = col((cob[(0,quad_num,sensor_num)] * col((0,0,0,1)))[0:3])
    #        fast   = col((cob[(0,quad_num,sensor_num)] * col((1,0,0,1)))[0:3]) - origin
    #        slow   = col((cob[(0,quad_num,sensor_num)] * col((0,1,0,1)))[0:3]) - origin
    #        pg2.set_local_frame(fast.elems,slow.elems,origin.elems)
    #        pg2.set_name('D%dQ%dS%d'%(det_num,quad_num,sensor_num))
    #        # Now deal with Az
    #        for asic_num in xrange(2):
    #          val = 'ARRAY_D0Q%dS%dA%d'%(quad_num,sensor_num,asic_num)
    #          p = pg2.add_panel()
    #          origin = col((cob[(0,quad_num,sensor_num, asic_num)] * col((0,0,0,1)))[0:3])
    #          fast   = col((cob[(0,quad_num,sensor_num, asic_num)] * col((1,0,0,1)))[0:3]) - origin
    #          slow   = col((cob[(0,quad_num,sensor_num, asic_num)] * col((0,1,0,1)))[0:3]) - origin
    #          p.set_local_frame(fast.elems,slow.elems,origin.elems)
    #          p.set_pixel_size((cspad_cbf_tbx.pixel_size, cspad_cbf_tbx.pixel_size))
    #          p.set_image_size(cspad_cbf_tbx.asic_dimension)
    #          p.set_trusted_range((cspad_tbx.cspad_min_trusted_value, cspad_tbx.cspad_saturated_value))
    #          p.set_name(val)

    #    try:
    #      beam = self._beam(index)
    #    except Exception:
    #      print('No beam object initialized. Returning CSPAD detector without parallax corrections')
    #      return d

    #    # take into consideration here the thickness of the sensor also the
    #    # wavelength of the radiation (which we have in the same file...)
    #    wavelength = beam.get_wavelength()
    #    thickness = 0.5 # mm, see Hart et al. 2012
    #    from cctbx.eltbx import attenuation_coefficient
    #    table = attenuation_coefficient.get_table("Si")
    #    # mu_at_angstrom returns cm^-1
    #    mu = table.mu_at_angstrom(wavelength) / 10.0 # mu: mm^-1
    #    t0 = thickness
    #    for panel in d:
    #      panel.set_px_mm_strategy(ParallaxCorrectedPxMmStrategy(mu, t0))
    #    return d

if __name__ == '__main__':
    import sys

    for arg in sys.argv[1:]:
        print(FormatHDF5D9114.understand(arg))
