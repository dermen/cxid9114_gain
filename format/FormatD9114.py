from __future__ import absolute_import, division, print_function

import numpy as np
import psana

from xfel.cxi.cspad_ana.cspad_tbx import env_distance
from cxid9114.common_mode.pppg import pppg
from cxid9114.mask import mask_utils
from cxid9114 import assemble_cspad
from scitbx import matrix
from dxtbx.model.detector import DetectorFactory
from dxtbx.format.FormatXTC import locator_str
from dxtbx.format.FormatXTCCspad import FormatXTCCspad, cspad_locator_str
from scitbx.array_family import flex
from libtbx import phil

d9114_locator_str = """
  d9114 {
    common_mode_algo = something # put this here to break the understand so D9114 doesnt override XTCCspad by default
      .type = str
      .help = Common mode correction ppg default or unbonded
    low_gain_zero_peak = (-5,5,100)
        .type = floats(size=3)
        .help = a numpy linspace specifying the ADU extent of the low-gain 0-photon peak 
    high_gain_zero_peak = (-10,10,200)
        .type = floats(size=3)
        .help = a numpy linspace specifying the ADU extent of the high-gain 0-photon peak 
    savgol_polyorder = 3
        .type = int
        .help = degree of polynomial used to smooth zero-photon peak
    savgol_windowlength = 51
        .type = int
        .help = window size of for the savgol smoothing, should be odd \
                (in relation to the low/high gain zero peak region)
    }
"""

d9114_locator_scope = phil.parse(d9114_locator_str + locator_str + cspad_locator_str,
                                 process_includes=True)

# load some masks
MASK1 = mask_utils.load_mask("detail_mask")
MASK2 = mask_utils.load_mask("corners_mask")
MASK3 = mask_utils.load_mask("small_regions_mask")
CSPAD_MASK = MASK1*MASK2*MASK3


class FormatD9114(FormatXTCCspad):
    run_number = None  # type: int

    def __init__(self, image_file, **kwargs):
        assert (self.understand(image_file))
        FormatXTCCspad.__init__(self, image_file, locator_scope=d9114_locator_scope, **kwargs)

        self._ds = FormatXTCCspad._get_datasource(image_file, self.params)
        self.run_number = self.params.run[0]
        self.cspad = psana.Detector(self.params.detector_address[0])
        self.dark = self.cspad.pedestals(self.run_number).astype(np.float64)
        self.gain = self.cspad.gain_mask(self.run_number) == 1.
        if CSPAD_MASK is not None:
            self.cspad_mask = CSPAD_MASK
        else:
            self.cspad_mask = np.ones_like( self.gain)
        self.nominal_gain_val = self.cspad._gain_mask_factor
        self.populate_events()
        self.n_images = len(self.times)
        self.params = FormatD9114.get_params(image_file)
        self._set_pppg_args()
        self._set_psf()
        self._set_2d_img_info()
        self.detector_distance = env_distance(self.params.detector_address[0],
                                              self._ds.env(), self.params.cspad.detz_offset)
        # self.feespec = psana.Detector("FeeSpec-bin")

    def _set_2d_img_info(self):
        dummie_event = self._get_event(0)
        self.img2d_mask = self.cspad.image(dummie_event, self.cspad_mask )\
            .astype(int).astype(bool)
        self.img_sh = self.img2d_mask.shape

    def show_data(self, index):
        data = self.get_psana_data(index)* self.cspad_mask
        assemble_cspad.assemble_cspad(data, self.psf)

    def _set_pppg_args(self):
        """
        sets the parameters for common mode correction pppg
        The boolean parameters should remain as-is
        """
        l1, l2, Nl = self.params.d9114.low_gain_zero_peak
        h1, h2, Nh = self.params.d9114.low_gain_zero_peak
        self.pppg_args = {"low_x1": l1, "low_x2": l2, "Nlow": Nl,
                            "high_x1": h1, "high_x2": l2, "Nhigh": Nh,
                            "polyorder": self.params.d9114.savgol_polyorder,
                            "window_length": self.params.d9114.savgol_windowlength,
                            "inplace": True, "plot_details": False, "verbose": False,
                            "plot_metric": False}

    @staticmethod
    def get_params(image_file):
        user_scope = phil.parse(file_name=image_file, process_includes=True)
        params = d9114_locator_scope.fetch(user_scope).extract()
        return params

    @staticmethod
    def understand(image_file):
        params = FormatD9114.get_params(image_file)
        return params.experiment == "cxid9114" and \
               params.d9114.common_mode_algo in ['default', 'pppg', 'unbonded']

    def get_detector(self, index=None):
        return self._detector(self, index)

    def get_psana_data( self, index):
        self.event = self._get_event(index)
        raw = self.cspad.raw(self.event).astype(np.float32)
        data = raw.astype(np.float64) - self.dark
        if self.params.d9114.common_mode_algo == 'default':
            self.cspad.common_mode_apply(self.run_number, data, (1, 25, 25, 100, 1))  # default for cspad
        elif self.params.d9114.common_mode_algo == 'unbonded':
            self.cspad.common_mode_apply(self.run_number, data, (
                5, 0, 0, 0, 0))  # default for non-bonded pixels, but these are not in cxid9114 i believe..
        elif self.params.d9114.common_mode_algo == "pppg":
            pppg(data, self.gain, self.cspad_mask, **self.pppg_args)

        data[self.gain] = data[self.gain] * self.nominal_gain_val

        return data

    def _set_psf(self):
        geom = self.cspad.geometry(self.run_number)
        self.psf = map(np.array, zip(*geom.get_psf()))

    def get_raw_data(self, index):
        """this is really corrected data..."""
        data = self.get_psana_data(index)
        data2d = self.cspad.image(self.event, data)
        assert( data.dtype==np.float64)
        #cctbx_det = self.get_detector(index)
        #self._raw_data = []
        #for quad_count, quad in enumerate(cctbx_det.hierarchy()):
        #    for sensor_count, sensor in enumerate(quad):
        #        for asic_count, asic in enumerate(sensor):
        #            fdim, sdim = asic.get_image_size()
        #            asic_data = data[sensor_count + quad_count * 8, :,
        #                        asic_count * fdim:(asic_count + 1) * fdim]  # 8 sensors per quad
        #            self._raw_data.append(flex.double(np.array(asic_data)))
        #return tuple(self._raw_data)
        return flex.double( data2d*self.img2d_mask)

    def get_detector(self, index=None):
        return self._get_detector()

    def _get_detector(self):
        pixel_size=0.10992
        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, -1.0, 0.0))
        image_size = (self.img_sh[1], self.img_sh[0])
        orig = matrix.col((-image_size[0]*pixel_size/2.,
                           image_size[1]*pixel_size/2., self.detector_distance))
        trusted_range = (-100, 2**16)
        return DetectorFactory.make_detector("", fast, slow, orig,
            (pixel_size, pixel_size), image_size, trusted_range)

if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatD9114.understand(arg))
