from __future__ import absolute_import, division, print_function

from dxtbx.format.FormatXTC import FormatXTC, locator_str
from dxtbx.format.FormatXTCCspad import FormatXTCCspad, cspad_locator_str
from scitbx.array_family import flex

import numpy as np

from libtbx import phil

try:
    from xfel.cxi.cspad_ana import cspad_tbx
    from xfel.cftbx.detector import cspad_cbf_tbx
except ImportError:
    # xfel not configured
    pass

import psana

d9114_locator_str = """
  d9114 {
    common_mode = default
      .type = str
      .help = Common mode correction ppg default or unbonded
    }
"""

d9114_locator_scope = phil.parse(d9114_locator_str + locator_str + cspad_locator_str,
                                 process_includes=True)

class FormatD9114(FormatXTCCspad):

    def __init__(self, image_file, **kwargs):
        assert (self.understand(image_file))
        FormatXTCCspad.__init__(self, image_file, **kwargs)

        self._ds = FormatXTCCspad._get_datasource(image_file, self.params)
        self.run_number = self.params.run[0]
        self.cspad = psana.Detector(self.params.detector_address[0])
        self.dark = self.cspad.pedestal(self.run_number).astype(np.float64)
        self.gain = self.cspad.gain_mask(self.run_number) == 1.
        self.nominal_gain_val = self.cspad._gain_mask_factor
        self.populate_events()
        self.n_images = len(self.times)

        # self.feespec = psana.Detector("FeeSpec-bin")

    @staticmethod
    def understand(image_file):
        try:
            params = FormatXTC.params_from_phil(d9114_locator_scope, image_file)
        except Exception:
            return False
        return params.experiment == "cxid9114" and \
            params.d9114.common_mode in ['default', 'pppg', 'unbonded']

    def get_raw_data(self, index):
        assert len(self.params.detector_address) == 1
        d = self.get_detector(self, index)

        event = self._get_event(index)
        raw = self.cspad.raw(event).astype(np.float32)
        data = raw.astype(np.float64) - self.dark
        if self.params.d9114.common_mode=='default':
            self.cspad.common_mode_apply(self.run_number, data, (1,25,25,100,1))
        elif self.params.d9114.common_mode=='unbonded':
            self.cspad.common_mode_apply( self.run_number, data, (5,0,0,0,0))

        data[self.gain] = data[self.gain]  * self.nominal_gain_val

        self._raw_data = []
        for quad_count, quad in enumerate(d.hierarchy()):
            for sensor_count, sensor in enumerate(quad):
                for asic_count, asic in enumerate(sensor):
                    fdim, sdim = asic.get_image_size()
                    asic_data = data[sensor_count + quad_count * 8, :,
                                asic_count * fdim:(asic_count + 1) * fdim]  # 8 sensors per quad
                    self._raw_data.append(flex.double(np.array(asic_data)))
        assert len(d) == len(self._raw_data)
        return tuple(self._raw_data)

if __name__ == '__main__':
    import sys

    for arg in sys.argv[1:]:
        print(FormatD9114.understand(arg))
