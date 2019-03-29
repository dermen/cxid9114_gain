# coding: utf-8
import dxtbx
from dxtbx.model.experiment_list import ExperimentListFactory
from cxid9114 import gain_utils
from cxid9114 import fit_utils
import sys
import numpy as np
from cxid9114 import utils

idx = int(sys.argv[1])
plot = int( sys.argv[2])
loader = dxtbx.load('image_files/_autogen_run62.loc')
data = [x.as_numpy_array() for x in loader.get_raw_data(idx)]
data32 = np.array([np.hstack([data[i*2], data[i*2+1]]) for i in range(32)])

# undo the gain correction
data32[loader.gain] /= loader.nominal_gain_val

out = gain_utils.get_gain_dists(data32, loader.gain, loader.cspad_mask, 
    plot=False, norm=True) #, bins_high=np.linspace(-20,100,600))
fit = fit_utils.fit_low_gain_dist(out[0], out[1], plot=plot)
fitH = fit_utils.fit_high_gain_dist(out[2], out[3], plot=plot)

bgg = fitH[2].params['wid0'].value / fit[2].params['wid0'].value
g = fitH[2].params['mu1'].value / fit[2].params['mu1'].value

ph = fitH[2].params['mu1'].value
ph2 = out[2][ utils.smooth(out[3], 11, 75)[200:350].argmax()+200]

print bgg, g, ph, ph2
data32[loader.gain] *= g



