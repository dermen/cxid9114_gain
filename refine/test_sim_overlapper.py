
from cxid9114 import utils
from cxid9114.sim import sim_utils
from cxid9114.spots import spot_utils

import sys

"""
This script outlines how to compute overlap between strong spots (refls)
and simulated two color images..  
"""

# this is a pickle file with at least one crystal and a reflection table in it
data_file = sys.argv[1]

crystal_key = sys.argv[2]  # name of crystal in dict
refl_key =sys.argv[3]  # name of relfection table in dict

data = utils.open_flex(data_file)
crystal = data[crystal_key]
refls = data[refl_key]

dump = sim_utils.sim_twocolors(crystal, Ncells_abc=(5,5,5), Gauss=False, oversample=2)
imgA, imgB = dump['imgA'], dump['imgB']

spotsA = spot_utils.get_spot_data(imgA, thresh=1e-6)
spotsB = spot_utils.get_spot_data(imgB, thresh=1e-6)

spot_utils.plot_overlap( spotsA, spotsB, refls)

