
from cxid9114 import utils
from cxid9114.sim import sim_utils
from scipy.spatial import cKDTree
from scipy import ndimage
from cxid9114.spots import spot_utils
import numpy as np
import pylab as plt

import sys

"""
This script is used to analyze the isr.py data files..
"""

data_file = sys.argv[1]

data = utils.open_flex(data_file)
C0 = data[sys.argv[2]]

Patts = sim_utils.PatternFactory(Ncells_abc=(5,5,5), Gauss=0, oversample=2)
en, fcalc = sim_utils.load_fcalc_file("../sim/fcalc_slim.pkl")
flux= [ data['fracA']*1e14, data['fracB']*1e14]
sim_patt = Patts.make_pattern2(C0, flux, en, fcalc, 20, 0.15, False)
imgA,imgB = sim_patt
threshA = threshB = 0

labimgA, nlabA = ndimage.label(imgA > threshA)
out = ndimage.center_of_mass(imgA, labimgA, range(1, nlabA))
yA, xA = map(np.array, zip(*out))

labimgB, nlabB = ndimage.label(imgB > threshB)
out = ndimage.center_of_mass(imgB, labimgB, range(1, nlabB))
yB, xB = map(np.array, zip(*out))

xAB, yAB = np.hstack((xA, xB)), np.hstack((yA, yB))

xdata, ydata, _ = map(np.array, spot_utils.xyz_from_refl(data['refl']))

tree = cKDTree( zip( xAB, yAB) )
dist, pos = tree.query(zip(xdata, ydata))
missedx, missedy = xdata[ dist >= 4], ydata[dist>=4]
# ====
# PLOT
# ====

s = 4
r = 5
plt.figure()
ax = plt.gca()
Square_styleA = {"ec": "C1", "fc": "C1", "lw": "1", "width": s, "height": s}
Square_styleB = {"ec": "C3", "fc": "C3", "lw": "1", "width": s, "height": s}
Circle_style = {"ec": "C0", "fc": "C0", "lw": "1", "radius": r}
Circle_styleMissed = {"ec": "Deeppink", "fc": "Deeppink", "lw": "1", "radius": r+4}
spot_utils.add_xy_to_ax_as_patches(zip(missedx, missedy), plt.Circle, Circle_styleMissed, ax)
spot_utils.add_xy_to_ax_as_patches(zip(xdata, ydata), plt.Circle, Circle_style, ax)
spot_utils.add_xy_to_ax_as_patches(zip(xA - s / 2., yA - s / 2.), plt.Rectangle, Square_styleA, ax)
spot_utils.add_xy_to_ax_as_patches(zip(xB - s / 2., yB - s / 2.), plt.Rectangle, Square_styleB, ax)
ax.set_ylim(0, 1800)
ax.set_xlim(0, 1800)
ax.set_aspect('equal')


n_idx = sum(dist < 4)
n_refl = len( dist)
title = " %d / %d ( %.2f %%) reflections were indexed within 4 pixels" % \
      (n_idx, n_refl, 100. * n_idx / n_refl )
ax.set_title(title)
plt.show()
