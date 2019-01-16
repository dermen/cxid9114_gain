
from cxid9114 import utils
from cxid9114.sim import sim_utils
from scipy.spatial import cKDTree
from scipy import ndimage
from cxid9114.spots import spot_utils
import numpy as np
import pylab as plt
import dxtbx
import os
import sys
from scitbx.matrix import sqr

"""
1 color
This script is used to analyze the isr.py data files..
"""

data_file = sys.argv[1]
sim_new_cryst = int( sys.argv[2])
reset_unit_cell = int( sys.argv[3])

data = utils.open_flex(data_file)
C0 = data['crystal']
C1 = data['sim_indexed_crystals'][0]

if reset_unit_cell:
    print C1.get_unit_cell()
    #a,b,c,_,_,_ = C0.get_unit_cell().parameters()
    #B = sqr( (a,0,0,   0,b,0,  0,0,c)).inverse()
    A = sqr(C0.get_A())
    #B = sqr( C0.get_B())
    #C1.set_B(B)
    C1.set_A(A)
    print C1.get_unit_cell()

if sim_new_cryst:
    Patts = sim_utils.PatternFactory(crystal=C1)
    en, fcalc = sim_utils.load_fcalc_file("../sim/fcalc_slim.pkl")
    flux= [ data['fracA']*1e14, data['fracB']*1e14]
    imgA, imgB = Patts.make_pattern2(C1, flux, en, fcalc, 20,0.1, False)

else:
    orig_file = os.path.splitext( data_file)[0] + ".h5"
    imgA = dxtbx.load(orig_file).get_raw_data(0).as_numpy_array()

threshA = threshB = 0
labimgA, nlabA = ndimage.label(imgA > threshA)
out = ndimage.center_of_mass(imgA, labimgA, range(1, nlabA))
yA, xA = map(np.array, zip(*out))

xdata, ydata, _ = map(np.array, spot_utils.xyz_from_refl(data['sim_indexed_refls']))

tree = cKDTree( zip( xA, yA) )
dist, pos = tree.query(zip(xdata, ydata))
print C0.get_unit_cell().parameters()
print C1.get_unit_cell().parameters()

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
spot_utils.add_xy_to_ax_as_patches(zip(xdata, ydata), plt.Circle, Circle_style, ax)
spot_utils.add_xy_to_ax_as_patches(zip(xA - s / 2., yA - s / 2.), plt.Rectangle, Square_styleA, ax)
ax.set_ylim(0, 1800)
ax.set_xlim(0, 1800)
ax.set_aspect('equal')

n_idx = sum(dist < 4)
n_refl = len( dist)
title = " %d / %d ( %.2f %%) reflections were indexed within 4 pixels" % \
      (n_idx, n_refl, 100. * n_idx / n_refl )
ax.set_title(title)
plt.show()
