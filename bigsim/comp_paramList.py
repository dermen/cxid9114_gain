#!/usr/bin/env libtbx.python

import glob
import os
from argparse import ArgumentParser

import numpy as np
import pylab as plt

import dxtbx
from dxtbx.model.crystal import CrystalFactory
from dials.algorithms.indexing.compare_orientation_matrices \
        import rotation_matrix_differences

from cxid9114 import utils

parser = ArgumentParser("Finds dirs and compares models")
parser.add_argument("-iglob", dest='iglob', help='glob strong for input directories', 
                required=True, type=str)
parser.add_argument("-b", dest='nbins', default=20, type=int, help="NUmber of hist bins")
parser.add_argument("-brange", nargs=2, dest='brange', type=float, default=[0,.1],
        help="Histogram min max passed as two consecutive args (min first)")
parser.add_argument("--no-absolute", dest='noabs', action='store_true', 
        help="Plot without taking the absolute value of deviation")
args = parser.parse_args()

cmd = 'find . -name "%s" -type d > _dirs_.txt' %args.iglob 
print(cmd)
os.system(cmd) 
dirs = np.loadtxt("_dirs_.txt", str)


inits = []
refs = []
for d in dirs:

    fnames = glob.glob(os.path.join(d, "*.pkl"))
    data = [utils.open_flex(f) for f in fnames]
    mxpos = np.argmax([d['F1'] for d in data])  # gets the F1 score (agreement between data and simulation)
    Cmax = data[mxpos]['crystal']
    data_orig = utils.open_flex(fnames[mxpos].split("_refine/")[0] + ".pkl")
    Cinit = data_orig["crystalAB"]
    img_f = data_orig["img_f"]

    loader = dxtbx.load(img_f)

    cryst_descr = {'__id__': 'crystal',
                  'real_space_a': loader._h5_handle["real_space_a"][()],
                  'real_space_b': loader._h5_handle["real_space_b"][()],
                  'real_space_c': loader._h5_handle["real_space_c"][()],
                  'space_group_hall_symbol': loader._h5_handle["space_group_hall_symbol"][()]}

    Csim = CrystalFactory.from_dict(cryst_descr)
    init_comp = rotation_matrix_differences((Csim, Cinit))
    ref_comp = rotation_matrix_differences((Csim, Cmax))
    init_rot = float(init_comp.split("\n")[-2].split()[2])
    ref_rot = float(ref_comp.split("\n")[-2].split()[2])
    print "Orientation deviation changed from %.4f to %.4f during refinement" % (init_rot, ref_rot)
    inits.append(init_rot)
    refs.append( ref_rot)


if not args.noabs:
    inits = np.abs(inits)
    refs = np.abs(refs)

plt.figure()
bins = np.linspace(args.brange[0],args.brange[1],args.nbins)
plt.hist( inits, bins, histtype='step', lw=2, label="Before refinement")
plt.hist( refs, bins, histtype='step', lw=2, label="After refinement")
plt.legend(prop=dict(size=13))
ax = plt.gca()
ax.tick_params(labelsize=14)
ax.set_xlabel("Crystal missetting differences in degrees", fontsize=14)
ax.set_title("Median orientation deviation: Before=%.3f, After=%.3f" \
            % (np.median(inits), np.median(refs)), fontsize=16)
plt.show()

