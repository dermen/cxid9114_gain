#!/usr/bin/env libtbx.python

from itertools import cycle
import argparse

import pylab as plt
import numpy as np

import dxtbx
from dxtbx.model.crystal import CrystalFactory
from dials.algorithms.indexing.compare_orientation_matrices import \
    rotation_matrix_differences


parser = argparse.ArgumentParser("Toggle two dxtbx bigsim images")
parser.add_argument('-i', nargs=2, dest='i',type=str,  required=True, help='images to toggle' )
parser.add_argument('-t',dest='t',type=float, default=0.5, help='toggle time interval in seconds' )
parser.add_argument('--binary-thresh', dest='binary', default=None, type=float, help='binary threshold comparison' )
parser.add_argument('-vmin', dest='vmin', type=float,  default=None, help='vmin')
parser.add_argument('-vmax', dest='vmax', type=float,  default=None, help='vmax')
args = parser.parse_args()

loader0 = dxtbx.load(args.i[0])
img0 = loader0.get_raw_data().as_numpy_array()
loader1 = dxtbx.load(args.i[1])
img1 = loader1.get_raw_data().as_numpy_array()

cryst_descrA = {'__id__': 'crystal',
              'real_space_a': loader0._h5_handle["real_space_a"][()],
              'real_space_b': loader0._h5_handle["real_space_b"][()],
              'real_space_c': loader0._h5_handle["real_space_c"][()],
              'space_group_hall_symbol': loader0._h5_handle["space_group_hall_symbol"][()]} 

cryst_descrB = {'__id__': 'crystal',
              'real_space_a': loader1._h5_handle["real_space_a"][()],
              'real_space_b': loader1._h5_handle["real_space_b"][()],
              'real_space_c': loader1._h5_handle["real_space_c"][()],
              'space_group_hall_symbol': loader1._h5_handle["space_group_hall_symbol"][()]} 

CrystalA = CrystalFactory.from_dict(cryst_descrA)
CrystalB = CrystalFactory.from_dict(cryst_descrB)

from IPython import embed
embed()
rot_diffs = rotation_matrix_differences((CrystalA, CrystalB))
print(rot_diffs)

if args.binary is not None:
    img0 = img0 > args.binary
    img1 = img1 > args.binary

imgs = cycle([img0,img1])

plt.figure()
ax = plt.gca()
im = ax.imshow(img0, vmin=args.vmin, vmax=args.vmax, cmap='gnuplot')

while 1:
    im.set_data(imgs.next())
    plt.draw()
    plt.pause(args.t)

