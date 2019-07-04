#!/usr/bin/env libtbx.python

from argparse import ArgumentParser

parser = ArgumentParser("try karl solvers")
parser.add_argument("-i", type=str, help='input_file')
parser.add_argument("-o", type=str, help='output mtz')
parser.add_argument("-sim", action='store_true')
parser.add_argument("-weights", action='store_true')
parser.add_argument("-N", type=int, default=None, help="Nshots max")
args = parser.parse_args()

import numpy as np
from IPython import embed
from cxid9114.solvers.karl_solver import karl_solver

karl_solver(np.load(args.i))
embed()

