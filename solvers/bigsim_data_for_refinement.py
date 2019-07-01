#!/usr/bin/env libtbx.python

from argparse import ArgumentParser

parser = ArgumentParser("test")
parser.add_argument("-i", help='input pickle', type=str, required=True)
parser.add_argument("-o", help='output npz', type=str, required=True)
parser.add_argument("-hanom", help='hAnom are defined in pickle, otherwise uses h2', action='store_true')
args = parser.parse_args()

import pandas
import numpy as np
from itertools import izip
import sys
#from cxid9114 import utils
from cxid9114.bigsim import sim_spectra
#from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW
import h5py

spec = sim_spectra.load_spectra("bigsim_rocketships/test_sfall.h5")
spec_f = h5py.File("bigsim_rocketships/test_data.h5", "r")
en_bins = spec_f["energy_bins"][()]
ilow = 19  # abs(en_bins - ENERGY_LOW).argmin()
ihigh = 110  #abs(en_bins - ENERGY_HIGH).argmin()
SA = spec[ilow].amplitudes()  # energy A (lower energy)
SB = spec[ihigh].amplitudes()  # energy B (higher energy)

# load the integrated data
df = pandas.read_pickle(args.i)

Kfact = df.K.values[0]

if args.hanom:
    hkey = ['hAnom', 'kAnom', 'lAnom']
else:
    hkey = ['h2', 'k2', 'l2']

hkl = tuple(map(tuple,  df[hkey].values.astype(int)))
hkl_map = {h: i for i, h in enumerate(set(hkl))}
hkl_idx = [hkl_map[h] for h in hkl]  # assigns a sparse matrix row ID for hkl

df['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in \
                    izip( df.shot_idx, df.run)]

shot_map = {s: i for i,s in enumerate(set(df.shot_loc.values))}
shot_idx = [shot_map[s] for s in df.shot_loc]  # assigns sparse matrix row ID for scale factors..

Nh = len(set(hkl_idx))
Ns = len(set(shot_idx))
print ("2x %d hlk and 2x %d shots = %d UNKNOWNS" % (Nh, Ns, 2*(Nh+Ns)))
print "%d MEASUREMENTS"%len(df)
df['hkl_idx'] = hkl_idx

gb = df.groupby(hkey)

SA_map = {SA.indices()[i]: SA.data()[i] for i in range(len(SA.indices()))}
SB_map = {SB.indices()[i]: SB.data()[i] for i in range(len(SB.indices()))}


I = []
IA = []
IB = []
Nmissed = 0
for h in set(hkl):
    I.append(np.random.normal(5000*gb.get_group(h).D.mean(), 100))
    IA.append(abs(SA_map[h])**2)
    IB.append(abs(SB_map[h])**2)
    #try:
    #    IA.append(abs(SA_map[h])**2)
    #except KeyError:
    #    IA.append(1000)
    #    Nmissed += 1
    #try:
    #    IB.append(abs(SB_map[h])**2)
    #except KeyError:
    #    IB.append(1000)
    #    Nmissed += 1


np.savez(args.o,
        IAprm=IA, IBprm=IB,
        Iprm=I, hkl_map=hkl_map, hkl_idx=hkl_idx,
        ydata=df.D.values, hkl=hkl,
        gains = df.gain,
        FAdata=df.FA, FBdata=df.FB,
        #ynoise=df.Dnoise.values,
        LAdata=df.LA, LBdata=df.LB, PAdata=df.PA/Kfact, PBdata=df.PB/Kfact,
        adata=hkl_idx, gdata=shot_idx)

