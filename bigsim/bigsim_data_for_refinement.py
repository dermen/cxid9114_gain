#!/usr/bin/env libtbx.python

import sys
import argparse
from itertools import izip

import pandas
import numpy as np
from cxid9114 import utils
import h5py

from cxid9114.bigsim import sim_spectra
from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW


parser = argparse.ArgumentParser("Ymodel and Yobs")
parser.add_argument("-i", dest='i', help='input integration pickle', type=str, required=True)
parser.add_argument("-o", dest='o', help='outputfile prefix (prefix.npz) ', 
        type=str, default=None)
parser.add_argument("-plot", dest='plot', action='store_true', help='plot Ymodel vs Yobs')
parser.add_argument("-anom", dest='anom', action='store_true', help='group h by anom')
args = parser.parse_args()

SFall = sim_spectra.load_spectra("test_sfall.h5")
# need to load the structure factors at the channel A and B wavelengths
spec_f = h5py.File("simMe_data_run62.h5", "r")
en_bins = spec_f["energy_bins"][()]
ilow = abs(en_bins - ENERGY_LOW).argmin()
ihigh = abs(en_bins - ENERGY_HIGH).argmin()
SA = SFall[ilow].amplitudes()  # energy A (lower energy)
SB = SFall[ihigh].amplitudes()  # energy B (higher energy)

# load the integrated data
df = pandas.read_pickle(args.i)

Kfact = 1e20

if args.anom:
    hkl = tuple(map(tuple,  df[['hAnom', 'kAnom', 'lAnom']].values.astype(int)))
else:
    hkl = tuple(map(tuple,  df[['h', 'k', 'l']].values.astype(int)))
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

gb = df.groupby(['hAnom', 'kAnom', 'lAnom'])

SA_map = {SA.indices()[i]: SA.data()[i] for i in range(len(SA.indices()))}
SB_map = {SB.indices()[i]: SB.data()[i] for i in range(len(SB.indices()))}

I = []
IA = []
IB = []
Nmissed = 0
for h in set(hkl):
    I.append(1000) #np.random.normal(5000*gb.get_group(h).D.mean(), 100))
    try:
        IA.append(abs(SA_map[h])**2)
    except KeyError:
        IA.append(1000)
        Nmissed += 1
    try:
        IB.append(abs(SB_map[h])**2)
    except KeyError:
        IB.append(1000)
        Nmissed += 1

IA_data = []
IB_data =[]
for i in hkl_idx:
    IA_data.append( IA[i])
    IB_data.append( IB[i])

df["IA"] = IA_data
df["IB"] = IB_data

from IPython import embed
embed()

if args.plot:
    import pylab as plt
    Yobs = df.D
    Ymod = df.LA*df.IA*df.PA/Kfact + df.LB*df.IB*df.PB/Kfact

    plt.plot( Yobs, Ymod, '.', ms=1)
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")
    plt.show()


if args.o is not None:
    np.savez(args.o,
        IAprm=IA, IBprm=IB,
          Iprm=I, hkl_map=hkl_map, hkl_idx=hkl_idx,
        ydata=df.D.values, hkl=hkl,
          ynoise=df.Dnoise.values,
        LA=df.LA, LB=df.LB, PA=df.PA/Kfact, PB=df.PB/Kfact,
          adata=hkl_idx, gdata=shot_idx)

