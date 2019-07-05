#!/usr/bin/env libtbx.python

from argparse import ArgumentParser

parser = ArgumentParser("test")
parser.add_argument("-i", help='input pickle', type=str, required=True)
parser.add_argument("-o", help='output npz', type=str, required=True)
parser.add_argument("-thresh", type=float, default=35.5)
parser.add_argument("-dmin", help="resolution min", type=float, default=None)
parser.add_argument("-hanom", help='hAnom are defined in pickle, otherwise uses h2', action='store_true')
parser.add_argument("-min-pix", dest='deltapix_min', type=float,
                    default=None)
parser.add_argument("-j", help='number of jobs', default=4, type=int)
parser.add_argument('-rotmin', default=None, type=float)
parser.add_argument('--make-shot-index', action='store_true', dest='shot_index')
parser.add_argument('--filt', action='store_true', dest='filt')
parser.add_argument('--gauss', action='store_true')
args = parser.parse_args()

import pandas
from joblib import Parallel, delayed
import numpy as np
from itertools import izip
import sys
#from cxid9114 import utils
from cxid9114.bigsim import sim_spectra
#from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW
import h5py
from cxid9114.solvers.filter_outliers import filter_outliers

spec = sim_spectra.load_spectra("bigsim_rocketships/test_sfall.h5")
spec_f = h5py.File("bigsim_rocketships/test_data.h5", "r")
en_bins = spec_f["energy_bins"][()]
ilow = 19  # abs(en_bins - ENERGY_LOW).argmin()
ihigh = 110  #abs(en_bins - ENERGY_HIGH).argmin()
SA = spec[ilow].amplitudes()  # energy A (lower energy)
SB = spec[ihigh].amplitudes()  # energy B (higher energy)

# load the integrated data
df = pandas.read_pickle(args.i)

df.reset_index(inplace=True)

if args.hanom:
    hkey = ['hAnom', 'kAnom', 'lAnom']
else:
    hkey = ['h2', 'k2', 'l2']

a,b,c,_,_,_ = SA.unit_cell().parameters()

h,k,l = df[hkey].values.T
df['reso'] = np.sqrt(1./ (h**2/a**2 + k**2/b**2 + l**2/c**2))

if args.shot_index:
    df['shot_idx'] = df.data_name.str.split("_").map(lambda x: x[1]).astype(int)

if args.dmin is not None:
    print "dmin"
    N = len(df)
    df = df.query("reso > %d" % args.dmin)
    print "\t\tremoved %d ( %d left)\n" % (N-len(df), len(df))

if args.rotmin is not None:
    print "rot min"
    N = len(df)
    ang_dev = df.init_rot.abs()
    df = df.query("@ang_dev < %f" % args.rotmin)
    print "\t\tremoved %d ( %d left)\n" % (N-len(df), len(df))

if args.deltapix_min is not None:
    print "delta pix min"
    N = len(df)
    df = df.query("delta_pix < %f" % args.deltapix_min)
    print "\t\tremoved %d ( %d left)\n" % (N-len(df), len(df))


df.reset_index(inplace=True)

Kfact = df.K.values[0]

hkl = tuple(map(tuple,  df[hkey].values.astype(int)))
hkl_map = {h: i for i, h in enumerate(set(hkl))}
hkl_idx = [hkl_map[h] for h in hkl]  # assigns a sparse matrix row ID for hkl

df['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in \
                    izip( df.shot_idx, df.run)]

shot_map = {s: i for i,s in enumerate(set(df.shot_loc.values))}
shot_loc_idx = [shot_map[s] for s in df.shot_loc]  # assigns sparse matrix row ID for scale factors..

df['shot_loc_idx'] = shot_loc_idx

Nh = len(set(hkl_idx))
Ns = len(set(shot_loc_idx))
print ("2x %d hlk and 1 x %d shots = %d UNKNOWNS" % (Nh, Ns, 2*Nh+Ns))
print "%d MEASUREMENTS"%len(df)
df['hkl_idx'] = hkl_idx

gb = df.groupby(hkey)

SA_map = {SA.indices()[i]: SA.data()[i] for i in range(len(SA.indices()))}
SB_map = {SB.indices()[i]: SB.data()[i] for i in range(len(SB.indices()))}

n_jobs=args.j
split_h = np.array_split( np.vstack(set(hkl)), n_jobs)


def main(jid):
    print_stride=40
    I = []
    IA = []
    IB = []
    new_dfs = []
    pFits = []
    covs = []
    jid_h = split_h[jid]
    Nh = len( jid_h)
    for i_h, h in enumerate(jid_h):
        h = tuple(h)
        dhkl = gb.get_group(h)

        I_guess, dhkl_filt, pFit, cov = filter_outliers(
            dhkl, fit_gauss=args.gauss,
            nsig=3, thresh=args.thresh,
            use_median=True)
        dhkl_filt["multiplicity"] = len(dhkl_filt)

        pFits.append(pFit)
        covs.append(cov)

        if args.filt:
            new_dfs.append(dhkl_filt)
        else:
            new_dfs.append(dhkl)

        I.append(I_guess)
        IA_truth = abs(SA_map[h])**2
        IB_truth = abs(SB_map[h])**2

        IA.append(IA_truth)
        IB.append(IB_truth)

        if i_h % print_stride==0:
            print "JOB {:d} Refl {:d}/{:d}  --  {:d}, {:d}, {:d} : IA truth={:.3f} ; IB truth={:.3f};  IA guess={:.3f}" \
                .format( jid, i_h+1, Nh, h[0], h[1], h[2], IA_truth, IB_truth, I_guess)

    return IA, IB, I, pFits, covs, new_dfs

results = Parallel(n_jobs=n_jobs)(delayed(main)(i) for i in range(n_jobs) )


print "combining results"
I = []
IA = []
IB = []
new_dfs = []
pFits = []
covs = []
for r in results:
    IA+= r[0]
    IB+= r[1]
    I+= r[2]
    pFits += r[3]
    covs += r[4]
    new_dfs += r[5]

print "Concat-ing the dataframes"
N = len(df)
df = pandas.concat(new_dfs)
print("Filtered %d out of %d rows" % (N-len(df), N))

if not args.filt:
    df['weights'] = 1

np.savez(args.o,
        IAprm=IA, IBprm=IB,
        Iprm=I, hkl_map=hkl_map, hkl_idx=hkl_idx,
        ydata=df.D.values, hkl=hkl,
        gains = df.gain,
        FAdata=df.FA.abs(), FBdata=df.FB.abs(),
        #ynoise=df.Dnoise.values,
        Weights=df.weights,
        LAdata=df.LA, LBdata=df.LB, PAdata=df.PA/Kfact, PBdata=df.PB/Kfact,
        adata=df.hkl_idx, gdata=df.shot_loc_idx)

