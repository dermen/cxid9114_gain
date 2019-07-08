#!/usr/bin/env libtbx.python

from argparse import ArgumentParser

parser = ArgumentParser("test")
parser.add_argument("-i", help='input pickle', type=str, required=True)
parser.add_argument("-o", help='output npz', type=str, required=True)
parser.add_argument("-dmin", help="resolution min", type=float, default=None)
parser.add_argument("-min-pix", dest='deltapix_min', type=float,
                    default=None)
parser.add_argument('-rotmin', default=None, type=float)
parser.add_argument('--make-shot-index', action='store_true', dest='shot_index')
parser.add_argument('-p', type=float, default=None, help='perturbation factor')
args = parser.parse_args()


import pandas
from itertools import izip
import numpy as np


df = pandas.read_pickle(args.i)
hkey = ['h2', 'k2', 'l2']


a,b,c = 79,79,38

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

df.reset_index(inplace=True,drop=True)


karlA = np.load("../sim/karl_8944.npz")
karlB = np.load("../sim/karl_9034.npz")

# load FT and FA from one of the energy channels only, as
# they should be independent of energy channel
print "loading FT"
FT = karlA["FT"][()]
print "loading FA"
FA = karlA["FA"][()]
print "loading ALPHA"
ALPHA = karlA["ALPHA"][()]

print "enA; loading a,b,c constants, probably faster to compute them on the fly... "
a_enA = karlA["A"][()]
b_enA = karlA["B"][()]
c_enA = karlA["C"][()]

print "enB; loading a,b,c constants, probably faster to compute them on the fly... "
a_enB = karlB["A"][()]
b_enB = karlB["B"][()]
c_enB = karlB["C"][()]

hkl = tuple(map(tuple,  df[hkey].values.astype(int)))
U_hkl = set(hkl)
hkl_map = {h: i for i, h in enumerate(U_hkl)}

a = [a_enA[h] for h in U_hkl]
b = [b_enA[h] for h in U_hkl]
c = [c_enA[h] for h in U_hkl]

a2 = [a_enB[h] for h in U_hkl]
b2 = [b_enB[h] for h in U_hkl]
c2 = [c_enB[h] for h in U_hkl]

df['Aidx'] = [hkl_map[h] for h in hkl]  # assigns a sparse matrix row ID for hkl

df['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in \
                    izip( df.shot_idx, df.run)]

shot_map = {s: i for i,s in enumerate(set(df.shot_loc.values))}
df['Gidx'] = [shot_map[s] for s in df.shot_loc]  # assigns sparse matrix row ID for scale factors..

Nh = df.Aidx.unique().shape[0]
Ns = df.Gidx.unique().shape[0]
print ("3x %d hlk and 1 x %d shots = %d UNKNOWNS" % (Nh, Ns, 3*Nh+Ns))
print "%d MEASUREMENTS"%len(df)

Fprot_tru = [abs(FT[h]) for h in U_hkl]
Fheav_tru = [abs(FA[h]) for h in U_hkl]
alpha_tru = [ALPHA[h] for h in U_hkl]
Gtru = np.random.uniform(1, 10, Ns)

Fprot_prm = np.array(Fprot_tru)
Fheav_prm = np.array(Fheav_tru)
alpha_prm = np.array(alpha_tru)
Gprm = np.array(Gtru)

# set the per shot gain values
assert(np.all(np.unique(df.Gidx.values) == np.arange(Ns)))
for gi in range(Ns):
    dfG = df.Gidx == gi
    df.loc[dfG, "D"] = df.loc[dfG, "D"] * Gprm[gi]

if args.p is not None:
    alpha_prm = np.random.permutation(alpha_prm)
    Fprot_prm = np.exp(np.random.uniform(np.log(Fprot_prm)-args.p, np.log(Fprot_prm)+args.p))
    Fheav_prm = np.random.uniform(min(Fheav_prm), max(Fheav_prm), len(Fheav_prm))
    Gprm = np.random.uniform(min(Gprm), max(Gprm), Ns)

np.savez(args.o,
        a_enA=a,
        b_enA=b,
        c_enA=c,
        a_enB=a2,
        b_enB=b2,
        c_enB=c2,
        Fprot_prm=Fprot_prm,
        Fheavy_prm=Fheav_prm,
        alpha_prm=alpha_prm,
        Gain_prm=Gprm,
        Fprot_tru=Fprot_tru,
        Fheavy_tru=Fheav_tru,
        alpha_tru=alpha_tru,
        Gain_tru=Gtru,
        Yobs=df.D.values,
        hkl=hkl, hkl_map=hkl_map,
        LA=df.LA, LB=df.LB, PA=df.PA/df.K, PB=df.PB/df.K,
        Aidx=df.Aidx, Gidx=df.Gidx)

