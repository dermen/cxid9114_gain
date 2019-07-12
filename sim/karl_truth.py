#!/usr/bin/env libtbx.python
#from argparse import ArgumentParser
#parser = ArgumentParser(" karl truth")
#parser.add_argument('--single',
#            help='use single channel only FT/FA',action='store_true')
#args = parser.parse_args()

import pandas
import numpy as np
from IPython import embed


df = pandas.read_pickle("perf2.pkl")
df.reset_index(inplace=True, drop=True)

karlA = np.load("karl_8944.npz")
karlB = np.load("karl_9034.npz")

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


hkey = ['h2','k2', 'l2']

gb = df.groupby(hkey)

unique_h = set( tuple(map(tuple,df[hkey].values)) )

print "Processing %d uniuqe HKL" % len(unique_h)

embed()

for i_h, h in enumerate(unique_h):
    print i_h, len(unique_h)
    df_h = gb.get_group(h)
    prot = abs(FT[h])
    heav = abs(FA[h])
    alpha = ALPHA[h]
    G = df_h.gain
    PA = df_h.PA/df_h.K
    PB = df_h.PB/df_h.K
    LB = df_h.LB
    LA = df_h.LA
    Aterm = PA*LA*(prot**2 + heav**2 * a_enA[h] + prot*heav*b_enA[h]*np.cos(alpha) +
                   prot*heav*c_enA[h]*np.sin(alpha))
    Bterm = PB*LB*(prot**2 + heav**2*a_enB[h] + prot*heav*b_enB[h]*np.cos(alpha) +
                   prot*heav*c_enB[h]*np.sin(alpha))
    rhs = G * (Aterm + Bterm)
    df.loc[df_h.index, "new_rhs"] = rhs
    #df_h['new_rhs'] = rhs

