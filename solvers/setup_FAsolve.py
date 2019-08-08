from argparse import ArgumentParser

parser = ArgumentParser("solve FA fixed")
parser.add_argument("-i", type=str, help="input data file")
parser.add_argument("-p", type=str, help="input parameter file from F+- refinements")
parser.add_argument("-o", type=str, help="output npz file")
parser.add_argument("-FA", type=str, help="npy storing dict of FA map")
args = parser.parse_args()

import numpy as np

karlA = np.load("../sim/karl_plain8944.npz")
karlB = np.load("../sim/karl_plain9034.npz")

print "loading FT"
FT = karlA["FT"][()]
print "loading FA"
#FA = karlA["FA"][()]
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

data = np.load(args.i)
FF_out = np.load(args.p)
Fheavy_map = np.load(args.FA)[()]
Nh = FF_out["Nh"]

FwaveA = FF_out["x"][:Nh]
FwaveB = FF_out["x"][Nh:2*Nh]
FT_init = np.vstack([np.sqrt(np.exp(FwaveA)), np.sqrt(np.exp(FwaveB))]).mean(0)

hkl_map_old = data["hkl_map"][()]
assert(Nh == len(hkl_map_old))
hkl2_map_old = {i: h for h, i in hkl_map_old.items()}

FT_map = {}
for i in range(Nh):
    h = hkl2_map_old[i]
    hpos = tuple(np.abs(h))
    val = FT_init[i]
    FT_map[h] = val
    FT_map[hpos] = val

hkl = map(tuple, data["hkl"][()])
hkl_pos = [(abs(h), abs(k), abs(l)) for h, k, l in hkl]
is_pos = [1 if all([h >= 0, k >= 0, l >= 0]) else -1 for h, k, l in hkl]

U_hkl = set(hkl)
U_hkl_pos = set(hkl_pos)
print "%d uique HKL and %d unique, positive HKL" % (len(U_hkl), len(U_hkl_pos))
hkl_pos_map = {h: i for i, h in enumerate(U_hkl_pos)}

Aidx = [hkl_pos_map[h] for h in hkl_pos]

a = [a_enA[h] for h in U_hkl_pos]
b = [b_enA[h] for h in U_hkl_pos]
c = [c_enA[h] for h in U_hkl_pos]

a2 = [a_enB[h] for h in U_hkl_pos]
b2 = [b_enB[h] for h in U_hkl_pos]
c2 = [c_enB[h] for h in U_hkl_pos]

Fprot_tru = [abs(FT[h]) for h in U_hkl_pos]
alpha_tru = [ALPHA[h] for h in U_hkl_pos]

FA_fix = [abs(Fheavy_map[h]) for h in U_hkl_pos]
PhiA_fix = [np.angle(Fheavy_map[h]) for h in U_hkl_pos]
FT_init = [abs(FT_map[h]) for h in U_hkl_pos]

Fprot_tru = [abs(FT[h]) for h in U_hkl_pos]
alpha_tru = [ALPHA[h] for h in U_hkl_pos]

Gain_fix = FF_out["x"][2*Nh:]

np.savez(args.o,
        a_enA=a,
        b_enA=b,
        c_enA=c,
        a_enB=a2,
        b_enB=b2,
        c_enB=c2,
        Fprot_prm=FT_init,
        Fheavy_prm=FA_fix,
        alpha_prm=np.random.permutation(alpha_tru),
        Gain_prm=Gain_fix,
        Fprot_tru=Fprot_tru,
        Fheavy_tru=FA_fix,
        alpha_tru=alpha_tru,
        Gain_tru=Gain_fix,
        Yobs=data["ydata"],
        hkl_pos=hkl_pos, hkl_pos_map=hkl_pos_map,
        LA=data["LAdata"], LB=data["LBdata"], PA=data["PAdata"], PB=data["PBdata"],
        is_pos=is_pos,
        Aidx=Aidx, Gidx=data["gdata"],
        PhiA_fix=PhiA_fix)

