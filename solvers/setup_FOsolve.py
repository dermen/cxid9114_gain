from argparse import ArgumentParser

parser = ArgumentParser("solve FA fixed")
parser.add_argument("-i", type=str, help="input data file")
parser.add_argument("-p", type=str, help="input parameter file from F+- refinements")
parser.add_argument("-o", type=str, help="output npz file")
parser.add_argument("-FA", type=str, help="npy storing dict of FA map")
args = parser.parse_args()

import numpy as np
from IPython import embed
import pandas


d = np.load(args.FA)
h_vals = d["h"]
al = d["alpha"]
fa = d["fa"]
alpha_map = {}
fa_map = {}
for i in range(len(h_vals)):
    hkl = tuple(h_vals[i])
    alpha_map[hkl] = al[i]
    fa_map[hkl] = fa[i]


# READ IN VALUES FROM PREVIOUS REFINEMENT
data = np.load(args.i)
FF_out = np.load(args.p)
Nh = FF_out["Nh"]

FwaveA = FF_out["x"][:Nh]
FwaveB = FF_out["x"][Nh:2*Nh]
FT_init = np.vstack([np.sqrt(np.exp(FwaveA)), np.sqrt(np.exp(FwaveB))]).mean(0)

Gain_fix = FF_out["x"][2*Nh:]

# MAKE A TABLE OF FT amplitude initial values
hkl_map_old = data["hkl_map"][()]
assert(Nh == len(hkl_map_old))
hkl2_map_old = {i: h for h, i in hkl_map_old.items()}

FT_init_map = {}
for i in range(Nh):
    h = hkl2_map_old[i]
    hpos = tuple(np.abs(h))
    val = FT_init[i]
    FT_init_map[h] = val
    FT_init_map[hpos] = val

hkl = map(tuple, data["hkl"][()])
h,k,l = zip(*hkl)
df = pandas.DataFrame({"LA": data["LAdata"],
                       "LB": data["LBdata"],
                       "PA": data["PAdata"],
                       "PB": data["PBdata"],
                       "Yobs": data["ydata"],
                       "gdata": data["gdata"],
                       "adata": data["adata"],
                        "h": h,
                       "k": k,
                       "l": l})

# iterate over the hkl values from the SOLVE output (only positive h)
gb_hkl = df.groupby(["h", "k", "l"])
hkeys = gb_hkl.groups.keys()
sub_dfs = []

for i_hkl, (h, k, l) in enumerate(h_vals):
    if i_hkl %20 == 0:
        print "Processing hkl %d / %d" % (i_hkl+1, len(h_vals))
    df_hkl = gb_hkl.get_group((h, k, l)).copy()
    if len(df_hkl) >0:
        df_hkl.reset_index(drop=True, inplace=True)
        df_hkl["is_pos"] = 1
        df_hkl["gain_true"] = Gain_fix[df_hkl.gdata]
        df_hkl["adata"] = i_hkl
        sub_dfs.append(df_hkl)

    if any([h ==0, k==0, l==0]):
        #print "Centric reflection, no neg"
        continue
    hneg = (-h,-k,-l)

    if hneg not in hkeys:
        #print "Not in hkeys"
        continue

    df_hkl_neg = gb_hkl.get_group(hneg).copy()

    if len(df_hkl_neg) > 0:
        df_hkl_neg.reset_index(drop=True, inplace=True)
        df_hkl_neg["is_pos"] = -1
        df_hkl_neg["gain_true"] = Gain_fix[df_hkl_neg.gdata]
        df_hkl_neg["adata"] = i_hkl
        sub_dfs.append(df_hkl_neg)


print "Combining all sub dataframes"
new_df = pandas.concat(sub_dfs)
new_df.reset_index(inplace=True,drop=True)

shot_map = {s:i for i,s in enumerate(new_df.gdata.unique())}
new_Gain_fix = [Gain_fix[s] for s in new_df.gdata.unique()]
new_df['new_gdata'] = [shot_map[s] for s in new_df.gdata]

print "Tuplizing"
h_vals = map(tuple, h_vals)

hkl_map = {h: i for i,h in enumerate(h_vals)}

#hkl_pos = [(abs(h), abs(k), abs(l)) for h, k, l in hkl]
#is_pos = [1 if all([h >= 0, k >= 0, l >= 0]) else -1 for h, k, l in hkl]

#U_hkl = set(hkl)
#h_vals = set(hkl_pos)
#print "%d uique HKL and %d unique, positive HKL" % (len(U_hkl), len(U_hkl_pos))
#hkl_pos_map = {h: i for i, h in enumerate(U_hkl_pos)}

#Aidx = [hkl_pos_map[h] for h in hkl_pos]

karlA = np.load("../sim/karl_plain8944.npz")
karlB = np.load("../sim/karl_plain9034.npz")

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


a = [a_enA[h] for h in h_vals]
b = [b_enA[h] for h in h_vals]
c = [c_enA[h] for h in h_vals]

a2 = [a_enB[h] for h in h_vals]
b2 = [b_enB[h] for h in h_vals]
c2 = [c_enB[h] for h in h_vals]

Fprot_tru = [abs(FT[h]) for h in h_vals]
alpha_tru = [ALPHA[h] for h in h_vals]

PhiA_fix = [np.angle(FA[h]) for h in h_vals]
FT_init = [abs(FT_init_map[h]) for h in h_vals]

FA_fix = [abs(fa_map[h]) for h in h_vals]
alpha_fix = [alpha_map[h] for h in h_vals]

Fprot_tru = [abs(FT[h]) for h in h_vals]
alpha_tru = [ALPHA[h] for h in h_vals]

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
        Gain_prm=new_Gain_fix,
        Fprot_tru=Fprot_tru,
        Fheavy_tru=FA_fix,
        alpha_fix=alpha_fix,
        alpha_tru=alpha_tru,
        Gain_tru=new_Gain_fix,
        Yobs=new_df.Yobs,
        hkl_map=hkl_map,
        LA=new_df.LA, LB=new_df.LB, PA=new_df.PA, PB=new_df.PB,
        is_pos=new_df.is_pos,
        Aidx=new_df.adata, Gidx=new_df.new_gdata,
        PhiA_fix=PhiA_fix)
