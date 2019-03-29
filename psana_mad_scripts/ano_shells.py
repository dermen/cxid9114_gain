# coding: utf-8
from cxid9114 import utils
import numpy as np
from cctbx import crystal, miller, sgtbx
from dials.array_family import flex
from operator import itemgetter
import pandas

res_lims = np.array([
             np.inf,  31.40920442,  15.7386434 ,  10.53009589,
         7.93687927,   6.38960755,   5.36511518,   4.63916117,
         4.0996054 ,   3.68413423,   3.35535065,   3.08945145,
         2.87056563,   2.68770553,   2.53302433,   2.40077038,
         2.28663331,   2.1873208 ,   2.10027621,   2.02348524,
         1.95534054,   1.89454478,   1.84004023,   1.79095656])

sg96 = sgtbx.space_group(" P 4nw 2abw")
Symm = crystal.symmetry( 
    unit_cell=(79,79,38,90,90,90), 
    space_group=sg96)

df = pandas.read_hdf(fname, "reflections")

k = 10000**2

que = ["AnotB", "BnotA", "AandB"]

hstr = [["hA", "kA", "lA"], 
        ["hB", "kB", "lB"], 
        ["hB", "kB", "lB"]]

all_hkl = []
all_PA = []
all_PB = []
all_G = []
all_LA = []
all_LB = []
all_yobs = []
all_shotid = []
for iq, q in enumerate(que):
    df_q = df.query(q)
    hkls = map(tuple,df_q[hstr[iq]].values)
    all_hkl += hkls
    all_PA.append( df_q.intensA.values)
    all_PB.append( df_q.intensB.values)
    all_LA.append( df_q.channA_intens.values)
    all_LB.append( df_q.channB_intens.values)
    all_G.append( df_q.gain.values)  # this is an unknown!
    all_yobs.append( df_q.intens.values)
    # this will serve to set up the Jacobian later,
    # as we know scale factor G depends on the run+shot index
    all_shotid += map(tuple, df_q[['run', 'shot_idx']].values)

all_G = np.hstack( all_G)
all_LA = np.hstack( all_LA)
all_LB = np.hstack( all_LB)
all_PA = np.hstack(all_PA)
all_PB = np.hstack(all_PB)
all_yobs = np.hstack( all_yobs)
print "Tuple-izing..."
all_hkl = tuple( all_hkl)
print "Tuple-izing..."
all_shotid = tuple( all_shotid)

print "Row mapping"
# always map to the same index for symm equivs
# just sort all equivs by h,k,l then  return first sorted idx
Hequiv_map = { hkl: 
    sorted(
        [h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], 
        key=itemgetter(0,1,2)
        )[0]  # take the first item in sorted array of equivalents
    for hkl in all_hkl}

unique_hkl = set(Hequiv_map.values())
Nhkl = len( unique_hkl)
# we need to map each gain and each Fhkl to 
# a column index for the Jacobian!
rowmap_hkl = {hkl: i for i,hkl in enumerate(unique_hkl)}
# (NOTE: we have two unknown Fhkl per measurement, one per energy channel)

unique_shotid = set(all_shotid)
Ngain = len( unique_shotid)
rowmap_gain = {shotid: i for i,shotid in enumerate(unique_shotid)}

Gdata = [rowmap_gain[shotid] for shotid in all_shotid]
Adata = [rowmap_hkl[ Hequiv_map[hkl] ] for hkl in all_hkl]

h,k,l = np.array(all_hkl).T
run,shot_idx = np.array(all_shotid).T
new_data = {"Gdata": Gdata, "Adata": Adata, 
        "Ydata": all_yobs, "Gain": all_G, "LA": all_LA, 
        "LB":all_LB, "PA": all_PA, "PB": all_PB,
        "h":h, "k":k, "l":l,"run": run, "shot_idx":shot_idx}

new_df = pandas.DataFrame(new_data)
new_df.to_hdf("fitme_simdata.hdf5", "data")


