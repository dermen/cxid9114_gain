import pandas
import numpy as np
from itertools import izip

df = pandas.read_pickle(
    "rocketships/all_the_goodies_wReso_and_anom_corrected.pkl")

# somehow duplicates ended up in dataframe, need to figure out why..
df = df.loc[~df.duplicated()].reset_index()

Kfact = 1e20

df = df.query("reso > 2.4")


from cxid9114.utils import is_outlier
out_thresh=45
df = df.loc[~is_outlier(df.Dnoise,out_thresh)]


hkl = tuple(map(tuple,  df[['hAnom', 'kAnom', 'lAnom']].values.astype(int)))
hkl_map = {h: i for i, h in enumerate(set(hkl))}
hkl_idx = [hkl_map[h] for h in hkl]  # assigns a sparse matrix row ID for hkl

df['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in \
                    izip( df.shot_idx, df.run)]

shot_map = {s: i for i,s in enumerate(set(df.shot_loc.values))}
shot_idx = [shot_map[s] for s in df.shot_loc]  # assigns sparse matrix row ID for scale factors..

Nh = len(set(hkl_idx))
Ns = len(set(shot_idx))
print ("2x %d hlk and 2x %d shots = %d UNKNOWNS" % (Nh, Ns, 2*(Nh+Ns)))
print len(df)
df['hkl_idx'] = hkl_idx

gb = df.groupby(['hAnom', 'kAnom', 'lAnom'])

import sys
from cxid9114 import utils
SA = utils.open_flex("SA.pkl")
SA_map = {SA.indices()[i]: SA.data()[i] for i in range(len(SA.indices()))}
SB = utils.open_flex("SB.pkl")
SB_map = {SB.indices()[i]: SB.data()[i] for i in range(len(SB.indices()))}

I = []
IA = []
IB = []
for h in set(hkl):
    I.append(np.random.normal(5000*gb.get_group(h).D.mean(), 100))
    try:
        IA.append(abs(SA_map[h])**2)
    except KeyError:
        IA.append(1000)
    try:
        IB.append(abs(SB_map[h])**2)
    except KeyError:
        IB.append(1000)


np.savez(sys.argv[1],
        IAprm=IA, IBprm=IB,
        Iprm=I, hkl_map=hkl_map, hkl_idx=hkl_idx,
        ydata=df.D.values, hkl=hkl,
        ynoise=df.Dnoise.values,
        LA=df.LA, LB=df.LB, PA=df.PA/Kfact, PB=df.PB/Kfact,
        adata=hkl_idx, gdata=shot_idx)

