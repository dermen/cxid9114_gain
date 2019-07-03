# coding: utf-8

import numpy as np
from cctbx import sgtbx, crystal
from cctbx.array_family import flex
from cctbx import miller
from cxid9114 import utils

data = np.load("_autogen_niter0.npz")
Nh = data["Nhkl"]
IA = data["x"][:Nh]


dataA = np.load("real_dataA.npz")
hkl_map = dataA["hkl_map"][()]
hkl_map2 = {v:k for k,v in hkl_map.iteritems()}
Nhkl = len(hkl_map)
assert( Nh==Nhkl)
print Nh
hout, Iout = [],[]
for i in range(Nhkl):
    h = hkl_map2[i]
    val = IA[i]
    hout.append(h)
    Iout.append(np.exp(val))

sg = sgtbx.space_group(" P 4nw 2abw")
Symm = crystal.symmetry( unit_cell=(79,79,38,90,90,90), space_group=sg)
hout = tuple(hout)
mil_idx = flex.miller_index(hout)
mil_set = miller.set(crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)
Iout_flex = flex.double(np.ascontiguousarray(Iout))
mil_ar = miller.array(mil_set, data=Iout_flex).set_observation_type_xray_intensity()
utils.save_flex(mil_ar, "n00b_begin.pkl")

