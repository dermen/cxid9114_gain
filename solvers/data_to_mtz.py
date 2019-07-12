# coding: utf-8

import numpy as np
from cctbx import sgtbx, crystal
from cctbx.array_family import flex
from cctbx import miller
from cxid9114 import utils


Nh = ES.Nhkl
IA = ES.helper.x[:Nh].as_numpy_array()
IB = ES.helper.x[Nh:2*Nh].as_numpy_array()

dataA = np.load(args.i)
hkl_map = dataA["hkl_map"][()]
hkl_map2 = {v:k for k,v in hkl_map.iteritems()}
Nhkl = len(hkl_map)
assert( Nh==Nhkl)
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
    
waveA = ENERGY_CONV/8944.
waveA = ENERGY_CONV/9034
IA = mil_ar 
out = IA.as_mtz_dataset(column_root_label="Iobs", title="B", wavelength=waveA)
out.add_miller_array(miller_array=IA.average_bijvoet_mates(), column_root_label="IMEAN")
obj = out.mtz_object()
obj.write(args.o)


