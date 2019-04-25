
from cxid9114 import utils
from cxid9114.sim import scattering_factors
import numpy as np
from iotbx import pdb
from scitbx.array_family import flex
import sys

energy_eV = float(sys.argv[1])
output_name = sys.argv[2]

lines = open('4bs7.pdb', "r").readlines()
pdbin = pdb.input(source_info=None, lines=lines)
xr = pdbin.xray_structure_simple()
sc = xr.scatterers()
sym = [s.element_symbol() for s in sc]
yb_pos = [i for i, s in enumerate(sym) if s == 'Yb']

# Lookup the anomolous contributions TODO: consider doing this for all atoms and checking the difference

print "Looking up anomalous contributions for all atoms!"
for i, s in enumerate(sc):
    if i in yb_pos:
        continue
    fp, fdp = scattering_factors.elem_fp_fdp_at_eV(
        s.element_symbol(),
        energy_eV,
        how='henke')
    s.fp = fp
    s.fdp = fdp

#fp, fdp = scattering_factors.Yb_fp_fdp_at_eV(energy_eV, 'henke')
#sc[yb_pos[0]].fp = fp
#sc[yb_pos[0]].fdp = fdp
#sc[yb_pos[1]].fp = fp
#sc[yb_pos[1]].fdp = fdp

xr2_sel = flex.bool(len(sc), True)
xr2_sel[yb_pos[0]] = False
xr2_sel[yb_pos[1]] = False
xr2 = xr.select(xr2_sel)
sc2 = xr2.scatterers()
print "Computing F protein"
Fp = xr2.structure_factors(d_min=2,
                           algorithm='direct',
                           anomalous_flag=True).f_calc()

yb_sel = flex.bool(np.logical_not(xr2_sel.as_numpy_array()))
xr3 = xr.select(yb_sel)
sc3 = xr3.scatterers()
print "Computing F heavy"
Fa = xr3.structure_factors(d_min=2,
                           algorithm='direct',
                           anomalous_flag=True).f_calc()

Ap = Fp.data().as_numpy_array()
Aa = Fa.data().as_numpy_array()

print "Computing F total"
Ftot = xr.structure_factors(d_min=2, algorithm='direct', anomalous_flag=True).f_calc()
Atot = Ftot.data().as_numpy_array()

print "These should be equalish!"
assert (np.allclose(Atot, Aa+Ap))

out = {"Aprotein": Fp, "Aheavy": Fa,"Atotal": Ftot }
utils.save_flex(out, output_name)

