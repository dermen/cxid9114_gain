#!/usr/bin/env libtbx.python

"""
Separates structure factors
for protein sans heavy atom and
heavy atom, and decouples wavelength contribu
"""

from argparse import ArgumentParser
parser = ArgumentParser("scatt heavy TT")
parser.add_argument("-e", help="energy in eV", type=float)
parser.add_argument("-o", help='output name', type=str)
parser.add_argument('-p', help='pdb input file', type=str)
parser.add_argument("-d", default=1.5, type=float, help='high reso limit for table')
args = parser.parse_args()

from cxid9114 import utils
from cxid9114.sim import scattering_factors
from iotbx import pdb
from scitbx.array_family import flex

d_min = args.d
pdbin = pdb.input(args.p)
xr = pdbin.xray_structure_simple()
sc = xr.scatterers()
sym = [s.element_symbol() for s in sc]
yb_pos = [i for i, s in enumerate(sym) if s == 'Yb']

#print "Positions of Yb:"
#print yb_pos
# Lookup the anomolous contributions TODO: consider doing this for all atoms and checking the difference

print "Looking up anomalous contributions for all atoms except the Yb!"
for i, s in enumerate(sc):
    if i in yb_pos:
        continue
    fp, fdp = scattering_factors.elem_fp_fdp_at_eV(
        s.element_symbol(),
        args.e,
        how='henke')
    s.fp = fp
    s.fdp = fdp

#fp, fdp = scattering_factors.Yb_fp_fdp_at_eV(energy_eV, 'henke')
#sc[yb_pos[0]].fp = fp
#sc[yb_pos[0]].fdp = fdp
#sc[yb_pos[1]].fp = fp
#sc[yb_pos[1]].fdp = fdp

xr2_sel = flex.bool(len(sc), True)
for pos in yb_pos:
    xr2_sel[pos] = False
xr2 = xr.select(xr2_sel)
sc2 = xr2.scatterers()
print "Computing F protein"
Fp_sansHeavy = xr2.structure_factors(d_min=d_min,
                           algorithm='direct',
                           anomalous_flag=True).f_calc()

yb_sel = flex.bool(len(sc), False)
for pos in yb_pos:
    yb_sel[pos] = True
xr3 = xr.select(yb_sel)
sc3 = xr3.scatterers()
print "Computing F heavy"
Fa = xr3.structure_factors(d_min=d_min,
                           algorithm='direct',
                           anomalous_flag=True).f_calc()

Ap = Fp_sansHeavy.data().as_numpy_array()
Aa = Fa.data().as_numpy_array()

print "Computing F total including all anom contr"

print 'adding in anom scattering for yb'
for i, s in enumerate(sc):
    if i in yb_pos:
        fp, fdp = scattering_factors.elem_fp_fdp_at_eV(
            s.element_symbol(),
            args.e,
            how='henke')
        s.fp = fp
        s.fdp = fdp

#fp, fdp = scattering_factors.Yb_fp_fdp_at_eV(energy_eV, 'henke')
#sc[yb_pos[0]].fp = fp
#sc[yb_pos[0]].fdp = fdp
#sc[yb_pos[1]].fp = fp
#sc[yb_pos[1]].fdp = fdp

#xr2_sel = flex.bool(len(sc), True)
#for pos in yb_pos:
#    xr2_sel[pos] = False
#xr2 = xr.select(xr2_sel)

Ftot = xr.structure_factors(d_min=d_min, algorithm='direct', anomalous_flag=True).f_calc()
Atot = Ftot.data().as_numpy_array()

#print "Ftotal and Fprot + Fheavy should be equalish!"
#assert (np.allclose(Atot, Aa+Ap))

out = {"Aprotein": Fp_sansHeavy, "Aheavy": Fa, "Atotal": Ftot}
utils.save_flex(out, args.o)
