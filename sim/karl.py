#!/usr/bin/env libtbx.python

from argparse import ArgumentParser
parser = ArgumentParser("I am karl")
parser.add_argument("-i", help="input karl", type=str)
parser.add_argument("-o", help='output name', type=str)
parser.add_argument("-e", help='energy in eV', type=float)
parser.add_argument('--tom', action='store_true', help='use the tom terwilliger approach')
args =parser.parse_args()

from itertools import izip

import numpy as np
cos = np.cos
sin = np.sin

from scitbx import matrix

from cxid9114 import utils
from cxid9114.sim import scattering_factors, helper

# load anom terms for Yb at given energy
fp, fdp = scattering_factors.Yb_fp_fdp_at_eV(args.e, 'henke')

# load the data dictionary, 3 complex miller arrays
D = utils.open_flex(args.i)

Fprot = helper.generate_table(D["Aprotein"].data(), D["Aprotein"].indices() )
Fheav = helper.generate_table(D["Aheavy"].data(), D["Aheavy"].indices() )
Ftot = helper.generate_table(D["Atotal"].data(), D["Atotal"].indices() )

hkl = np.load("hkl_karl_test.npy").astype(int)
hkl = tuple(map(tuple,hkl))
hkl = np.array(list(set(hkl)))

B = matrix.sqr((79, 0, 0, 0, 79, 0, 0, 0, 38)).inverse()
B = B.as_numpy_array()

FT_map = {}
FA_map = {}
ALPHA_map = {}
A_map = {}
B_map = {}
C_map = {}

for i_h,H in enumerate(hkl):
    res = 1./np.linalg.norm(np.dot( B, H))
    Yb_f0 = scattering_factors.Yb_f0_at_reso(res)
    a = (fp*fp + fdp*fdp) / (Yb_f0[0]*Yb_f0[0])
    b = 2*fp/Yb_f0[0]
    c = 2*fdp/Yb_f0[0]

    _, prot = helper.get_val_at_hkl(H, Fprot)
    _, heav = helper.get_val_at_hkl(H, Fheav)
    _, tot = helper.get_val_at_hkl(H, Ftot)

    if any([prot is None, heav is None, tot is None]):
        print "Neg", H
        continue

    if np.all(H < 0):
        hand =- 1
        #alpha = np.pi - (np.angle(prot) - np.angle(heav))
    elif np.all(H > 0):
        hand = 1
        #alpha = np.angle(prot) - np.angle(heav)
    #elif np.any(H == 0):
    #    continue
    #else:
    #    print "what am I doing here? ", H
    #    continue

    alpha = np.angle(prot) - np.angle(heav)
    Ipp = abs(prot)**2
    Ihh = abs(heav)**2
    Iph = abs(prot)*abs(heav)
    
    #if hand==1:
    COS = cos(alpha)
    SIN = sin(alpha)
    if args.tom:
        karl = Ipp + (1 + a + b)*Ihh + (2*COS + b*COS + c*SIN)*Iph
    else: 
        karl = Ipp + a * Ihh + Iph * b * COS + c * Iph * SIN
    
    resid = np.abs(np.abs(tot) - np.sqrt(karl))
    if resid > 1:  # print high residuals for debugging
        print "H (%d/%d): %d %d %d ; reso=%.1f , Ftot=%.3f, Fkarl=%.3f, residual=%.3f" % \
            (i_h+1, len(hkl), H[0], H[1], H[2], res, abs(tot), np.sqrt(karl), resid)

    H = tuple(H)
    A_map[H] = a
    B_map[H] = b
    C_map[H] = c
    FT_map[H] = prot
    FA_map[H] = heav
    ALPHA_map[H] = alpha

np.savez(args.o, A=A_map, B=B_map, C=C_map, FT=FT_map, FA=FA_map, ALPHA=ALPHA_map)


