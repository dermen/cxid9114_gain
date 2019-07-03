
from itertools import izip

from IPython import embed
import numpy as np
cos = np.cos
sin = np.sin

from cctbx import crystal, miller, sgtbx
from cctbx.array_family import flex
from scitbx import matrix

from cxid9114 import utils
from cxid9114.sim import scattering_factors, helper

# load anom terms for Yb at given energy
fp, fdp = scattering_factors.Yb_fp_fdp_at_eV(8944, 'henke')

# load the data dictionary, 3 complex miller arrays
D = utils.open_flex("karl.pkl")

sg96 = sgtbx.space_group(" P 4nw 2abw")


Fprot = helper.generate_table(D["Aprotein"].data(), D["Aprotein"].indices() )
Fheav = helper.generate_table(D["Aheavy"].data(), D["Aheavy"].indices() )
Ftot = helper.generate_table(D["Atotal"].data(), D["Atotal"].indices() )

hkl = np.load("hkl_karl_test.npy").astype(int)
hkl = tuple(map(tuple,hkl))
hkl = np.array(list(set(hkl)))

B = matrix.sqr((79, 0, 0, 0, 79, 0, 0, 0, 38)).inverse()
B = B.as_numpy_array()


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
    elif np.any(H == 0):
        continue
    else:
        print "what am I doing here? ", H  
        continue 

    alpha = np.angle(prot) - np.angle(heav)
    Ipp = abs(prot)**2
    Ihh = abs(heav)**2
    Iph = abs(prot)*abs(heav)
    #if hand==1:
    karl = Ipp + a * Ihh + Iph * b * cos(alpha) + hand * c * Iph * sin(alpha)
    #else:
    #karl = Ipp + a * Ihh + Iph * b * cos(alpha) - hand * c * Iph * sin(alpha)
    karl2 = Ipp + a * Ihh + Iph * b * cos(alpha) - hand * c * Iph * sin(alpha)
   
    if hand==1: 
        resid = np.abs(np.abs(tot) - np.sqrt(karl))
    else:
        resid = np.abs(np.abs(tot) - np.sqrt(karl2))
    if resid > 1:
        print "H (%d/%d): %d %d %d ; res=%.1f , Ftot=%.3f, Fkarl=%.3f, Fkarl2=%.3f" % \
            (i_h+1, len(hkl), H[0], H[1], H[2], res, abs(tot), np.sqrt(karl), np.sqrt(karl2))


embed()
