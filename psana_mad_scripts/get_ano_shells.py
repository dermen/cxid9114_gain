# coding: utf-8
from cxid9114 import utils
import numpy as np
from cctbx import miller


res_lims = np.array([
             np.inf,  31.40920442,  15.7386434 ,  10.53009589,
         7.93687927,   6.38960755,   5.36511518,   4.63916117,
         4.0996054 ,   3.68413423,   3.35535065,   3.08945145,
         2.87056563,   2.68770553,   2.53302433,   2.40077038,
         2.28663331,   2.1873208 ,   2.10027621,   2.02348524,
         1.95534054,   1.89454478,   1.84004023,   1.79095656])

a = utils.open_flex('SA.pkl')
b = utils.open_flex('SB.pkl')
H = [a.indices()[i] for i in len(a)]

def res_from_idx(h,k,l,a=79.,b=79.,c=38.):
    return 1./ np.sqrt( h*h/a/a + k*k/b/b + l*l/c/c)
RES = [res_from_idx(*h) for h in H]

sg96 = a.space_group()

#from operator import itemgetter
#Hequiv_map = { hkl: sorted([h.h() 
#    for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2))[:] for hkl in H}

HA_val_map = { hkl:a.value_at_index(hkl) for hkl in H}
HB_val_map = { hkl:b.value_at_index(hkl) for hkl in H}

a2 = a.select_acentric()
H2 = [i for i in a2.indices()]
RES2 = [res_from_idx(*h) for h in H2]
bin_ass2 = np.digitize( RES2, bins=res_lims)

def get_neg_equiv(hkl):
    poss_equivs = [i.h() for i in 
        miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
    
    for hkl2 in poss_equivs:
        if hkl2 in HA_val_map:  # fast lookup
            break
    return hkl2

FdelA = []
FdelB = []
for i,hkl in enumerate(H2):
    hkl2 = get_neg_equiv(hkl)
    if i%200==0:
        print i, len(H2), hkl, hkl2
    F1 = HA_val_map[hkl]
    F0 = HA_val_map[hkl2]
    FdelA.append( np.abs(F1)-np.abs(F0))
    F1 = HB_val_map[hkl]
    F0 = HB_val_map[hkl2]
    FdelB.append( np.abs(F1)-np.abs(F0))
    
    
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[g]
    gB = FdelB[g]
    top = np.mean( gA*gB)
    bottom= np.sqrt( np.mean(gA**2))*np.sqrt(np.mean(gB**2))
    coeffs.append(top/bottom)
    print top/bottom
    

# total content    
FAvals = np.array([a.value_at_index(hkl) for hkl in H])
FBvals = np.array([b.value_at_index(hkl) for hkl in H])
Acontent = np.mean(np.abs(FdelA)) / np.mean(np.abs(FAvals))
Bcontent = np.mean(np.abs(FdelB)) / np.mean(np.abs(FBvals))

