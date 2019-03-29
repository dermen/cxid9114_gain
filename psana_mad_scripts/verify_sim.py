from cctbx import miller, sgtbx
from cxid9114 import utils
sg96 = sgtbx.space_group(" P 4nw 2abw")
FA = utils.open_flex('SA.pkl')
FB = utils.open_flex('SB.pkl')
HA = tuple([hkl for hkl in FA.indices()])
HB = tuple([hkl for hkl in FB.indices()])
HA_val_map = { hkl:FA.value_at_index(hkl) for hkl in HA}
HB_val_map = { hkl:FB.value_at_index(hkl) for hkl in HB}

def get_val_at_hkl(hkl, val_map):
    poss_equivs = [i.h() for i in
        miller.sym_equiv_indices(sg96,hkl).indices()]
    for hkl2 in poss_equivs:
        if hkl2 in val_map:  # fast lookup
            break
    return val_map[hkl2]

G = d["gain"]
LA,LB = d["flux_data"]
K = (1e4)**2 * 1e12
I=[]
I2=[]
for i_r in range(Nrefl):
    idxA = residA["indexed"][i_r]
    idxB = residB["indexed"][i_r]
    hklA = tuple(residA['hkl'][i_r])
    hklB = tuple(residB['hkl'][i_r])
    if idxA:
        PA = residA["sim_intens"][i_r]
    else: PA=0
    if idxB:
        PB = residB["sim_intens"][i_r]
    else: PB=0
    IA = abs(get_val_at_hkl(hklA, HA_val_map) )**2
    IB = abs(get_val_at_hkl(hklB, HB_val_map) )**2
    RHS = G*(IA*LA*(PA/K) + IB*LB*(PB/K))
    LHS = d["refls_data"][i_r]["intensity.sum.value"]
    I.append(LHS)
    I2.append(RHS)
    if not all([ ia==ib for ia,ib in zip(hklA,hklB)]):
        print i_r, hklA, hklB


