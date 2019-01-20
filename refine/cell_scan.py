from copy import deepcopy
from scitbx.matrix import sqr
import numpy as np
from cxid9114.sim import sim_utils
from cxid9114.spots import spot_utils

def refine_cell(data):

    spot_mask = spot_utils.strong_spot_mask( data['refl'], (1800,1800))

    Patts = sim_utils.PatternFactory()
    Patts.adjust_mosaicity(2,0.5)
    energy, fcalc = sim_utils.load_fcalc_file(data['fcalc_f'])

    crystal = data['crystal']
    a,b,c,_,_,_ = crystal.get_unit_cell().parameters()

    optX, optY = data['optX'], data['optY']
    optX_fine, optY_fine = data['optX_fine'], data['optY_fine']
    Op = optX_fine * optY_fine * optX * optY
    crystal.set_U(Op)

    overlaps = []
    imgs_all = []
    percs = np.arange( -0.005, 0.006, 0.001)
    crystals = []
    for i in percs:
        for j in percs:
            crystal2 = deepcopy(crystal)
            a2 = a + a*i
            c2 = c + c*j
            B2 = sqr((a2, 0, 0, 0, a2, 0, 0, 0, c2)).inverse()
            crystal2.set_B(B2)
            sim_patt = Patts.make_pattern2(crystal=crystal2,
                                       flux_per_en=[data['fracA']*1e14, data['fracB']*1e14],
                                       energies_eV=energy,
                                       fcalcs_at_energies=fcalc,
                                       mosaic_spread=None,
                                       mosaic_domains=None,
                                       ret_sum=True,
                                       Op=None)

            sim_sig_mask = sim_patt > 0
            overlaps.append( np.sum(sim_sig_mask * spot_mask))
            crystals.append( deepcopy(crystal2))
            imgs_all.append( sim_patt)
    refls_all = [data["refl"]] * len( imgs_all)
    utils.images_and_refls_to_simview("cell_refine", imgs_all, refls_all)
    return overlaps, crystals



if __name__ == "__main__":
    import sys
    from cxid9114 import utils
    data = utils.open_flex(sys.argv[1])
    results, crystals = refine_cell(data)

    from IPython import embed
    embed()
