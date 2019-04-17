
import sys
from cxid9114 import utils
from cxid9114.geom import geom_utils
from dxtbx.model.experiment_list import  ExperimentListFactory
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import numpy as np
import scipy.ndimage
from cxid9114.refine import metrics
from cctbx import miller, sgtbx
from cxid9114 import utils

exp_name = sys.argv[1]
data_name = sys.argv[2]  
tag = sys.argv[3]
hkl_tol = .15

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors

# load sf for the data, contains wavelength dependence!
FFdat = [utils.open_flex("SA.pkl"), utils.open_flex("SB.pkl")]


FLUX = [1e12, 1e12]  # fluxes of the beams

flux_frac = np.random.uniform(.2,.8)
chanA_flux = flux_frac*1e12
chanB_flux = (1.-flux_frac)*1e12
FLUXdat = [chanA_flux, chanB_flux]
GAIN = np.random.uniform(0.5,3)

waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
exp_lst = ExperimentListFactory.from_json_file(exp_name) #, check_format=False)
iset = exp_lst.imagesets()[0]
detector = iset.get_detector(0)
data = utils.open_flex( data_name)
beamA = deepcopy(iset.get_beam())
beamB = deepcopy(iset.get_beam())
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

crystalAB = data["crystalAB"]

simsAB = sim_utils.sim_twocolors2(
    crystalAB, detector, iset.get_beam(0), FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=1, mos_spread=0.0)

simsData = sim_utils.sim_twocolors2(
    crystalAB, detector, iset.get_beam(0), FFdat,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUXdat, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=1, mos_spread=0.)

simsDataSum = GAIN * ( np.array(simsData[0]) + np.array(simsData[1]))

refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA) 
refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB) 

# This only uses the beam to instatiate an imageset / datablock
# but otherwise the return value (refl_data) is indepent of the 
# beam object passed
refl_data = spot_utils.refls_from_sims(simsDataSum, detector, beamA) 
residA = metrics.check_indexable2(
    refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
residB = metrics.check_indexable2(
    refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)


sg96 = sgtbx.space_group(" P 4nw 2abw")
FA = utils.open_flex('SA.pkl')
FB = utils.open_flex('SB.pkl')
HA = tuple([hkl for hkl in FA.indices()])
HB = tuple([hkl for hkl in FB.indices()])
HA_val_map = { hkl:FA.value_at_index(hkl) for hkl in HA}
HB_val_map = { hkl:FB.value_at_index(hkl) for hkl in HB}


d = {"crystalAB": crystalAB,
        "residA": residA,
        "residB": residB,
        "beamA": beamA,
        "beamB": beamB,
        "detector": detector,
        "refls_simA": refl_simA,
        "refls_simB": refl_simB,
        "flux_data": FLUXdat,
        "gain": GAIN,
        "refls_data": refl_data}

def get_val_at_hkl(hkl, val_map):
    poss_equivs = [i.h() for i in
        miller.sym_equiv_indices(sg96,hkl).indices()]
    for hkl2 in poss_equivs:
        if hkl2 in val_map:  # fast lookup
            break
    return val_map[hkl2]


rpp = spot_utils.refls_by_panelname(refl_data)
rppA = spot_utils.refls_by_panelname(refl_simA)
rppB = spot_utils.refls_by_panelname(refl_simB)

for pid in rpp:
    R = rpp[pid]
    RA = rppA[pid]
    RB = rppB[pid]
    x,y,_ = spot_utils.xyz_from_refl(R)
    x = np.array(x)
    y = np.array(y)
    
    xA,yA,_ = spot_utils.xyz_from_refl(RA)
    xB,yB,_ = spot_utils.xyz_from_refl(RB)
    
    points = np.array(zip(x,y))
    pointsA = np.array(zip(xA,yA))
    pointsB = np.array(zip(xB,yB))
    tree = cKDTree(points)
    treeA = cKDTree(pointsA)
    treeB = cKDTree(pointsB)
    rmax = geom_utils.twocolor_deltapix(detector[pid], beamA, beamB)
    merge_me = treeA.query_ball_tree( treeB, r=rmax+6)
    
    HA, HiA,QA = spot_utils.refls_to_hkl(
        RA, detector, beamA,
        crystal=crystalAB, returnQ=True )

    HB, HiB, QB = spot_utils.refls_to_hkl(
        RB, detector, beamB,
        crystal=crystalAB, returnQ=True )
    
    panX,panY = detector[pid].get_image_size()
    boxes = []
    mergesA = []
    mergesB = []
    int_meAB = []
    for iA,iB in enumerate(merge_me):
        if not iB:
            continue
        iB = iB[0]
        x1A,x2A,y1A,y2A,_,_ = RA[iA]['shoebox'].bbox
        x1B,x2B,y1B,y2B,_,_ = RB[iB]['shoebox'].bbox
        
        xlow = max( [0, min((x1A, x1B))-sz  ] )
        xhigh = min( [panX, max((x2A, x2B))+sz   ])
        ylow = max([0, min((y1A, y1B)) -sz   ])
        yhigh = min([panY, max((y2A,y2B)) + sz])
        boxes.append( [xlow, xhigh, ylow, yhigh])
        print pid,xlow, xhigh,ylow, yhigh
        
        int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
        if int_me.size:
            int_meAB.append( int_me)
        mergesA.append(iA)
        mergesB.append(iB)


    boxesA = []
    int_meA = []
    for iA,ref in enumerate(RA):
        if iA in mergesA:
            continue
        x1A,x2A,y1A,y2A,_,_ = RA[iA]['shoebox'].bbox
        xlow = max((0,x1A-sz))
        xhigh = min((panX, x2A+sz))
        ylow = max((0,y1A-sz))
        yhigh = min((panY, y2A+sz))
        boxesA.append( (xlow, xhigh, ylow, yhigh))
        int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]

        if int_me.size:
            int_meA.append( int_me)
    
    boxesB = []
    int_meB = []
    for iB,ref in enumerate(RB):
        if iB in mergesB:
            continue
        x1B,x2B,y1B,y2B,_,_ = RB[iB]['shoebox'].bbox
        xlow = max((0,x1B-sz)) 
        xhigh = min((panX, x2B+sz))
        ylow = max((0,y1B-sz)) 
        yhigh = min((panY, y2B+sz))
        boxesB.append( (xlow, xhigh, ylow, yhigh))
        #subimg = simsDataSum[pid][ylow:yhigh, xlow:xhigh]
        #bg = 0
        int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
        
        if int_me.size:
            int_meB.append( int_me)
    #print pid, [ (HiA[i], HiB[val[0]]) for i,val in enumerate(merge_me) if val]
    print pid, int_meA, int_meB




Nrefl = len( d['refls_data'])
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

I = np.array(I)
I2 = np.array(I2)

from IPython import embed
embed()
#dump_name = data_name.replace(".pkl", "_%s.pkl" % tag)
#utils.save_flex(dump, dump_name)
#print "Wrote %s" % dump_name
