
import glob
import sys
from collections import Counter
from cxid9114 import utils
from scipy.spatial import distance
from cxid9114 import utils
import numpy as np
from scipy.spatial import KDTree

from cxid9114.parameters import ENERGY_CONV

def waveB_fromA(delta_r, xy, detnode, beamA):
    # delta_r is in mm, xy is in pixels, detnode is dxtbx DetNode, 
    # beamA is dxtbx beam
    reso = detnode.get_resolution_at_pixel( beamA.get_s0(),xy)
    twothetaA = detnode.get_two_theta_at_pixel(beamA.get_s0(),xy)
    dist = detnode.get_distance()  # returns in mm
    waveB = 2*reso*np.sin(.5*np.arctan(np.tan(twothetaA)-delta_r/dist))

    return waveB, ENERGY_CONV/waveB

run = int( sys.argv[1])
nom_enA = int(sys.argv[2])

waveA_default = ENERGY_CONV/float(nom_enA)

fnames = glob.glob("results/run%d_alpha1alpha2/prepped*dumpsy2" % run)

offsets = {}
for pid in range(64):
    offsets[pid] = []
all_f, all_iA, all_iB = [],[],[]
waveBs,enBs,del_ens = [],[],[]
del_spot_rad, del_spot_abs = [],[]
resos = []
for f in fnames:
    d = utils.open_flex(f)
    rA = d['residA']
    rB = d['residB']
    beamA = d['beamA']
    beamA.set_wavelength(waveA_default)
    beamB = d['beamB']
    det = d['detector']
    refls = d['refls_data']
    idxA = rA['indexed']
    idxB = rB['indexed']
    hkl = rA['hkl']
    tree = KDTree(hkl)
    pairs =  tree.query_pairs(1e-3)
    for i1,i2 in map(list,pairs):
        if not idxA[i1] and not idxA[i2] and not idxB[i1] and not idxB[i2]:
            continue
        elif idxA[i1] and not idxA[i2] and not idxB[i1] and idxB[i2]:
            refA = refls[i1]
            refB = refls[i2]
            iA = i1
            iB = i2
        elif idxA[i2] and not idxA[i1] and not idxB[i2] and idxB[i1]:
            refA = refls[i2]
            refB = refls[i1]
            iA = i2
            iB = i1
        else:
            print idxA[[i1,i2]], idxB[[i1,i2]]
            print "Spots should already be separated, I should not be here!"
            assert(False)
        
        if refA['panel'] != refB['panel']:
            print( "Spots should be on same panel,"\
                "should already have been checked!")
            assert(False)
        pid = refA['panel']
        node = det[pid]
        pixsize = node.get_pixel_size()[0]
        xA,yA,_ = refA['xyzobs.px.value'] 
        xB,yB,_ = refB['xyzobs.px.value']
        deltapix = distance.euclidean( (xA,yA),(xB,yB))
        
        # spot A should always be further radially from the beam
        # because it is lower energy!
        labA = node.get_pixel_lab_coord( (xA,yA))
        labB = node.get_pixel_lab_coord( (xB,yB))

        lab_cent = node.get_beam_centre_lab( beamA.get_s0())  

        radialA = distance.euclidean( labA,lab_cent )
        radialB = distance.euclidean( labB,lab_cent )
        
        if radialB > radialA:
            print "rB must be less than rA, find the bugs!" 
            assert(False)
        
        del_spot = radialA - radialB

        #offsets[pid].append((del_spot, deltapix*pixsize))
        waveB, enB = waveB_fromA(del_spot, (xA,yA), node, beamA) 
        delta_en = enB - ENERGY_CONV / beamA.get_wavelength() 
        reso = node.get_resolution_at_pixel( beamA.get_s0(),(xA,yA)) 
        
        resos.append( reso)
        waveBs.append(waveB)
        enBs.append(enB)
        del_ens.append( delta_en)
        all_f.append( f)
        all_iA.append( iA)
        all_iB.append( iB)
        del_spot_rad.append( del_spot)
        del_spot_abs.append( deltapix * pixsize)
        print enB
    
np.savez( "run%d_waveBs_nomA%d" % (run, nom_enA), offsets=offsets, 
    all_iA=all_iA, all_iB=all_iB, all_f=all_f,
    enBs=enBs, del_ens = del_ens, del_spot_rad=del_spot_rad,
    del_spot_abs = del_spot_abs, resos=resos)



