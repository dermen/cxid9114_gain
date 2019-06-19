# USE THIS CODE TO HELP ASSIGN COLORS
# TO SPOTS THAT ARE SOMEWHAT NEIGHBORS

from collections import Counter
from IPython import embed

import numpy as np
from scipy.spatial import distance

from cxid9114 import utils
from cxid9114.geom import geom_utils


def fix_edge_cases(d):

    refls_data = d['refls_data']
    resA = d['residA']
    resB = d['residB']
    det = d['detector']
    beamA = d['beamA']
    beamB = d['beamB']

#   its assumed hereafter that beamA is the lower energy beam
    assert( beamA.get_wavelength() > beamB.get_wavelength())

    spot_h = map( tuple, resA['hkl'])

    h_multiples = \
        [h for h,count in Counter(spot_h).items() 
            if  count > 1]

    idxA,idxB = resA['indexed'], resB['indexed']
    print "spots by A only: %d" % sum(np.logical_and(idxA, ~idxB))
    print "spots by B only: %d" % sum(np.logical_and(~idxA, idxB))
    print "spots by A and B: %d" % sum(np.logical_and(idxA, idxB))

    fudge=.5  # 1/2 pixel fudge factor for pixels being considered close
    for h,k,l in h_multiples:
        is_same = [ all((hh==h, kk==k,ll==l)) for hh,kk,ll in spot_h]
        if sum( is_same) > 2:  # NOTE: not sure what to do with >2 refls..
            print "Boo!"
            ii = np.where( is_same)[0]
            print  ii
            resA['indexed'][ii] = False
            resB['indexed'][ii] = False
            continue

        # the reflection indices!
        i1,i2 = np.where( is_same)[0]  # NOTE: guarenteed to be 2 here
        
        # the reflection data
        r1,r2 = refls_data[i1], refls_data[i2]
        pid1, pid2 = r1['panel'], r2['panel']
        if pid1 != pid2:
            print "on separate panels, weird"
            print  i1,i2
            resA['indexed'][[i1,i2]] = False
            resB['indexed'][[i1,i2]] = False
            continue

        pid = pid1 = pid2
        node = det[pid]
        pixsize = node.get_pixel_size()[0]
        max_deltapix = geom_utils.twocolor_deltapix(
            node, beamA, beamB)

        x1,y1,_ = r1['xyzobs.px.value'] 
        x2,y2,_ = r2['xyzobs.px.value']
        deltapix = distance.euclidean( (x1,y1),(x2,y2))

        if deltapix > max_deltapix+fudge:
            print " not close enough, not sure what to do"
            print i1,i2
            resA['indexed'][[i1,i2]] = False
            resB['indexed'][[i1,i2]] = False
            continue

        # NOTE: if we made it this far, the two 
        # spots are likely neighboring slit spots 
        #  and we should assign i1 / i2 to inner, outer
        lab1 = node.get_pixel_lab_coord( (x1,y1))
        lab2 = node.get_pixel_lab_coord( (x2,y2))

        lab_cent = node.get_beam_centre_lab( beamA.get_s0())  
        # NOTE : can use beamA or beamB so long as they are colinear
        # to define cent

        radial1 = distance.euclidean( lab1,lab_cent )
        radial2 = distance.euclidean( lab2,lab_cent )
       
        if abs( radial1 - radial2) / pixsize < max_deltapix /2.:
            # NOTE: if the spots are close but not radially separated...
            print "close by not cigar"
            print i1,i2
            resA['indexed'][[i1,i2]] = False
            resB['indexed'][[i1,i2]] = False
            continue

        if radial1 < radial2:
            inner = i1
            outer = i2
        else:
            inner = i2
            outer = i1
        # was the ref indexed by colorA/B
        inner_byA = resA['indexed'][inner]
        inner_byB = resB['indexed'][inner]
        outer_byA = resA['indexed'][outer]
        outer_byB = resB['indexed'][outer]
        
        # NOTE we care about the cases when byA and byB are both True 
        # for the same h
        if inner_byA and inner_byB and outer_byA and outer_byB:
            # NOTE in this case assign the inner peak to B
            resA['indexed'][inner] = False
            resB['indexed'][outer] = False
            print "case 1" 
            print i1,i2, inner, outer
        
        elif inner_byA and inner_byB and outer_byA and not outer_byB:
            print "case 2"
            resA['indexed'][inner] = False
            print i1,i2, inner, outer
        elif not inner_byA and inner_byB and outer_byA and outer_byB:
            print "case 3"
            resB['indexed'][outer] = False
            print i1,i2, inner, outer
        
        else: # NOTE: in all other cases, they shouldnt arise, so set indexed to false
            print "case 0"
            resA['indexed'][[i1,i2]] = False
            resB['indexed'][[i1,i2]] = False
            print i1,i2, inner, outer

    idxA,idxB = resA['indexed'], resB['indexed']
    print "spots by A only: %d"% sum(np.logical_and(idxA, ~idxB))
    print "spots by B only: %d"% sum(np.logical_and(~idxA, idxB))
    print "spots by A and B: %d"% sum(np.logical_and(idxA, idxB))

    return d

if __name__=="__main__":
    d = utils.open_flex('t2.dumpsy')
    d2 = fix_edge_cases(d)


