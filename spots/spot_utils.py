

import sys
from collections import Counter
from itertools import groupby
import cPickle as pickle
from copy import deepcopy
import numpy as np
from scipy import ndimage
MAX_FILT = ndimage.maximum_filter

from scitbx.matrix import sqr
from scipy.spatial import cKDTree
from dials.array_family import flex


#def multipanel_refls_to_hkl(refls, detector, beam, crystal):
#    panel_refls = refls_by_panelname(refls)
#    all_h = []
#    all_hi  = []
#    for pid in panel_refls:
#        prefls = panel_refls[pid]
#        h,hi = refls_to_hkl(prefls, detector[pid], beam, crystal)
#        all_h.append(h)
#        all_hi.append(hi)
#    return np.vstack(all_h), np.vstack(all_hi)


def refls_to_hkl(refls, detector, beam, crystal,
                 update_table=False):
    """
    convert pixel panel reflections to miller index data

    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param as_numpy_arrays: return data as numpy arrays
    :return: if as_numpy two Nx3 numpy arrays are returned
        (one for fractional and one for whole HKL)
        else dictionary of hkl_i (nearest) and hkl (fractional)
    """
    if 'rlp' not in list(refls.keys()):
        q_vecs = refls_to_q(refls, detector, beam, update_table=update_table)
    else:
        q_vecs = np.vstack([r['rlp'] for r in refls])
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot( Ai, q_vecs.T)
    HKLi = map( lambda h: np.ceil(h-0.5).astype(int), HKL)
    if update_table:
        refls['miller_index'] = flex.vec3_int(tuple(map(tuple, HKLi)))
    return np.vstack(HKL).T, np.vstack(HKLi).T


#def multipanel_refls_to_q(refls, detector, beam):
#    R = refls_by_panelname(refls)
#    all_q = []
#    for pid in R:
#        x,y,z = xyz_from_refl(R[pid])
#        q_vecs = xy_to_q(x,y,detector[pid], beam)
#        all_q.append( q_vecs)
#    return np.vstack( all_q)


def refls_to_q(refls, detector, beam, update_table=False,oldmethod=False):

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set([r['panel'] for r in refls])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    for r in refls:
        pid = r['panel']
        i_fs, i_ss,_ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append( s1)
        q_vecs.append( s1-beam.get_s0())

    if update_table:
        refls['s1'] = flex.vec3_double(tuple(map(tuple,s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple,q_vecs)))

    return np.vstack(q_vecs)


def xy_to_q(x,y, panel, beam, oldmethod=False):
    """
    convert pixel coords to q-vectors

    :param x,y:  pixel coordinates of spots as separate lists/arrays
                x should be the fast-scan coord
    :param panel: dxtbx detector Panel (detectorNode)
    :param beam:  dxtbx beam model
    :param oldmethod: whether to use method 1 below..
    :return: the Q-vectors corresponding to the spots
    """
    if oldmethod:
        pixsizeFS, pixsizeSS = panel.get_pixel_size()
        orig = np.array(panel.get_origin())
        ss = np.array(panel.get_slow_axis())
        fs = np.array(panel.get_fast_axis())
        s0 = np.array(beam.get_s0())  # this is already scaled by 1/wavelength
        pix_mm = np.vstack( [orig + fs*i*pixsizeFS + ss*j*pixsizeSS
                for i,j in zip(x,y)])
        s1 = pix_mm / np.linalg.norm( pix_mm,axis=1)[:,None] / beam.get_wavelength()
        q_vecs = (s1 - s0)
    else:
        #pix_mm = panel.pixel_to_millimeter( flex.vec2_double( zip(x,y)))
        #coords = panel.get_lab_coord( pix_mm)
        coords = flex.vec3_double(tuple(
            panel.get_pixel_lab_coord((i,j)) for i,j in zip(x,y) ))
        coords = coords / coords.norms()
        q_vecs = coords *(1./beam.get_wavelength()) - beam.get_s0()

        q_vecs = q_vecs.as_double().as_numpy_array().reshape((len(q_vecs), 3))

    return q_vecs


def spotdat_xyz_in_lab(spot_data, detector, xy_key="comIpos"):
    all_spot_lab = []
    pids = [i for i in range(64) if i in spot_data]
    for pid in pids:
        if spot_data[pid] is None:
            continue
        spotpix = spot_data[pid][xy_key]
        panel = detector[pid]
        all_spot_lab += [panel.get_pixel_lab_coord((xpix,ypix)) for ypix, xpix in spotpix]
    return np.array(all_spot_lab)


def refls_xyz_in_lab(refls, detector, xy_key="xyzobs.px.value"):
    all_refls_lab = []
    for i,r in enumerate(refls):
        panel_name = r['panel']
        panel = detector[panel_name]
        x,y,_ = r[xy_key]
        lab_xyz = panel.get_pixel_lab_coord((x,y))
        all_refls_lab.append( lab_xyz)
    return np.array(all_refls_lab)


def spots_from_sim(img, thresh=0, as_tuple=False):
    """
    :param img:  simtbx simulated image
    :param thresh:  threshold above which to look for spots
    :return:
    """
    labimg, nlab = ndimage.label(img > thresh)
    out = ndimage.center_of_mass(img, labimg, range(1, nlab))
    if as_tuple:  # this option here cause flex likes tuples
        y,x = zip(*out)
    else:  # as numpy array
        y, x = map(np.array, zip(*out))
    return x,y


def strong_spot_mask(refl_tbl, img_size):
    Nrefl = len( refl_tbl)
    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
              for i in range(Nrefl)]
    x1, x2, y1, y2, z1, z2 = zip(*[refl_tbl[i]['shoebox'].bbox
                                   for i in range(Nrefl)])
    spot_mask = np.zeros(img_size, bool)
    for i1, i2, j1, j2, M in zip(x1, x2, y1, y2, masks):
        slcX = slice(i1, i2, 1)
        slcY = slice(j1, j2, 1)
        spot_mask[slcY, slcX] = M == 5
    return spot_mask


def combine_refls(refl_tbl_lst):
    """
    concatenates a list of reflection tables
    and adjusts the bbox 4,5 elements according
    to the index position of each reflection table
    in the list
    :param refl_tbl_lst: list of flex reflection tables
    :return: concatenated list of tables as a single table
    """
    combined_refls = flex.reflection_table()
    for i_pattern, patt_refls in enumerate( refl_tbl_lst):
        patt_refls = deepcopy(patt_refls)
        for i_refl, refl in enumerate(patt_refls):
            bb = list(patt_refls['bbox'][i_refl])
            bb[4] = i_pattern
            bb[5] = i_pattern + 1
            patt_refls['bbox'][i_refl] = tuple(bb)
            sb = patt_refls['shoebox'][i_refl]
            sb_bb = list(sb.bbox)
            sb_bb[4] = i_pattern
            sb_bb[5] = i_pattern + 1
            sb.bbox = tuple( sb_bb)
            patt_refls['shoebox'][i_refl] = sb
        combined_refls.extend(patt_refls)
    return combined_refls


def xyz_from_refl(refl, key="xyzobs.px.value"):
    x,y,z = zip( * [refl[key][i] for i in range(len(refl))])
    return x,y,z


def add_xy_to_ax_as_patches(xy, patch_type, patch_sty, ax, sizes=None):
    import matplotlib as mpl
    patches = [patch_type(xy=(i_, j_), **patch_sty)
             for i_, j_ in xy]
    if sizes is not None:
        if not patch_type == mpl.patches.Circle:
            print("Cannot update size for non-circle patches")
        else:
            for i in range(len(patches)):
                patches[i].radius = sizes[i]

    patch_coll = mpl.collections.PatchCollection(patches, match_original=True)
    ax.add_collection(patch_coll)


def make_color_data_object(refls, beam, crystal, detector):
    """
    make the values expected in the `color_data` dictionary parameter passed to
    the `compute_indexability` method.
    :param refls:
    :param beam:
    :param crystal:
    :param detector:
    :return:
    """
    x,y,_ = xyz_from_refl(refls)
    spots = zip(x,y)
    hkl, hkli = refls_to_hkl(refls, detector, beam, crystal)
    data = {'spots': spots,
            'tree': cKDTree(spots),
            'H': hkl,
            'Hi': hkli,
            'beam': beam,
            'crystal': crystal,
            'detector': detector}
    return data


def compute_indexability(refls, color_data, hkl_tol=0.15):
    xdata, ydata, _ = xyz_from_refl(refls)
    spotsData = zip(xdata, ydata)

    # for each spot on the refls,
    # we want to determine the fractional hkl to compare with those
    # from the simulations
    data_H = {}
    for c, cdata in color_data.iteritems():
        data_hkl, data_hkli = refls_to_hkl(refls,
                                        cdata['detector'],
                                        cdata['beam'],
                                        cdata['crystal'])

        data_H[c] = {'H': data_hkl,'Hi': data_hkli}

    # iterate over each spot, and determine
    # - can the spot be indexed by the simulated spots
    # - if so, how which simulated colors can index the spot
    # if indexability is None, then the spot cant be indexed!
    indexability = []
    for i_spot, spot in enumerate(spotsData):
        # determine which colors might index the spot

        can_index = {}
        for color, cdata in color_data.iteritems():
            # for this spot, get the nearest whole hkl for this color
            dist, idx = cdata['tree'].query(spot)
            whole_hkl = cdata['Hi'][idx]

            # get the fractional hkl for this specific spot, at this specific color
            spot_hkl = data_H[color]['H'][i_spot]

            resid = np.abs(spot_hkl - whole_hkl)
            if not np.all(resid < hkl_tol):
                continue

            # else, if color can index, then record
            can_index[color] = {'dist': dist,
                                'spot': cdata['tree'].data[idx],
                                'hkli': whole_hkl,
                                'hkl': spot_hkl,
                                'resid': resid}

        if not can_index:
            print("Cannot index the spot: %d" % i_spot)
            indexability.append(None)
            continue

        elif len(can_index) == 1:
            print("Only one color can index the spot %d" % i_spot)
            # sum_pixels_in_spot
        else:
            print("Multiple colors can index the spot %d." % i_spot)
            # deconvolve_and_sum_pix

        indexability.append(can_index)

        #shoebox = refls['shoebox'][i_spot]

    return indexability


def get_spot_data(img, thresh=0, filter=None, **kwargs):
    """
    TODO: make a elx.refltable return option
    Kwargs are passed to the filter used to smear the spots
    :param img: numpy image, assumed to be simulated
    :param thresh: minimum value, this should be  >= the minimum intensity separating spots..
    :param filter: a filter to apply to the data, typically one of scipy.ndimage
        the kwargs will be passed along to this filter
    :return: useful spot dictionary, numpy version of a reflection table..
    """

    if filter is not None:
        labimg, nlab = ndimage.label( filter(img, **kwargs) > thresh)
    else:
        labimg, nlab = ndimage.label( img > thresh)

    if nlab == 0:
        return None

    bboxes = ndimage.find_objects(labimg)

    comIpos = ndimage.center_of_mass(img, labimg, range(1, nlab+1))
    maxIpos = ndimage.maximum_position(img, labimg, range(1, nlab+1))
    maxI = ndimage.maximum(img, labimg, range(1, nlab+1))
    meanI = ndimage.mean(img, labimg, range(1,nlab+1))
    varI = ndimage.variance(img, labimg, range(1,nlab+1))

    return {'comIpos': comIpos,
            'bboxes': bboxes,
            'maxIpos': maxIpos,
            'maxI': maxI,
            'meanI': meanI,
            'varI': varI}



def plot_overlap(spotdataA, spotdataB, refls, is_multi=True, detector=None):
    """

    :param spotdataA: return value of get_spot_data method
    :param spotdataB:
    :param refls:
    :return:
    """
    import pylab as plt

    # compute a simple overlap to within 4 pixels, not very important, just quick debugging
    if is_multi:
        xA,yA,zA = map( np.array, zip(*spotdat_xyz_in_lab(spotdataA, detector )))
        xB,yB,zB = map( np.array, zip(*spotdat_xyz_in_lab(spotdataB, detector )))
        allIA, allIB =[],[]
        #all_yA, all_xA ,all_yB, all_xB, allIA, allIB = [],[],[],[],[],[]
        for i in range(64):
            if i in spotdataA:
                if spotdataA[i] is not None:
                    #yA,xA = zip(*spotdataA[i]["comIpos"])
                    #all_yA.append(yA)
                    #all_xA.append(xA)
                    allIA.append(spotdataA[i]['meanI'])
            if i in spotdataB:
                if spotdataB[i] is not None:
                    #yB,xB = zip(*spotdataB[i]["comIpos"])
                    #all_yB.append(yB)
                    #all_xB.append(xB)
                    allIB.append(spotdataB[i]['meanI'])
        #xA = np.hstack(all_xA)
        #xB = np.hstack(all_xB)
        #yA = np.hstack(all_yA)
        #yB = np.hstack(all_yB)
        allIB = np.hstack( allIB)
        allIA = np.hstack(allIA)
    else:
        yA, xA = map(np.array, zip(*spotdataA["comIpos"]))
        yB, xB = map(np.array, zip(*spotdataB["comIpos"]))


    xAB = np.hstack((xA, xB))
    yAB = np.hstack((yA, yB))
    tree = cKDTree(zip(xAB, yAB))
    if is_multi:
        xdata,ydata,_ = map( np.array, zip(*refls_xyz_in_lab(refls, detector)))
    else:
        xdata, ydata, _ = map(np.array, xyz_from_refl(refls))
    dist, pos = tree.query(zip(xdata, ydata))
    missedx, missedy = xdata[dist >= 4], ydata[dist >= 4]

    n_idx = sum(dist < 4)
    n_refl = len(dist)

    from IPython import embed
    embed()

    s = 4
    r = 5
    plt.figure()
    ax = plt.gca()
    Square_style = {"ec": "C2", "fc": "none", "lw": "1", "width": s + 4, "height": s + 4}
    Square_styleMissed = {"ec": "Deeppink", "fc": "Deeppink", "lw": "1", "width": s + 4, "height": s + 4}

    Circle_styleA = {"ec": "C1", "fc": "C1", "lw": "1", "radius": r}
    Circle_styleB = {"ec": "C3", "fc": "C3", "lw": "1", "radius": r}

    # size the stots according to mean intensity
    if is_multi:
        sizesA = np.log10(allIA)
        sizesB = np.log10(allIB)
    else:
        sizesA = np.log10(spotdataA["meanI"])
        sizesB = np.log10(spotdataB["meanI"])
    sizesA -= sizesA.min()
    sizesB -= sizesB.min()

    add_xy_to_ax_as_patches(zip(xA, yA), plt.Circle, Circle_styleA, ax, sizes=sizesA)
    add_xy_to_ax_as_patches(zip(xB, yB), plt.Circle, Circle_styleB, ax, sizes=sizesB)

    if missedx.size:
        add_xy_to_ax_as_patches(zip(missedx - s / 2., missedy - s / 2.), plt.Rectangle, Square_styleMissed,
                                           ax)
    add_xy_to_ax_as_patches(zip(xdata - s / 2., ydata - s / 2.), plt.Rectangle, Square_style, ax)

    # spot_utils.add_xy_to_ax_as_patches(zip(xA - s / 2., yA - s / 2.), plt.Rectangle, Square_styleA, ax)
    # spot_utils.add_xy_to_ax_as_patches(zip(xB - s / 2., yB - s / 2.), plt.Rectangle, Square_styleB, ax)
    ax.set_ylim(0, 1800)
    ax.set_xlim(0, 1800)
    ax.set_aspect('equal')

    title = " %d / %d ( %.2f %%) reflections were indexed within 4 pixels" % \
            (n_idx, n_refl, 100. * n_idx / n_refl)
    ax.set_title(title)

    plt.show()


def count_roi_overlap( rois, img_size):
    """
    Normalization term which tracks overlap in ROI lists
    :param rois: list of tuples of the form ((x1,x2), (y1,y2))
    specifying regions of interest for simtbx
    :param img_size: size of whole image wherein ROI is referencing
    :return: number of times a pixel in the img of size img_size is
    referenced, this can be used to normalize simulated images
    with overlappig ROIs
    """
    counts = np.zeros( img_size)
    for (x1,x2), (y1,y2) in rois:
        counts[y1:y2, x1:x2] += 1
    return counts


def get_spot_roi( refl, dxtbx_image_size, szx=10, szy=10):
    """
    get the regions of interest around each reflection
    :param refl: reflection table
    :param szx: region of interest dimension by 2
    :param szy: region of interest dimension by 2
    :return: list of tuples of the form ((x1,x2), (y1,y1))
    specifying the ROIs to be used with simtbx
    """
    x,y,_ = xyz_from_refl(refl)
    rois = []
    for i,j in zip(x,y):
        i1 = max(int(i)-szx,0)
        i2 = min(int(i)+szx, dxtbx_image_size[0])

        j1 = max( int(j)-szy,0)
        j2 = min( int(j)+szy, dxtbx_image_size[1])

        rois.append( ((i1,i2),(j1,j2)) )

    return rois


def count_spots(pickle_fname):
    """
    Count the number of spots in a pickle file per shot index
    :param pickle_fname: path to a strong.pickle
    :return: tupe of two arrays: (shot index, number of spots per that index)
    """
    refl = pickle.load(open(pickle_fname,"r"))
    Nrefl = len(refl)
    # because these are stills,
    # the z1 coordinate of the bounding box
    # specifies frame index
    # and z2 should always be z1+1
    x1,x2,y1,y2,z1,z2 = map( np.array, zip(* [refl['bbox'][i] for i in range(Nrefl)]))
    assert( np.all(z2==z1+1))
    shot_idx, Nspots_per_shot = map( np.array, zip(*Counter(z1).items()))
    return shot_idx, Nspots_per_shot


def group_refl_by_shotID(refl):
    """
    groups shots according to shot index
    :param refl: flex reflection table
    :return: grouped objetcs dictionary, keys are the shot index, values are the reflections
    """
    get_shot_idx = lambda x: x['bbox'][4]
    grouped = groupby( sorted(refl,key=get_shot_idx), get_shot_idx)
    return {shot_idx:list(v) for shot_idx,v in grouped}

class ReflectionSelect:
    def __init__(self, refl_tbl):
        """
        caches the shot index of reflections
        :param refl_tbl:
        """
        self.refl = refl_tbl
        self.Nrefl = len( refl_tbl)
        self.all_shot_idx = flex.int( [ refl_tbl['bbox'][i][4] for i in range(self.Nrefl)])

    def select(self, shot_idx):
        shot_refl = self.refl.select( self.all_shot_idx==shot_idx )

        bbox = shot_refl['bbox']
        sb = shot_refl["shoebox"]
        for i in range(len(shot_refl)):
            bbox[i] = (bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], 0, 1)
            sb_bbox = sb[i].bbox
            sb[i].bbox = (sb_bbox[0], sb_bbox[1], sb_bbox[2], sb_bbox[3], 0, 1)
        return shot_refl


def as_single_shot_reflections(refl_, inplace=True):
    """
    sets all z-coors to 0,1
    :param refl_: reflection table , should be just single image
    :return: updated table
    """
    if not inplace:
        refl = deepcopy( refl_)
    else:
        refl = refl_
    bbox = refl['bbox']
    sb = refl["shoebox"]
    for i in range(len(refl)):
        bbox[i] = (bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], 0, 1)
        sb_bbox = sb[i].bbox
        sb[i].bbox = (sb_bbox[0], sb_bbox[1], sb_bbox[2], sb_bbox[3], 0, 1)
    if not inplace:
        return refl


def refls_by_panelname(refls):
    Nrefl = len( refls)
    panel_names = np.array([refls["panel"][i] for i in range( Nrefl)])
    uniq_names = np.unique(panel_names)
    refls_by_panel = {name: refls.select(flex.bool(panel_names == name))
                      for name in uniq_names }
    return refls_by_panel


def select_refl(refl, shot_idx):
    """
    selects reflections belinging to a particular shot index
    :param refl: reflection table
    :param shot_idx: int, shot index
    :return:
    """
    n = len( refl)
    select_me = flex.bool([refl['bbox'][i][4] == int(shot_idx) for i in range(n)])
    return refl.select(select_me)



def get_spot_data_multipanel(data, detector, beam, crystal,
                             pids=None, thresh=0, filter=None,  **kwargs):
    """
    TODO: flex.refltable return option

    This is a function for taking images (typically simulated) and
    grabbing the spots

    :param data:
    :param detector:
    :param beam:
    :param crystal:
    :param pids:
    :param thresh:
    :param filter:
    :param kwargs:
    :return:
    """

    if pids == None:
        pids = range (len(data))
    else:
        assert( len(pids) == len(data))

    spot_data = {}
    all_q = []
    all_h = []
    all_hi = []
    for pid in pids:
        img = data[pid]
        spot_data[pid] = get_spot_data(
            img,
            thresh=thresh,
            filter=filter,
            **kwargs)
        if spot_data[pid] is None:
            continue

        y,x = zip(*spot_data[pid]['comIpos'])
        detnode = detector[pid]
        q_vecs = xy_to_q(x,y,detnode, beam)
        spot_data[pid]["q_vecs"] = q_vecs

        #h, hi = refls_to_hkl((x,y), detnode, beam, crystal)

        #spot_data[pid]['H'] = h
        #spot_data[pid]['Hi'] = hi

        #all_h.append(h)
        #all_hi.append(hi)
        all_q.append(q_vecs)
    if all_q:
        spot_data["Q"] = np.vstack(all_q)
    #if all_h:
        #spot_data["H"] = np.vstack( all_h)
        #spot_data["Hi"] = np.vstack( all_hi)

    return spot_data


def msisrp_overlap(spot_data, refls, detector, xy_only=False):
    """
    Given a spot_data object and a reflection table
    Find the proximity of each reflection to each spo

    Spot_data is basically a reflection table
    made from the simulated data, using scipy.ndimage

    :param spot_data: spot data object
    :param refls: reflection tabel
    :param detector: dxtbx detector model
    :param xy_only: whether to return distances in the plane
        perpendicular to the z coordinate, or full 3D
    :return: the dists and the residual vectors pointing from the
        prediction to the strong spot
    """

    if xy_only:
        end=2
    else:
        end=3

    xyz_spot = spotdat_xyz_in_lab(spot_data, detector)
    tree = cKDTree(xyz_spot[:,:end])

    xyz_refls = refls_xyz_in_lab( refls, detector)

    dists, nn = tree.query(xyz_refls[:,:end])

    dist_vecs = xyz_refls[:,:end] - tree.data[nn]

    return dists, dist_vecs


if __name__=="__main__":
    plot = True
    fname = sys.argv[1]  # path to a strong.pickle
    cutoff = int(sys.argv[2])  # min number of spots  per hit
    shot_idx, Nspots_per_shot = count_spots(fname)
    print ("there are {:d} / {:d} shots with >= {:d} spots"
           .format((Nspots_per_shot > cutoff).sum(), len(shot_idx), cutoff))

    if plot:
        import pylab as plt
        plt.figure()
        bins = np.logspace(0, 3, 50)
        ax = plt.gca()
        plt.hist(Nspots_per_shot, bins=bins, )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.tick_params(labelsize=14)
        ax.grid(1, which='both')
        plt.xlabel("Spots per shot", fontsize=14)
        plt.ylabel("bincount", fontsize=14)
        plt.show()