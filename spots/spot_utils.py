
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
from dials.algorithms.shoebox import MaskCode


def q_to_hkl(q_vecs, crystal):
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot( Ai, q_vecs.T)
    HKLi = map( lambda h: np.ceil(h-0.5).astype(int), HKL)
    return np.vstack(HKL).T, np.vstack(HKLi).T


def refls_to_pixelmm(refls, detector):
    """
    returns the mm position of the spot in pixels
    referenced to the panel origin, which I think is the
    center of the first pixel in memory for the panel
    :param reflection table
    :param dxtbx detecotr
    :return: np.array Nreflsx2 for fast,slow mm coord referenced to panel origin
        in the plane of the panel
    """
    ij_mm = np.zeros( (len(refls),2))
    for i_r,r in enumerate(refls):
        panel = detector[r['panel']]
        i,j,_ = r['xyzobs.px.value']
        ij_mm[i_r] = panel.pixel_to_millimeter( (i,j) )
    return ij_mm


def refls_to_hkl(refls, detector, beam, crystal,
                 update_table=False, returnQ=False):
    """
    convert pixel panel reflections to miller index data

    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param update_table: whether to update the refltable
    :param returnQ: whether to return intermediately computed q vectors
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
        refls['miller_index'] = flex.miller_index(len(refls),(0,0,0))
        mil_idx = flex.vec3_int(tuple(map(tuple, np.vstack(HKLi).T)))
        for i in range(len(refls)):
            refls['miller_index'][i] = mil_idx[i]
    if returnQ:
        return np.vstack(HKL).T, np.vstack(HKLi).T, q_vecs
    else:
        return np.vstack(HKL).T, np.vstack(HKLi).T


def refls_to_q(refls, detector, beam, update_table=False):

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
        i_fs, i_ss, _ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append(s1)
        q_vecs.append(s1-beam.get_s0())

    if update_table:
        refls['s1'] = flex.vec3_double(tuple(map(tuple,s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple,q_vecs)))

    return np.vstack(q_vecs)

def refls_xyz_in_lab(refls, detector, xy_key="xyzobs.px.value"):
    all_refls_lab = []
    for i,r in enumerate(refls):
        panel_name = r['panel']
        panel = detector[panel_name]
        x,y,_ = r[xy_key]
        lab_xyz = panel.get_pixel_lab_coord((x,y))
        all_refls_lab.append( lab_xyz)
    return np.array(all_refls_lab)


def fs_ss_to_q(fs, ss, pids, detector, beam):

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set(pids)

    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    for i_fs, i_ss, pid in zip(fs,ss, pids):
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append( s1)
        q_vecs.append( s1-beam.get_s0())

    return np.vstack(q_vecs)


def npix_per_spot(refl_tbl):
    """counts the number of pixels per reflection"""
    Nrefl = len( refl_tbl)
    
    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
              for i in range(Nrefl)]
    
    code = MaskCode.Foreground.real

    all_npix = np.zeros( Nrefl, int)
    for i_ref, M in enumerate( masks):
        all_npix[i_ref] = np.sum(M & code == code)
    return all_npix 


def get_single_refl_spot_mask(refl, img_size):
    """assume refl contains shoebox, img_size is (Nslow-scan , Nfast-scan) format"""
    from dials.algorithms.shoebox import MaskCode
    mask = refl['shoebox'].mask.as_numpy_array()

    code = MaskCode.Foreground.real

    x1, x2, y1, y2, z1, z2 = refl['shoebox'].bbox

    spot_mask = np.zeros(img_size, bool)

    slcX = slice(x1, x2, 1)
    slcY = slice(y1, y2, 1)
    spot_mask[slcY, slcX] = mask & code == code
    return spot_mask


def strong_spot_mask(refl_tbl, img_size, as_composite=True):
    from dials.algorithms.shoebox import MaskCode
    Nrefl = len( refl_tbl)
    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
              for i in range(Nrefl)]
    code = MaskCode.Foreground.real

    x1, x2, y1, y2, z1, z2 = zip(*[refl_tbl[i]['shoebox'].bbox
                                   for i in range(Nrefl)])
    if not as_composite:
        spot_masks = []
    spot_mask = np.zeros(img_size, bool)
    for i1, i2, j1, j2, M in zip(x1, x2, y1, y2, masks):
        slcX = slice(i1, i2, 1)
        slcY = slice(j1, j2, 1)
        spot_mask[slcY, slcX] = M & code == code
        if not as_composite:
            spot_masks.append(spot_mask.copy())
            spot_mask *= False
    if as_composite:
        return spot_mask
    else:
        return spot_masks

#def strong_spot_mask(refl_tbl, img_size):
#    Nrefl = len( refl_tbl)
#    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
#              for i in range(Nrefl)]
#    x1, x2, y1, y2, z1, z2 = zip(*[refl_tbl[i]['shoebox'].bbox
#                                   for i in range(Nrefl)])
#    spot_mask = np.zeros(img_size, bool)
#    for i1, i2, j1, j2, M in zip(x1, x2, y1, y2, masks):
#        slcX = slice(i1, i2, 1)
#        slcY = slice(j1, j2, 1)
#        spot_mask[slcY, slcX] = M == 5
#    return spot_mask


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
    """returns the xyz of the pixels by default in weird (xpix, ypix, zmm) format"""
    x,y,z = zip( * [refl[key][i] for i in range(len(refl))])
    return x,y,z





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


def add_xy_to_ax_as_patches(xy, patch_type, patch_sty, ax, sizes=None, alpha=1.):
    import matplotlib as mpl
    patches = [patch_type(xy=(i_, j_), alpha=alpha, **patch_sty)
             for i_, j_ in xy]
    if sizes is not None:
        if not patch_type == mpl.patches.Circle:
            print("Cannot update size for non-circle patches")
        else:
            for i in range(len(patches)):
                patches[i].radius = sizes[i]

    patch_coll = mpl.collections.PatchCollection(patches, match_original=True)
    ax.add_collection(patch_coll)


def plot_overlap(refls_simA, refls_simB, refls_data, detector, alpha=.75,
                 square_s=4, cutoff=4, circ_r=5, scale_sizes=False):
    """
    # so this should change
    refls tables from simulations A and B as well as the data
    and a dxtbx detector model
    """
    import pylab as plt
    pixsize = detector[0].get_pixel_size()[0]

    xA,yA,_ = refls_xyz_in_lab(refls_simA, detector).T
    xB,yB,_ = refls_xyz_in_lab(refls_simB, detector).T

    allIA = np.array( [r['intensity.sum.value'] for r in refls_simA])
    allIB = np.array( [r['intensity.sum.value'] for r in refls_simB])

    xAB = np.hstack((xA, xB))
    yAB = np.hstack((yA, yB))
    tree = cKDTree(zip(xAB, yAB))


    xdata,ydata,_ = refls_xyz_in_lab(refls_data, detector).T
    dist, pos = tree.query(zip(xdata, ydata))
    missedx, missedy = xdata[dist >= cutoff*pixsize], ydata[dist >= cutoff*pixsize]

    n_idx = sum(dist < cutoff*pixsize)
    n_refl = len(dist)

    s = square_s * pixsize  # size of square in pixels
    r = circ_r * pixsize # size of circle in pixels
    plt.figure()
    ax = plt.gca()
    Square_style = {"ec": "C2", "fc": "none", "lw": "1", "width": s, "height": s}
    Square_styleMissed = {"ec": "Deeppink", "fc": "Deeppink", "lw": "1", "width": s, "height": s}

    Circle_styleA = {"ec": "C1", "fc": "C1", "lw": "1", "radius": r}
    Circle_styleB = {"ec": "C3", "fc": "C3", "lw": "1", "radius": r}

    # size the stots according to mean intensity
    sizesA = sizesB = None
    if scale_sizes:
        sizesA = np.log10(allIA)
        sizesB = np.log10(allIB)
        sizesA -= sizesA.min()
        sizesB -= sizesB.min()

    add_xy_to_ax_as_patches(zip(xA, yA), plt.Circle, Circle_styleA, ax, alpha=alpha, sizes=sizesA)
    add_xy_to_ax_as_patches(zip(xB, yB), plt.Circle, Circle_styleB, ax, alpha=alpha, sizes=sizesB)

    if missedx.size:
        add_xy_to_ax_as_patches(zip(missedx - s/2., missedy - s/2.), plt.Rectangle, Square_styleMissed,
                                           ax, alpha=alpha)
    add_xy_to_ax_as_patches(zip(xdata - s/2., ydata - s/2.), plt.Rectangle, Square_style, ax, alpha=alpha)

    # spot_utils.add_xy_to_ax_as_patches(zip(xA - s / 2., yA - s / 2.), plt.Rectangle, Square_styleA, ax)
    # spot_utils.add_xy_to_ax_as_patches(zip(xB - s / 2., yB - s / 2.), plt.Rectangle, Square_styleB, ax)
    ax.set_ylim(yAB.min()-1, yAB.max()+1)
    ax.set_xlim(xAB.min()-1, xAB.max()+1)
    ax.set_aspect('equal')

    title = " %d / %d ( %.2f %%) reflections were indexed within %d pixels" % \
            (n_idx, n_refl, 100. * n_idx / n_refl, cutoff)
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


def get_fs_rois( f,s, shape_fs, szx=10, szy=10):
    """
    get the regions of interest around each reflection
    :param f: fast scan coords
    :param s: slow scan coords
    :param shape_fs, image shape in fast-scan, slow-scan
    :param szx: region of interest dimension by 2
    :param szy: region of interest dimension by 2
    :return: list of tuples of the form ((x1,x2), (y1,y1))
    specifying the ROIs to be used with simtbx
    """
    rois = []
    for i, j in zip(f, s):
        i1 = max(int(i) - szx, 0)
        i2 = min(int(i) + szx, shape_fs[0])

        j1 = max(int(j) - szy, 0)
        j2 = min(int(j) + szy, shape_fs[1])

        rois.append(((i1, i2), (j1, j2)))

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


def msisrp_overlap(refls_sim, refls_data, detector, xy_only=False):
    """
    Given a spot_data object and a reflection table
    Find the proximity of each reflection to each spo

    Spot_data is basically a reflection table
    made from the simulated data, using scipy.ndimage

    :param refls_sim: reflection table for simulated data  spots
    :param refls: reflection table for strong spots in the data
    :param detector: dxtbx detector model
    :param xy_only: whether to return distances in the plane
        perpendicular to the z coordinate, or full 3D
    :return: the dists and the residual vectors pointing from the
        prediction to the strong spot
    """

    if xy_only:
        end=2  # just compare x and y
    else:
        end=3  # compare distances in 3D xyz

    xyz_sim = refls_xyz_in_lab(refls_sim, detector)
    simtree = cKDTree(xyz_sim[:,:end])

    xyz_data = refls_xyz_in_lab( refls_data, detector)

    dists, nn = simtree.query(xyz_data[:,:end])

    dist_vecs = xyz_data[:,:end] - simtree.data[nn]

    return dists, dist_vecs


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None, **kwargs ):
    """
    This class is for converting the centroids in the noiseless simtbx images
    to a multi panel reflection table

    TODO: bring up poor documentation and consider asking the dials team
    to make a push to beter document for the sake of developers
    This function took 3 hours to figure out how to do...

    :param panel_imgs: list or 3D array of detector panel simulations
        currently supports CSPAD only (194x185 shaped panels)
    :param detector: dxtbx  detector model of a caspad
    :param beam:  dxtxb beam model
    :param thresh: threshol intensity for labeling centroids
    :param filter: optional filter to apply to images before
        labeling threshold, typically one of scipy.ndimage's filters
    :param pids: panel IDS , else assumes panel_imgs is same length as detector
    :param kwargs: kwargs to pass along to the optional filter
    :return: a reflection table of spot centroids
    """
    from dials.algorithms.spot_finding.factory import FilterRunner
    from dials.model.data import PixelListLabeller, PixelList
    from dials.algorithms.spot_finding.finder import PixelListToReflectionTable
    from cxid9114 import utils

    if panel_ids is None:
        panel_ids = np.arange(len(detector))
    pxlst_labs = []
    for i, pid in enumerate(panel_ids):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        img_sz = detector[pid].get_image_size()
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append( plab)

    pixlst_to_reftbl = PixelListToReflectionTable(
        min_spot_size=1,
        max_spot_size=194*184,
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)

    dblock = utils.datablock_from_numpyarrays(panel_imgs, detector, beam)
    iset = dblock.extract_imagesets()[0]
    refls = pixlst_to_reftbl(iset, pxlst_labs)[0]

    return refls

def get_prediction_boxes(refls, detector, beam, crystal,
                    twopi_conv=True, delta_q=0.015, **kwargs):
    """
    this function returns a list of pylab square patches to overlay on on image
    :param refls:
    :param detector:
    :param beam:
    :param crystal:
    :param delta_q: width of reciprocal space box in angstrom
    :return:
    """
    import matplotlib as mpl
    import pylab as plt

    H, Hi, Q = refls_to_hkl(
        refls, detector, beam, crystal,  returnQ=True)

    Q = np.linalg.norm(Q,axis=1)
    if twopi_conv:
        Q*=2*np.pi
    detdist = detector[0].get_distance()
    pixsize = detector[0].get_pixel_size()[0]
    wavelen = beam.get_wavelength()

    rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Q-delta_q*.5)*wavelen/4/np.pi))
    rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Q+delta_q*.5)*wavelen/4/np.pi))
    delrad = rad2-rad1

    patches = []
    x,y,_ = xyz_from_refl(refls)
    x,y = np.array(x), np.array(y)
    x -= (1 + delrad)/2.
    y -= (1 + delrad)/2.

    for i_ref, xy in enumerate(zip(x,y)):
        R = plt.Rectangle(xy=xy,
                          width=delrad[i_ref],
                          height=delrad[i_ref],
                          **kwargs)
        patches.append(R)

    patch_coll = mpl.collections.PatchCollection(patches,
                    match_original=True)
    return patch_coll


def get_white_boxes(refls_at_colors, detector, beams_of_colors, crystal,
                    twopi_conv=True, delta_q=0.015, **kwargs):
    """
    this function returns a list of pylab square patches to overlay on on image
    :param refls:
    :param detector:
    :param beam:
    :param crystal:
    :param delta_q: width of reciprocal space box in angstrom
    :return:
    """
    import matplotlib as mpl
    import pylab as plt

    color_data = {}
    color_data["Q"] = []
    color_data["H"] = []
    color_data["Hi"] = []
    color_data["x"] = []
    color_data["y"] = []
    color_data["Qmag"] = []
    detdist = detector[0].get_distance()
    pixsize = detector[0].get_pixel_size()[0]

    for refls, beam in zip(refls_at_colors, beams_of_colors):

        H, Hi, Q = refls_to_hkl(
            refls, detector, beam, crystal,  returnQ=True)

        color_data["Q"].append(list(Q))
        color_data["H"].append(list(H))
        color_data["Hi"].append(list(map(tuple, Hi)))
        Qmag = np.linalg.norm(Q, axis=1)
        if twopi_conv:
            Qmag*=2*np.pi

        x, y, _ = xyz_from_refl(refls)
        color_data["x"].append(x)
        color_data["y"].append(y)
        color_data["Qmag"].append(Qmag)

    ave_wave = np.mean( [beam.get_wavelength() for beam in beams_of_colors])
    all_indexed_Hi = [tuple(h) for hlist in color_data["Hi"] for h in hlist]
    unique_indexed_Hi = set( all_indexed_Hi)

    all_x, all_y, all_H = [], [], []
    patches = []
    for H in unique_indexed_Hi:
        x_com = 0
        y_com = 0
        Qmag = 0
        n_counts = 0
        for i_color in range(len(beams_of_colors)):
            in_color = H in color_data["Hi"][i_color]
            if not in_color:
                continue

            idx = color_data["Hi"][i_color].index(H)
            x_com += color_data["x"][i_color][idx] - 0.5
            y_com += color_data["y"][i_color][idx] - 0.5
            Qmag += color_data["Qmag"][i_color][idx]
            n_counts += 1
        Qmag = Qmag / n_counts
        all_x.append(x_com / n_counts)
        all_y.append(y_com/ n_counts)

        x_com = x_com / n_counts
        y_com = y_com / n_counts

        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
        delrad = rad2-rad1

        R = plt.Rectangle(xy=(x_com-delrad/2., y_com-delrad/2.),
                          width=delrad,
                          height=delrad,
                          **kwargs)
        patches.append(R)

    patch_coll = mpl.collections.PatchCollection(patches,
                                                 match_original=True)
    return patch_coll


    #from collections import Counter

    #counts =  Counter( all_indexed_Hi)
    #for h,N in counts.items():
    #    print h, N


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
