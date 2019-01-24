

import sys
from collections import Counter
from itertools import groupby
import cPickle as pickle
from copy import deepcopy
import numpy as np
from scipy import ndimage

from scitbx.matrix import sqr
from scipy.spatial import cKDTree
from dials.array_family import flex


def xy_to_hkl(x,y, detector, beam, crystal, as_numpy_arrays=True):
    """
    convert pixel xy to miller index data

    :param x: fast scan coord of spot, array-like
    :param y:
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param as_numpy_arrays: return data as numpy arrays
    :return: if as_numpy twp Nx3 numpy arrays are returned
        (one for fractional and one for whole HKL)
        else dictionary of hkl_i (nearest) and hkl (fractional)
    """
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()

    q_vecs = xy_to_q( x,y, detector, beam)

    HKL = np.dot( Ai, q_vecs.T)
    HKLi = map( lambda h: np.ceil(h-0.5), HKL)
    if as_numpy_arrays:
        return np.vstack(HKL).T, np.vstack(HKLi).T
    else:
        return {'hkl':HKL, 'hkl_i': HKLi}

def xy_to_q(x,y, detector, beam, oldmethod=False, panel_id=0):
    """
    convert pixel coords to q-vectors

    :param x,y:  pixel coordinates of spots as separate lists/arrays
                x should be the fast-scan coord
    :param detector: dxtbx detector model
    :param beam:  dxtbx beam model
    :param method1: whether to use method 1 below..
    :return: the Q-vectors corresponding to the spots
    """
    panel = detector[panel_id]
    if oldmethod:
        pixsizeFS, pixsizeSS = panel.get_pixel_size()[0]
        orig = np.array(panel.get_origin())
        ss = np.array(panel.get_slow_axis())
        fs = np.array(panel.get_fast_axis())
        s0 = np.array(beam.get_s0())  # this is already scaled by 1/wavelength
        pix_mm = np.vstack( [orig + fs*i*pixsizeFS + ss*j*pixsizeSS
                for i,j in zip(x,y)])
        s1 = pix_mm / np.linalg.norm( pix_mm,axis=1)[:,None] / beam.get_wavelength()
        q_vecs = (s1 - s0)
    else:
        pix_mm = panel.pixel_to_millimeter( flex.vec2_double( zip(x,y)))
        coords = panel.get_lab_coord( pix_mm)
        coords = coords / coords.norms()
        q_vecs = coords *(1./beam.get_wavelength()) - beam.get_s0()

        # I like to return as numpy array...
        q_vecs = q_vecs.as_double().as_numpy_array().reshape((len(q_vecs), 3))

    return q_vecs


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


def make_color_data_object(x, y, beam, crystal, detector):
    """
    make the values expected in the `color_data` dictionary parameter passed to
    the `compute_indexability` method.
    :param x:
    :param y:
    :param beam:
    :param crystal:
    :param detector:
    :return:
    """
    spots = zip(x,y)
    hkl, hkli = xy_to_hkl(x, y, detector, beam, crystal)
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
        data_hkl, data_hkli = xy_to_hkl(xdata, ydata,
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


def get_spot_data(img, thresh=0):
    """
    TODO: implement this using flex
    :param img: numpy image, assumed to be simulated
    :param thresh: minimum value, this should be  >= the minimum intensity separating spots..
    :return: useful spot dictionary, numpy version of a reflection table..
    """
    labimg, nlab = ndimage.label(img > thresh)
    bboxes = ndimage.find_objects( labimg)

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


def plot_overlap(spotdataA, spotdataB, refls):
    """

    :param spotdataA:
    :param spotdataB:
    :param refls:
    :return:
    """
    import pylab as plt

    # compute a simple overlap to within 4 pixels, not very important, just quick debugging
    yA, xA = map(np.array, zip(*spotdataA["comIpos"]))
    yB, xB = map(np.array, zip(*spotdataB["comIpos"]))
    xAB = np.hstack((xA, xB))
    yAB = np.hstack((yA, yB))
    tree = cKDTree(zip(xAB, yAB))
    xdata, ydata, _ = map(np.array, xyz_from_refl(refls))
    dist, pos = tree.query(zip(xdata, ydata))
    missedx, missedy = xdata[dist >= 4], ydata[dist >= 4]

    n_idx = sum(dist < 4)
    n_refl = len(dist)

    s = 4
    r = 5
    plt.figure()
    ax = plt.gca()
    Square_style = {"ec": "C2", "fc": "none", "lw": "1", "width": s + 4, "height": s + 4}
    Square_styleMissed = {"ec": "Deeppink", "fc": "Deeppink", "lw": "1", "width": s + 4, "height": s + 4}

    Circle_styleA = {"ec": "C1", "fc": "C1", "lw": "1", "radius": r}
    Circle_styleB = {"ec": "C3", "fc": "C3", "lw": "1", "radius": r}

    # size the stots according to mean intensity
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


def get_spot_roi( refl, szx=10, szy=10):
    """
    get the regions of interest around each reflection
    :param refl: reflection table
    :param szx: region of interest dimension by 2
    :param szy: region of interest dimension by 2
    :return: list of tuples of the form ((x1,x2), (y1,y1))
    specifying the ROIs to be used with simtbx
    """
    x,y,_ = xyz_from_refl(refl)
    rois = [((int(i) - szx, int(i) + szx), (int(j) - szy, int(j) + szy))
            for i, j in zip(x, y)]
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