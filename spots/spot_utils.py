from dials.array_family import flex
from copy import deepcopy
import numpy as np
from scipy import ndimage
from scitbx.matrix import sqr
from scipy.spatial import cKDTree


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
