from dials.array_family import flex
from copy import deepcopy
import numpy as np

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


def add_xy_to_ax_as_patches(xy, patch_type, patch_sty, ax):
    import matplotlib as mpl
    patches = [patch_type(xy=(i_, j_), **patch_sty)
             for i_, j_ in xy]

    #squares = [Rectangle(xy=(i_ - s / 2., j_ - s / 2.), **Square_style) for i_, j_ in zip(xp, yp)]
    #coll_cir = mpl.collections.PatchCollection(circs, match_original=True)
    patch_coll = mpl.collections.PatchCollection(patches, match_original=True)
    ax.add_collection(patch_coll)

#r = 6 # circle radius
#Circ_style = {"ec":"C0", "fc":"C0", "lw":"1", "radius":r}
## predictions will be squares
#s = 0.4*sqrt(((2*r)**2)/2) # square edge
#Square_style = {"ec":"C1", "fc":"C1", "lw":"1", "width":s, "height":s}