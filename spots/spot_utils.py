from dials.array_family import flex
from copy import deepcopy

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
