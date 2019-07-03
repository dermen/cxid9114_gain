from cctbx import crystal, miller, sgtbx
from cctbx.array_family import flex
import numpy as np
from itertools import izip


def get_val_at_hkl(hkl, val_map, sg=sgtbx.space_group(" P 4nw 2abw")):
    """
    given a miller index, find its value in the miller dictionary array (val_map)

    :param hkl:  tuple
    :param val_map:  miller data dictionary
    :param sg: space group
    :return: miller value
    """
    poss_equivs = [i.h() for i in miller.sym_equiv_indices(sg, hkl).indices()]

    in_map = False
    for hkl2 in poss_equivs:
        if hkl2 in val_map:  # fast lookup
            in_map = True
            break
    if in_map:
        return hkl2, val_map[hkl2]
    else:
        return (None, None, None), None


def generate_table(complex_data, indices, numpy_args=False, anom=True):
    """

    :param complex_data: structure factors
    :param indices: miller indices in a list, Nx3
    :param numpy_args: are the complex data and indices numpy type, if not
        assume flex
    :param anom: return a miller array with +H and -H ?
    :return: dictionary whose keys are miller index tuple and values are structure fact
    """
    sg = sgtbx.space_group(" P 4nw 2abw")
    Symm = crystal.symmetry(unit_cell=(79, 79, 38, 90, 90, 90), space_group=sg)
    if numpy_args:
        assert type(indices) == tuple
        assert (type(indices[0]) == tuple)
        indices = flex.miller_index(indices)

    mil_set = miller.set(crystal_symmetry=Symm, indices=indices, anomalous_flag=anom)
    if numpy_args:
        complex_data = flex.complex_double(np.ascontiguousarray(complex_data))
    mil_ar = miller.array(mil_set, data=complex_data)

    mil_dict = {h: val for h, val in izip(mil_ar.indices(), mil_ar.data())}
    return mil_dict