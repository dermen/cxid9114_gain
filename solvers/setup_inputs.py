from dials.array_family import flex
from cctbx import miller
from cctbx import crystal
from cctbx import sgtbx


def single_to_asu(h, ano=True):
    sg = sgtbx.space_group(" P 4nw 2abw")  # SG 96!
    sym = crystal.symmetry(unit_cell=(79, 79, 38, 90, 90, 90),
                           space_group=sg)
    idx = flex.miller_index((h,))
    mill_set = miller.set(crystal_symmetry=sym, indices=idx,
                        anomalous_flag=ano)
    mill_asu = mill_set.map_to_asu().indices()
    return mill_asu[0]


def many_to_asu_and_whether_positive(h, return_aso_only=True):
    """h is a tuple of 3-tuples"""
    sg = sgtbx.space_group(" P 4nw 2abw")  # SG 96!
    sym = crystal.symmetry(unit_cell=(79, 79, 38, 90, 90, 90),
                           space_group=sg)
    idx = flex.miller_index(h)
    mill_set_ano = miller.set(crystal_symmetry=sym, indices=idx, anomalous_flag=True)
    mill_set = miller.set(crystal_symmetry=sym, indices=idx, anomalous_flag=False)

    mill_asu_ano = mill_set_ano.map_to_asu()
    mill_asu = mill_set.map_to_asu()

    is_positive = mill_asu_ano.indices() == mill_asu.indices()

    if return_aso_only:
        return mill_asu_ano, is_positive
    else:
        return mill_asu_ano, mill_asu, is_positive


def query_df(df, FA, FB):
    gb = df.groupby(['h', 'k', 'l'])
    hkl_groups = gb.groups.keys()

    sigA = []
    sigB = []
    for i_hkl, hkl in enumerate(hkl_groups):
        sub_d = gb.get_group(hkl)
        neg_idx = sub_d.query("~is_pos").index.values
        pos_idx = sub_d.query("is_pos").index.values
        if not neg_idx.size or not pos_idx.size:
            continue
        FA_pos = FA[pos_idx].mean()
        FA_neg = FA[neg_idx].mean()

        FB_pos = FB[pos_idx].mean()
        FB_neg = FB[neg_idx].mean()

        print i_hkl
        sigA.append(abs(FA_pos - FA_neg))
        sigB.append(abs(FB_pos - FB_neg))
    return sigA, sigB
