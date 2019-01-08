from dials.array_family import flex

def combine_refls(refls_per_patt):
    combined_refls = flex.reflection_table()
    for i_pattern, patt_refls in enumerate( refls_per_patt):
        for i, refl in enumerate(patt_refls):
            bb = list(patt_refls['bbox'][i])
            bb[4] = i
            bb[5] = i+1
            patt_refls['bbox'][i] = tuple(bb)
            sb = patt_refls['shoebox'][i]
            sb_bb = list(sb.bbox)
            sb_bb[4] = i
            sb_bb[5] = i+1
            sb.bbox = tuple( sb_bb)
            patt_refls['shoebox'][i] = sb
        combined_refls.extend(patt_refls)
    return combined_refls
