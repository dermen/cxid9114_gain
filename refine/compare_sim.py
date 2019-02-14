from cxid9114 import utils
import glob
from scitbx.matrix import sqr
from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color

fnames = glob.glob( "results/good_hit_*_10_crys.pkl")
for fname in fnames:
    data = utils.open_flex(fname)

    crystal = data['crystal']
    xrot = data['optX']
    yrot = data['optY']
    new_A = xrot * yrot * sqr(crystal.get_U()) * sqr(crystal.get_B())
    crystal.set_A(new_A)

    orient = indexer_two_color(
        reflections=refl, imagesets=[hit_imgset], params=params)

    # h5_name = fname.replace(".pkl", ".h5")
