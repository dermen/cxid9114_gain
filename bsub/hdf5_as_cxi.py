import os
import sys
import cPickle
import h5py
import numpy as np
from cxid9114 import utils
import dxtbx
from cxid9114.spots import count_spots

MIN_SPOT_PER_HIT = 30
Nmax_write = 20
output_dir = "."

pickle_fname = sys.argv[1]
image_fname = sys.argv[2]

print('Loading reflections')
with open(pickle_fname, 'r') as f:
    found_refl = cPickle.load(f)
refl_select = count_spots.ReflectionSelect(found_refl)

print('Loading format')
loader = dxtbx.load(image_fname)
imgset = loader.get_imageset(loader.get_image_file())

print('Counting spots')
idx, Nspot_at_idx = count_spots.count_spots(pickle_fname)
where_hits = np.where(Nspot_at_idx > MIN_SPOT_PER_HIT)[0]
Nhits = where_hits.shape[0]

mask2d = loader.assemble( loader.mask).astype(int).astype(bool)
dummie_image = loader.get_raw_data(0).as_numpy_array()
img_sh = dummie_image.shape

# ============
if len(sys.argv) == 3:
    output_h5_name = os.path.join(output_dir, "hits.cxi")
else:
    output_h5_name = sys.argv[3]


with h5py.File( output_h5_name, "w") as out_h5:

    out_h5.create_dataset("mask", data=mask2d, dtype=np.bool)
    panel_dset = out_h5.create_dataset("data",
                                       dtype=np.float32,
                                       shape=(Nhits, img_sh[0], img_sh[1]) )
    all_spotX, all_spotY, all_spotI = [], [], []
    for i_hit in range(Nhits):
        if i_hit==Nmax_write:
            break
        print '\rSaving hit {:d}/{:d}'.format(i_hit+1, Nhits),
        sys.stdout.flush()
        shot_idx = idx[where_hits[i_hit]]
        refls = refl_select.select(shot_idx)
        X, Y, _ = zip(*[refls["xyzobs.px.value"][i] for i in range(len(refls))])
        I = [refls['intensity.sum.value'][i] for i in range(len(refls))]
        all_spotX.append(X)
        all_spotY.append(Y)
        all_spotI.append(I)
        panel_dset[i_hit] = loader.get_raw_data(shot_idx).as_numpy_array().astype(np.float32)

    utils.write_cxi_peaks( out_h5, "peaks", all_spotX, all_spotY, all_spotI)

