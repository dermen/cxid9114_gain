import os
import sys
import cPickle
import h5py
import numpy as np

import dxtbx
from cxid9114.spots import count_spots

MIN_SPOT_PER_HIT = 20
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

# ============
output_h5_name = os.path.join( output_dir, "run%d_hits.h5" % loader.run_number)
with h5py.File( output_h5_name, "w") as out_h5:

    out_h5.create_dataset("panel_masks", data=loader.cspad_mask, dtype=np.bool)
    out_h5.create_dataset("panel_gainmasks", data=loader.gain)
    x, y = loader.cspad.coords_xy(loader._get_event(where_hits[0]))
    out_h5.create_dataset("panel_x", data=x/109.92)
    out_h5.create_dataset("panel_y", data=y/109.92)
    out_h5.create_dataset("panel_z", data=np.ones_like(x)*loader.detector_distance)
    out_h5.create_dataset("pedestal", data=loader.dark)
    out_h5.create_dataset("gain_val", data=loader.nominal_gain_val)

    panel_dset = out_h5.create_dataset("panels",
                                       dtype=np.int16,
                                       shape=(Nhits, 32, 185, 388))
    for i_hit in range(Nhits):
        print '\rSaving hit {:d}/{:d}'.format(i_hit+1, Nhits),
        sys.stdout.flush()
        shot_idx = idx[where_hits[i_hit]]
        panel_dset[i_hit] = loader.get_psana_raw(shot_idx)
