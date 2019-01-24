import os
import sys
import cPickle
import h5py
import numpy as np

import dxtbx
from cxid9114.spots import spot_utils
from cxid9114 import utils

MIN_SPOT_PER_HIT = 20
run = int(sys.argv[3])
output_dir = "./run%d"%run

pickle_fname = sys.argv[1]
image_fname = sys.argv[2]
output_tag = sys.argv[3]

print('Loading reflections')
with open(pickle_fname, 'r') as f:
    found_refl = cPickle.load(f)
refl_select = spot_utils.ReflectionSelect(found_refl)

image_contents = open(image_fname,"r").read()
image_contents = image_contents.replace("run = 96", "run = %d" % run)
new_image_fname = "run%d/loc_d9114_run%d.txt" % (run,run)
with open( new_image_fname, "w") as out:
    out.write( image_contents)

print('Loading format')
loader = dxtbx.load(new_image_fname)
imgset = loader.get_imageset(loader.get_image_file())

print('Counting spots')
idx, Nspot_at_idx = spot_utils.count_spots(pickle_fname)
where_hits = np.where(Nspot_at_idx > MIN_SPOT_PER_HIT)[0]
Nhits = where_hits.shape[0]

# ============
output_h5_name = os.path.join(output_dir, 
    "run%d_hits_%s.h5" % (loader.run_number, output_tag))

with h5py.File( output_h5_name, "w") as out_h5:

    out_h5.create_dataset("panel_masks", data=loader.cspad_mask, dtype=np.bool)
    out_h5.create_dataset("panel_gainmasks", data=loader.gain)
    panel_x, panel_y = loader.cspad.coords_xy(loader._get_event(where_hits[0]))
    out_h5.create_dataset("panel_x", data=panel_x/109.92)
    out_h5.create_dataset("panel_y", data=panel_y/109.92)
    out_h5.create_dataset("panel_z", data=np.ones_like(panel_x)*loader.detector_distance)
    out_h5.create_dataset("pedestal", data=loader.dark)
    out_h5.create_dataset("gain_val", data=loader.nominal_gain_val)

    panel_dset = out_h5.create_dataset("panels",
                                       dtype=np.int16,
                                       shape=(Nhits, 32, 185, 388))
    times_dset = out_h5.create_dataset("event_times",
                                       dtype=np.int64,
                                       shape=(Nhits,))
    
    seconds, nanoseconds, fids  = [],[],[]
    #psana_time_nums = []
    for i_hit in range(Nhits):
        print '\rSaving hit {:d}/{:d}'.format(i_hit+1, Nhits),
        sys.stdout.flush()
        shot_idx = idx[where_hits[i_hit]]
        
        t = loader.times[ shot_idx]  # event time
        sec, nsec, fid = t.seconds(), t.nanoseconds(), t.fiducial()
        t_num, _ = utils.make_event_time( sec, nsec, fid)
        #seconds.append( sec)
        #nanoseconds.append( nsec)
        #fids.append( fid)
        panel_dset[i_hit] = loader.get_psana_raw(shot_idx)
        times_dset[i_hit] = t_num
        #psana_time_nums.append( t_num)

