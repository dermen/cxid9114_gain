
import subprocess
import sys
import dxtbx
import os
import numpy as np

run_num = int(sys.argv[1])
n_jobs = int( sys.argv[2])
Q = sys.argv[3]
script = "msi_feb8th.py"
tag = "feb8th"

barg = (run_num, Q, run_num)
bsub = "bsub -J msi.%d -q %s -o logs/msi.%d.log -n 1" 
bsub %= barg

img_file_str="""experiment = cxid9114
run = %d
mode = idx
detector_address = "CxiDs2.0:Cspad.0"
cspad  {
    dark_correction = True
    apply_gain_mask = True
    detz_offset = 572.3938
    common_mode = default
}
d9114 {
    common_mode_algo = pppg
    savgol_polyorder = 3
    mask = d9114_32pan_mask.npy
}""" % run_num

img_file_name = "image_files/_autogen_run%d.loc" % run_num
with open(img_file_name,"w") as img_fid:
    img_fid.write(img_file_str)

loader = dxtbx.load(img_file_name)
N_img = loader.get_num_images()
del loader

starts = np.array_split(np.arange(N_img), n_jobs)
print starts

#Nper = N_img / n_jobs
#Nper = int( Nper)


output_dir = "results/run%d" % run_num
if not os.path.exists(output_dir):
    os.makedirs( output_dir)

for jid in range( n_jobs):
    start = starts[jid][0]
    N = len(starts[jid])
    cmd = [
      "libtbx.python",
      script,
      img_file_name,
      output_dir,
      str(start),
      str(N),
      "ref1_det.pkl",
      "ref3_beam.pkl",
      tag]

    cmd = " ".join( cmd)
    bcmd = "%s %s " % (bsub,  cmd)
    #print bcmd
    #os.system(bcmd)
    subprocess.call(bcmd, shell=True)
