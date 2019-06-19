import sys
import glob
import subprocess
import os

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
call = int(sys.argv[3])
tag = "_betelgeuse"
run_num = 63

all_shot_idx = [int(f.split("_")[1]) for f in glob.glob("results/run63/dump*feb8th.pkl")]

for i, shot_idx in enumerate(all_shot_idx):

    if i % n_jobs != rank:
        continue

    cmd_basic="libtbx.python vid_maker_apr1st.py %d %s" % (
        shot_idx, tag )
    
    print cmd_basic
   
    if call: 
        subprocess.call(cmd_basic, shell=True)
    
