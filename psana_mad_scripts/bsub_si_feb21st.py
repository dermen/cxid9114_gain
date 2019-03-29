import sys
import glob
import subprocess
import os
import numpy as np

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])

exp_names = glob.glob("results/run*/exp*th.json") 

cuda = 0
run_num = exp_names
for i in range(len(exp_names)):

    if i % n_jobs != rank:
        continue

    rund = exp_names[i].split("/")[1]
    outdir = os.path.join( "jitterize", rund)
    # print fnames, expnames
    dump_name = exp_names[i].replace("json", "pkl").replace("exp_","dump_")
    cmd = ["libtbx.python si_mar22nd.py %s %s %d %s" % (
        exp_names[i], dump_name, cuda , outdir )]
    print " ".join(cmd)
    #subprocess.call(" ".join(cmd), shell=True)
    
