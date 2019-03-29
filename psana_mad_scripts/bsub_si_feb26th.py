import sys
import glob
import subprocess
import os
import numpy as np

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
odir = "simdata2"

exp_names = glob.glob("results/run62/exp*th.json") 

for i in range(len(exp_names)):
    if i % n_jobs != rank:
        continue
    # print fnames, expnames
    dump_name = exp_names[i].replace("json", "pkl").replace("exp_","dump_")
    
    ofile = os.path.basename( dump_name)
    ofile = ofile.replace(".pkl", "_int.pdpkl")
    ofile = os.path.join( odir, ofile)
    cmd = ["libtbx.python si_feb26th.py %s %s %s" % (
        exp_names[i], dump_name, ofile)]
    print " ".join(cmd)
    subprocess.call(" ".join(cmd), shell=True)
    
