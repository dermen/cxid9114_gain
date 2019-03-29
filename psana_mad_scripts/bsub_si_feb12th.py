import sys
import glob
import subprocess
import os
import numpy as np

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
tag = sys.argv[3]
jitt = sys.argv[4]

exp_names = glob.glob("results/run[1][5][5-9]/exp*th.json") + glob.glob("results/run[1][6-9][0-9]/exp*th.json")  + glob.glob("results/run[2][0-9][0-9]/exp*th.json")


for i in range(len(exp_names)):
    if i % n_jobs != rank:
        continue
    # print fnames, expnames
    dump_name = exp_names[i].replace("json", "pkl").replace("exp_","dump_")
    cmd = ["libtbx.python si_feb9th.py %s %s %s %s" % (
        exp_names[i], dump_name, tag, jitt)]
    print " ".join(cmd)
    subprocess.call(" ".join(cmd), shell=True)

