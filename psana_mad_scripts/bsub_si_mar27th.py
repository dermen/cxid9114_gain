import sys
import glob
import subprocess
import os

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
run_num = int(sys.argv[3])
exp_names = glob.glob("results/run%d/exp*th.json" % run_num) 
Ntrial = 250
cuda = 0
for i in range(len(exp_names)):

    if i % n_jobs != rank:
        continue

    outdir = "jitterize3/run%d" % run_num 
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # print fnames, expnames
    dump_name = exp_names[i].replace("json", "pkl").replace("exp_","dump_")
    cmd = ["libtbx.python si_mar27th.py %s %s %d %s %d" % (
        exp_names[i], dump_name, cuda, outdir, Ntrial)]
    print " ".join(cmd)
    subprocess.call(" ".join(cmd), shell=True)
    
