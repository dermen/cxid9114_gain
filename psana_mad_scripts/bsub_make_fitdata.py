import sys
import glob
import subprocess
import os

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
call = int(sys.argv[3])

run_num = 63
best_names = glob.glob("jitterize3/run63/*jitt.pkl")

for i, best_f in enumerate(best_names):

    if i % n_jobs != rank:
        continue

    shot_idx = int(best_f.split("/")[-1].split("_")[1])
    outdir = "jitterize_compare/run%d" % run_num 
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    exp_f = "results/run%d/exp_%d_feb8th.json" % (run_num, shot_idx)
    dump_f = "results/run%d/dump_%d_feb8th.pkl" % (run_num, shot_idx)
    out_f_basic = os.path.join( outdir, "run%d_shot%d_basic.pkl" % (run_num, shot_idx))
    out_f_opt = os.path.join( outdir, "run%d_shot%d_opt.pkl" % (run_num, shot_idx))
    out_f_optMC = os.path.join( outdir, "run%d_shot%d_optMC.pkl" % (run_num, shot_idx))
    
    cmd_basic="libtbx.python make_fitdata_from_sims_and_data.py %s %s %s" % (
        exp_f, dump_f, out_f_basic)
    
    cmd_opt_MC = "libtbx.python make_fitdata_from_sims_and_data.py %s %s %s %s 1" % (
        exp_f, dump_f, out_f_optMC, best_f)
    
    cmd_opt = "libtbx.python make_fitdata_from_sims_and_data.py %s %s %s %s 0" % (
        exp_f, dump_f, out_f_opt, best_f)
    
    print cmd_basic
    print
    print cmd_opt_MC
    print 
    print cmd_opt
    print 
    print
   
    if call: 
        subprocess.call(cmd_basic, shell=True)
        subprocess.call(cmd_opt_MC, shell=True)
        subprocess.call(cmd_opt, shell=True)
    
