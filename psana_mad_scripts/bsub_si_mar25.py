import sys
import glob
import subprocess
import os

rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
tag = sys.argv[3]
jitt = sys.argv[4]

exp_names = glob.glob("results/run*/exp*feb8th.json")
dump_names = glob.glob("results/run*/*resid.pkl")
f2 = lambda x: x.replace("exp", "dump").replace(".json", "_%s.pkl"%tag)

exp_names2 = []
for i,f in enumerate(exp_names):
    if i % 50==0:
        print i
    if f2(f) in dump_names:
        continue
    exp_names2.append( f)

np.savetxt("exp_names2", exp_names2, fmt='%s')

print len( exp_names2)
exit()

#out_f2 = []
#for f in exp_names:
#    if not os.path.exists(f2):
#        out_f2.append( f2)

print len( exp_names)
exit()
#exit()

import random
random.shuffle(exp_names)

for i in range(len(exp_names)):
    if i % n_jobs != rank:
        continue
    # print fnames, expnames
    dump_name = exp_names[i].replace("json", "pkl").replace("exp_","dump_")
    cmd = ["libtbx.python si_feb9th.py %s %s %s %s" % (
        exp_names[i], dump_name, tag, jitt)]
    print " ".join(cmd)
    subprocess.call(" ".join(cmd), shell=True)

