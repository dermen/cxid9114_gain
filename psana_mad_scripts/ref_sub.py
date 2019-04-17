import sys
import os


rank = int(sys.argv[1])
xtc_file = sys.argv[2]
outdir = sys.argv[3]

stride = 2500
first = 0
last = 70000
starts = np.arange(first, last,stride)


for i_s,s in enumerate(starts):
    if i_s != rank:
        continue
    cmd = ["libtbx.python ssirp.py %s %s" % (expnames[i].strip(), fnames[i].strip())]
    print " ".join(cmd)
    os.system(" ".join(cmd))
