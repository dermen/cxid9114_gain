# coding: utf-8
import glob
import sys
import pylab as plt


imdir = sys.argv[1]

sim_idx = int( sys.argv[2])

try:
    pause=float( sys.argv[3])
except IndexError:
    pause=0.1


im = plt.imshow( [[10,10],[10,10]])
plt.draw()
plt.pause(0.1)


img_dirs = glob.glob("%s/panel*" % imdir)
print (img_dirs)
for img_dir in img_dirs:
    print (img_dir)
    fnames = glob.glob("%s/*.jpg" % img_dir)
    fnames = sorted(fnames, key=lambda x: int(x.split("trial")[1].split(".")[0]) )
    for i_f,f in enumerate(fnames):
        
        if i_f != sim_idx: #len(fnames)-1:
            continue
        imdata = plt.mpl.image.imread(f)  # NOTE : requires PILLOW for jpg
        im.set_data( imdata)
        plt.draw()
        plt.pause(pause)
    plt.pause(.01)
plt.show()

