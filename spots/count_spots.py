import pickle
import numpy as np

import sys
sys.path = ['/home/dermen/.local/lib/python2.7/site-packages'] + sys.path
refl = pickle.load(open("strong2.pickle"))

Nrefl = len(refl)
# because these are stills,
# the z1 coordinate of the bounding box
# specifies frame index
# and z2 should always be z1+1
x1,x2,y1,y2,z1,z2 = map( np.array, zip(* [refl['bbox'][i] for i in range(Nrefl)]))
assert( np.all(z2==z1+1))

#
from collections import Counter
import pylab as plt
shot_idx, Nspots_per_shot = map( np.array, zip(*Counter(z1).items()))
plt.figure()
bins = np.logspace(0, 3, 50)
ax = plt.gca()
plt.hist(Nspots_per_shot, bins=bins, )
ax.set_yscale("log")
ax.set_xscale("log")
ax.tick_params(labelsize=14)
ax.grid(1, which='both')
plt.xlabel("Spots per shot", fontsize=14)
plt.ylabel("bincount", fontsize=14)

cutoff = 40
print ("there are %d shots with >= %d spots" % ((Nspots_per_shot > cutoff).sum(), cutoff))

plt.show()

