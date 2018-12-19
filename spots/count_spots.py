import cPickle as pickle
from dials.array_family import flex
import numpy as np
from collections import Counter
import sys

def count_spots(pickle_fname):
    refl = pickle.load(open(pickle_fname,"r"))
    Nrefl = len(refl)
    # because these are stills,
    # the z1 coordinate of the bounding box
    # specifies frame index
    # and z2 should always be z1+1
    x1,x2,y1,y2,z1,z2 = map( np.array, zip(* [refl['bbox'][i] for i in range(Nrefl)]))
    assert( np.all(z2==z1+1))
    shot_idx, Nspots_per_shot = map( np.array, zip(*Counter(z1).items()))
    return shot_idx, Nspots_per_shot

if __name__=="__main__":
    plot = True 
    fname = sys.argv[1]
    cutoff = int(sys.argv[2]) # min number of spots  per hit
    shot_idx, Nspots_per_shot = count_spots(fname)
    print ("there are %d / %d shots with >= %d spots" % \
        ((Nspots_per_shot > cutoff).sum(), len(shot_idx), cutoff))
    
    if plot:
        import pylab as plt
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
        plt.show()

