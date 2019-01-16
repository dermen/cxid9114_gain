from collections import Counter
from itertools import groupby
import cPickle as pickle
import sys
import numpy as np
from copy import deepcopy
from dials.array_family import flex

def count_spots(pickle_fname):
    """
    Count the number of spots in a pickle file per shot index
    :param pickle_fname: path to a strong.pickle
    :return: tupe of two arrays: (shot index, number of spots per that index)
    """
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

def group_refl_by_shotID(refl):
    """
    groups shots according to shot index
    :param refl: flex reflection table
    :return: grouped objetcs dictionary, keys are the shot index, values are the reflections
    """
    get_shot_idx = lambda x: x['bbox'][4]
    grouped = groupby( sorted(refl,key=get_shot_idx), get_shot_idx)
    return {shot_idx:list(v) for shot_idx,v in grouped}

class ReflectionSelect:
    def __init__(self, refl_tbl):
        """
        caches the shot index of reflections
        :param refl_tbl:
        """
        self.refl = refl_tbl
        self.Nrefl = len( refl_tbl)
        self.all_shot_idx = flex.int( [ refl_tbl['bbox'][i][4] for i in range(self.Nrefl)])

    def select(self, shot_idx):
        shot_refl = self.refl.select( self.all_shot_idx==shot_idx )

        bbox = shot_refl['bbox']
        sb = shot_refl["shoebox"]
        for i in range(len(shot_refl)):
            bbox[i] = (bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], 0, 1)
            sb_bbox = sb[i].bbox
            sb[i].bbox = (sb_bbox[0], sb_bbox[1], sb_bbox[2], sb_bbox[3], 0, 1)
        return shot_refl

def as_single_shot_reflections(refl_, inplace=True):
    """
    sets all z-coors to 0,1
    :param refl_: reflection table , should be just single image
    :return: updated table
    """
    if not inplace:
        refl = deepcopy( refl_)
    else:
        refl = refl_
    bbox = refl['bbox']
    sb = refl["shoebox"]
    for i in range(len(refl)):
        bbox[i] = (bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], 0, 1)
        sb_bbox = sb[i].bbox
        sb[i].bbox = (sb_bbox[0], sb_bbox[1], sb_bbox[2], sb_bbox[3], 0, 1)
    if not inplace:
        return refl


def select_refl(refl, shot_idx):
    """
    selects reflections belinging to a particular shot index
    :param refl: reflection table
    :param shot_idx: int, shot index
    :return:
    """
    from IPython import embed
    embed()
    n = len( refl)
    select_me = flex.bool([refl['bbox'][i][4] == int(shot_idx) for i in range(n)])
    return refl.select(select_me)

if __name__=="__main__":
    plot = True
    fname = sys.argv[1]  # path to a strong.pickle
    cutoff = int(sys.argv[2])  # min number of spots  per hit
    shot_idx, Nspots_per_shot = count_spots(fname)
    print ("there are {:d} / {:d} shots with >= {:d} spots"
           .format((Nspots_per_shot > cutoff).sum(), len(shot_idx), cutoff))

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

