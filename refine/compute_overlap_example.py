import glob
import pylab as plt
import numpy as np

import dxtbx
from scipy.spatial import cKDTree

from dxtbx.datablock import DataBlockFactory
from scitbx.matrix import sqr
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from cxid9114 import utils
from cxid9114.spots import spot_utils
from libtbx.phil import parse
from scipy import ndimage

interactive = False
use_fine = False
# toggles
dials_find_sim_spots = False
plot = False

# spot params:
find_spot_params = find_spots_phil_scope.fetch(source=parse("")).extract()
find_spot_params.spotfinder.threshold.dispersion.global_threshold = 0
find_spot_params.spotfinder.threshold.dispersion.sigma_strong = 0.5
find_spot_params.spotfinder.filter.min_spot_size = 6

fnames = glob.glob("results/good_hit_*_5shots.pkl")
#fnames = glob.glob("results/good_hit_*_10_cryst_weights.pkl")

print fnames

all_dists = []
all_Nspots = []
for fname in fnames:
    data = utils.open_flex(fname)

    crystal = data['crystal']
    xrot = data['optX']
    yrot = data['optY']
    if 'optX_fine' in data.keys() and use_fine:
        xrot2 = data['optX_fine']
        yrot2 = data['optY_fine']
        new_A = xrot2*yrot2*xrot*yrot* sqr(crystal.get_U()) * sqr(crystal.get_B())
        h5_name = fname.replace(".pkl", ".h5")
    else:
        new_A = xrot*yrot* sqr(crystal.get_U()) * sqr(crystal.get_B())
        h5_name = fname.replace(".pkl", ".h5")
    crystal.set_A(new_A)
    loader = dxtbx.load(h5_name)
    imgA = loader.get_raw_data(0).as_numpy_array()
    imgB = loader.get_raw_data(1).as_numpy_array()
    imgAB = loader.get_raw_data(2).as_numpy_array()

    #####################################
    # format of this special imageset
    # its 4 images,
    # 0th is simulated colorA,
    # 1st is simulated colorB
    # 2nd is two color,
    # 3rd is the data image that was indexed
    # we will grab just the first 3 simulated
    # images and find spots..

    xdata, ydata, _  = map( np.array, spot_utils.xyz_from_refl(data['refl']) )

    iset = loader.get_imageset(loader.get_image_file())
    dblockA = DataBlockFactory.from_imageset(iset[0:1])[0]
    dblockB = DataBlockFactory.from_imageset(iset[1:2])[0]
    dblockAB = DataBlockFactory.from_imageset(iset[2:3])[0]

    reflA = flex.reflection_table.from_observations(dblockA, find_spot_params)
    reflB = flex.reflection_table.from_observations(dblockB, find_spot_params)
    reflAB = flex.reflection_table.from_observations(dblockAB, find_spot_params)
    refl_dat = data['refl']  # experimental image observations are stored here..

    # adhoc thresholds:
    threshA = 0 #imgA[ imgA > 0].mean() * 0.05
    threshB = 0 #imgB[ imgB > 0].mean() * 0.05
    threshAB = 0 #imgAB[ imgAB > 0].mean() * 0.05

    labimgA, nlabA = ndimage.label(imgA > threshA)
    out = ndimage.center_of_mass(imgA, labimgA, range(1, nlabA))
    yA,xA = map( np.array, zip(*out))

    labimgB, nlabB = ndimage.label(imgB > threshB)
    out = ndimage.center_of_mass(imgB, labimgB, range(1, nlabB))
    yB,xB = map( np.array, zip(*out))

    labimgAB, nlabAB = ndimage.label(imgAB > threshAB)
    out = ndimage.center_of_mass(imgAB, labimgAB, range(1, nlabAB))
    yAB,xAB = map( np.array, zip(*out))

    xAB2, yAB2 = np.hstack((xA, xB)), np.hstack((yA, yB))

    if plot:
        ax = plt.gca()
        ax.clear()

        s = 4
        r = 5
        Square_styleA = {"ec": "C1", "fc": "C1", "lw": "1", "width": s, "height": s}
        Square_styleB = {"ec": "C3", "fc": "C3", "lw": "1", "width": s, "height": s}
        Circle_style = {"ec":"C0", "fc":"C0", "lw":"1", "radius":r}
        spot_utils.add_xy_to_ax_as_patches( zip(xdata,ydata), plt.Circle, Circle_style, ax)
        spot_utils.add_xy_to_ax_as_patches( zip(xA-s/2.,yA-s/2.), plt.Rectangle, Square_styleA, ax)
        spot_utils.add_xy_to_ax_as_patches( zip(xB-s/2.,yB-s/2.), plt.Rectangle, Square_styleB, ax)
        ax.set_ylim(0, 1800)
        ax.set_xlim(0, 1800)
        ax.set_aspect('equal')

        plt.figure()
        xAB, yAB = np.hstack((xA, xB)), np.hstack((yA, yB))
        plt.plot( xAB2, yAB2 , 'rs')
        plt.plot(xdata, ydata , '.', color='C0')

        if interactive:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.show()

    tree = cKDTree( zip( xAB2, yAB2) )
    dist, pos = tree.query( zip( xdata, ydata))
    all_dists.append( dist)
    all_Nspots.append( len(xdata))

from IPython import embed
embed()
