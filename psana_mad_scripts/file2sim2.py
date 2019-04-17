# coding: utf-8
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from cxid9114 import utils
#D = utils.open_flex("dump_18555_feb8th_resid.pkl")
get_ipython().magic(u'pwd ')
D = utils.open_flex("results/run120/dump_62573_feb8th_resid.pkl")
loader = dxtbx.load("image_files/_autogen_run120.loc")
import dxtbx
loader = dxtbx.load("image_files/_autogen_run120.loc")
ISET = loader.get_imageset( loader.get_image_file())
#iset = ISET[shot_idx:shot_idx+1]
shot_idx = 62573
run=120
iset = ISET[shot_idx:shot_idx+1]
FLUX = [1e12,1e12]
FF = [1e4,None]
beamA = D['beamA']
beamB = D['beamB']
det = D['detector']
cryst = D["optCrystal"]
simsAB_old = sim_utils.sim_twocolors2(
    cryst, det, beamA, FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=20, mos_spread=0.0)
sim_utils.save_twocolor(simsAB_old, iset, "run%d_shot%d.hdf5" %(run, shot_idx))
raws = iset.get_raw_data(0)
img = raws[62].as_numpy_array()
get_ipython().magic(u'pylab')
imshow( img, vmax=20)
imshow( img, vmax=200)
x = 145.537-.5
y = 119.91-.5
plot( x,y,'o', ms=16, mec='r', mfc='none')
fig = figure()
ax = fig.gca(projection='3d')
from mpl_toolkits.mplot3d import Axes3D
ax = fig.gca(projection='3d')
Y,X = np.indices( img.shape)
Z = img
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
Z[Z<1] = 1
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
clf()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
figure(1)
imshow( img, vmax=200, origin='lower')
figure(3)
figure(2)
ax.set_xlabel("x", fontsize=18, labelpad=10)
ax.set_ylabel("y", fontsize=18, labelpad=10)
ax.set_zlabel("$\log(I)$", fontsize=22, labelpad=10)
ax.tick_params(labelsize=13)
ax.tick_params(labelsize=12)
figure(1)
imshow( np.log1p(Z), vmax=200, origin='lower')
imshow( np.log1p(Z), vmax=10, origin='lower')
colorbar()
from cxid9114.spots import spot_utils
spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=30)
reflsA = D["refls_simA"]
reflsB = D["refls_simB"]
refls_strong = D['refls_strong']
spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=30)
get_ipython().magic(u'pinfo spot_utils.plot_overlap')
det
det = D['detector']
spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=30)
shot_idx
#shot_idx = 18555
D = utils.open_flex("results/run82/dump_18555_feb8th_resid.pkl")
D = utils.open_flex("results/run81/dump_18555_feb8th_resid.pkl")
reflsA = D["refls_simA"]
reflsB = D["refls_simB"]
refls_strong = D['refls_strong']
spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=30)
spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=50)
get_ipython().magic(u'save file2sim2.py 1-91')
