# coding: utf-8
from cxid9114 import utils, sim, spots
D = utils.open_flex("dump_18555_feb8th_resid.pkl")
D.keys()
reflsA = D["refls_simA"]
reflsB = D["refls_simB"]
get_ipython().magic(u'pinfo spots.spot_utils.plot_overlap')
det = D['detector']
refls_strong = D['refls_strong']
spots.spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=10))
spots.spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=10)
spots.spot_utils.plot_overlap(reflsA, reflsB, refls_strong, det, square_s=30)
beamA = D['beamA']
beamB = D['beamB']
cryst = D["optCrystal"]
simsAB_old = sim_utils.sim_twocolors2(
    cryst, det, beamA, FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=20, mos_spread=0.0)
from cxid9114 import prameters
FF = [1e4,None]
FLUX = [1e12,1e12]
simsAB_old = sim.sim_utils.sim_twocolors2(
    cryst, det, beamA, FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=20, mos_spread=0.0)
from cxid9114.sim import sim_utils
simsAB_old = sim_utils.sim_twocolors2(
    cryst, det, beamA, FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=20, mos_spread=0.0)
from cxid9114 import parameters
simsAB_old = sim_utils.sim_twocolors2(
    cryst, det, beamA, FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=20, mos_spread=0.0)
get_ipython().magic(u'pwd ')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'cd _autogen_run100.loc')
get_ipython().magic(u'cd image_files/')
import dxtbx
loader = dxtbx.load("_autogen_run81.loc")
# = utils.open_flex("dump_18555_feb8th_resid.pkl")
shot_idx = 18555
ISET = loader.get_imageset( loader.get_image_file())
iset = ISET[shot_idx:shot_idx+1]
get_ipython().magic(u'pinfo sim_utils.save_twocolor')
sim_utils.save_twocolor(simsAB_old, iset, "run%d_shot%2.hdf5" %(81, shot_idx))
#sim_utils.save_twocolor(simsAB_old, iset, "run%d_shot%d.hdf5" %(81, shot_idx))
get_ipython().magic(u'pwd ')
get_ipython().magic(u'cd ../')
sim_utils.save_twocolor(simsAB_old, iset, "run%d_shot%d.hdf5" %(81, shot_idx))
raws =iset.get_raw_data(0)
img = raws[1].as_numpy_array()
get_ipython().magic(u'pylab')
imshow( img )
imshow( img , vmax=200)
x = 144.627-.5
y = 21.4534 - .5
plot( x,y,'o', ms=10, mec='r', mfc='none')
ax = gca()
ax.lines.pop()
plot( x,y,'o', ms=16, mec='r', mfc='none')
imshow( img , vmax=300)
imshow( img , vmax=800)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
img.shape
Y,X = np.indices( img.shape)
Z = img
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_facecolor("grey")
ax.set_facecolor("white")
ax.set_facecolor("orange")
ax.clear()
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       linewidth=0, antialiased=True)
ax.set_zscale("log")
ax.set_zscale("linear")
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
clf()
ax = gca()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
get_ipython().magic(u'pinfo log1p')
Z[Z<1] = 1
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
clf()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, log1p(Z), cmap='viridis',
                       linewidth=0, antialiased=True)
ax.set_zlabel("$\log(I)$", fontsize=24)
ax.set_zlabel("$\log(I)$", fontsize=22)
ax.set_zlabel("$\log(I)$", fontsize=22, labelpad=10)
ax.tick_params(labelsize=14)
ax.set_xlabel("x", fontsize=18)
ax.set_ylabel("y", fontsize=18)
ax.set_ylabel("y", fontsize=18, labelpad=10)
ax.set_xlabel("x", fontsize=18, labelpad=10)
figure(1)
imshow( img , vmax=800, origin='lower')
get_ipython().magic(u'pwd ')
