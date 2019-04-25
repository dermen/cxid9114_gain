# coding: utf-8
import h5py
import geom_help
import numpy as np
import pylab as plt
import os,sys


try:
    img_tag = sys.argv[2]
except IndexError:
    img_tag = ""

tag = sys.argv[1] 

fname = os.path.join( tag, "simparams_%s.h5py" % tag )

f = h5py.File(fname, "r")
d = np.load("xfel_psf.npz")
psf = np.array([d['p'], d['s'], d['f']])
p64, s64, f64 = geom_help.cspad_geom_splitter(psf, 'pixels')

spotdata = np.load("crystR.spotdata.pkl.npz")
Malls = spotdata["Malls"][()]

fig = plt.figure()
fig.set_size_inches([12.46,  6.65])
ax = plt.gca()
ax.set_aspect('auto')
ax.set_xlim(-900,900)  # work in pixel units
ax.set_ylim( 900,-900)
ax.set_facecolor('dimgray')

cmap = plt.cm.Blues
cmap.set_bad(color='Darkorange')
imshow_arg = {"vmin":1e-3, "vmax":1, "interpolation":'none', "cmap":cmap}

sim_keys = f.keys()
# for now, order the sim keys according to the integer
sim_keys  = sorted( sim_keys, key=lambda x: int(x.split("_")[-1]))

subp = {'left':.05, 'bottom':0.05, 'right':1, 'top':1}

for i_fig,k in enumerate(sim_keys):
    asic64 = f[k]
    ax.clear()
    for i in range(64): 
        img = asic64[i].copy()
        mask = np.any(Malls[i],axis=0)
        img[mask] = np.nan
        geom_help.add_asic_to_ax( ax=ax, I=img, 
                    p=p64[i], fs=f64[i],ss=s64[i], s='', **imshow_arg)
    #ax.set_xlim((14.193155793403356, 873.3295699909963))
    #ax.set_ylim((35.81022017425094, -381.0165570529714))
    ax.set_xlim((14.193155793403356, 400)) #873.3295699909963))
    ax.set_ylim((35.81022017425094, -150))  #-381.0165570529714))
    ax.set_aspect('auto')
    plt.subplots_adjust(**subp)
    
    figname = os.path.join( tag, "%s%s.png" % (k, img_tag) )
    plt.savefig(figname, dpi=150)
    print i_fig, len( sim_keys)
    print "saved %s" % figname
    print



