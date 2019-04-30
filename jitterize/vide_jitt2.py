#!/usr/local/bin/python3

import h5py
import geom_help
import numpy as np
import pylab as plt
import os,sys


def cscale(img, contrast=0.1):
    m90 = np.percentile(img, 90) 
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)

try:
    img_tag = sys.argv[2]
except IndexError:
    img_tag = ""

tag = sys.argv[1] 

fname = os.path.join( tag, "simparams_%s.h5py" % tag )

f = h5py.File(fname, "r")
if 'panel_ids' in f:
    panel_ids = f['panel_ids'].value
else:
    panel_ids = np.arange(64)

print (panel_ids)

pan_map = {pid:i_pan  for i_pan,pid in enumerate(panel_ids)}

d = np.load("xfel_psf.npz")
psf = np.array([d['p'], d['s'], d['f']])
p64, s64, f64 = geom_help.cspad_geom_splitter(psf, 'pixels')

# NOTE this pattern has no mono spec and no mosaicity
#data = np.load("noise_img3.npz")["img"]

# NOTE this patterns has poly spec but no mosaicity
#dataname = "with_spec_gamma5/ps2.crystRwith_spec_gamma5_spotdata.pkl.npz"
dataname = "orig_data_gamma2_alpha/cryst0_spotdata.pkl.npz" 
#data = np.load(dataname)['sims_wNoise']
data = np.load(dataname)['sims']
data = np.array([np.random.normal(I+10, I.mean()*2) for I in data])
#data = [data[ipid] for pid in panel_ids]
print (100)

#spotdata = np.load("crystR.spotdata.pkl.npz")
#Malls = spotdata["Malls"][()]

fig = plt.figure()
fig.set_size_inches([12.46,  6.65])
ax = plt.gca()
ax.set_aspect('auto')
ax.set_facecolor('dimgray')

cmap = plt.cm.gray
cmap.set_bad(color='Limegreen')

#cmap = plt.cm.gray_r
#cmap.set_bad(color='Deeppink')
#cmap = plt.cm.Blues
#cmap.set_bad(color='Darkorange')
imshow_arg = {"vmin":-0.1, "vmax":1, "interpolation":'none', "cmap":cmap}
#imshow_arg = {"vmin":0, "vmax":6e8, "interpolation":'none', "cmap":cmap}

sim_keys = [k for k in f.keys() if k.startswith("sim")]
# for now, order the sim keys according to the integer
sim_keys  = sorted( sim_keys, key=lambda x: int(x.split("_")[-1]))

subp = {'left':.05, 'bottom':0.05, 'right':1, 'top':1}
for i_fig,k in enumerate(sim_keys):
    model = f[k].value > 1e-3
    assert( model.shape[0] == len( panel_ids))
    #model = np.array([model[32], model[34], model[50]])
    ax.clear()
    for i_pan, pid in enumerate( panel_ids): 
        img = data[i_pan].copy().astype(np.float64)
        
        img = cscale(img, 0.075)

        img[model[i_pan]] = np.nan
        geom_help.add_asic_to_ax( ax=ax, I=img,
                    p=p64[pid], fs=f64[pid],ss=s64[pid], s='', **imshow_arg)
    
    ax.set_xlim((14.193155793403356-25, 400-25))
    ax.set_ylim((35.81022017425094, -150)) 
    plt.subplots_adjust(**subp)
    circ1 = plt.Circle(xy=(0,0), radius=1, fc='k', ec='none')  
    circ2 = plt.Circle(xy=(0,0), radius=3, fc='none', ec='k', lw=3)  
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    figname = os.path.join( tag, "%s%s.png" % (k, img_tag) )
    plt.savefig(figname, dpi=150)
    print (i_fig, len( sim_keys))
    print ("saved %s" % figname)
    print ("")


