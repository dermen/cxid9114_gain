
import numpy as np
import pylab as plt

import geom_help


def load_psf_from_npz(psf_npz_fname):
    d = np.load(psf_npz_fname)
    psf = np.array([d['p'], d['s'], d['f']])
    return psf


def cscale(img, contrast=0.1):
    """ img is a 2D image to scale, contrast is a slider parameter"""
    m90 = np.percentile(img, 90) 
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)


def plot_assem(asics, psf, contrast=0.075, panel_ids=None):
    """
    list of CSPAD asics (64 asic format)
    psf is the geometry specifier, see for Psana Detector Geometry access get_psf method
        it is origin, slow, fast scan of each asic
    """
    p64, s64, f64 = geom_help.cspad_geom_splitter(psf, 'pixels')

    fig = plt.figure()

    ax = plt.gca()
    ax.set_facecolor('dimgray')

    imshow_arg = {"vmin":-0.1, "vmax":1, "interpolation":'none', "cmap":"gray"}
   
    if panel_ids is None:
        panel_ids = np.arange(len(asics))
    for i_pan, pid in enumerate( panel_ids):
        img = asics[i_pan].copy().astype(np.float64)
        img = cscale(img, contrast)
        geom_help.add_asic_to_ax( ax=ax, I=img,
                    p=p64[pid], fs=f64[pid],ss=s64[pid], s='', **imshow_arg)
    ax.set_xlim((-900,900)) 
    ax.set_ylim((900,-900)) 

    # plot the direct beam
    circ1 = plt.Circle(xy=(0,0), radius=1, fc='k', ec='none')  
    circ2 = plt.Circle(xy=(0,0), radius=3, fc='none', ec='k', lw=3)  
    ax.add_patch(circ1)
    ax.add_patch(circ2)

    #plt.draw()
    #plt.pause(0.1)
    return fig, ax


if __name__ =="__main__":
    import sys
    dataname = sys.argv[1]
    data = np.load(dataname)
    title = sys.argv[2]
    try:
        contrast = float(sys.argv[3])
    except IndexError:
        contrast = 0.075
    psf = load_psf_from_npz("xfel_psf.npz") 
    img2 = data['sims_wNoise']
    img2 = data['sims']
    img2 = np.array([np.random.normal(I+10, I.mean()*2) for I in img2])
    try:
        vmax = float(sys.argv[4])
    except IndexError:
        vmax=None
    try:
        vmin = float(sys.argv[5])
    except IndexError:
        vmin=None
    if 'panel_ids' in data.keys():
        panel_ids = data['panel_ids']
    else:
        panel_ids = None

    fig,ax = plot_assem(img2, psf,contrast, panel_ids=panel_ids)
    ax.set_title(title)
    ax.set_aspect("auto")

    plt.show()
