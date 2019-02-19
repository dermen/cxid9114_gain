# coding: utf-8
from cxid9114 import utils
from cxid9114.spots import spot_utils
import numpy as np


def tilting_plane(img, mask=None, zscore=2 ):
    """
    fit tilting plane to img data, used for background subtraction of spots
    :param img:  numpy image
    :param mask:  boolean mask, same shape as img, True is good pixels
        mask should include strong spot pixels and bad pixels, e.g. zingers
    :param zscore: modified z-score for outlier detection, lower increases number of outliers
    :return: tilting plane, same shape as img
    """
    Y,X = np.indices( img.shape)
    YY,XX = Y.ravel(), X.ravel()

    img1d = img.ravel()

    if mask is None:
        mask = np.ones( img.shape, bool)
    mask1d = mask.ravel()

    out1d = np.zeros( mask1d.shape, bool)
    out1d[mask1d] = utils.is_outlier( img1d[mask1d].ravel(), zscore)
    out2d = out1d.reshape (img.shape)

    fit_sel = np.logical_and(~out2d, mask)  # fit plane to these points, no outliers, no masked
    x,y,z = X[fit_sel], Y[fit_sel], img[fit_sel]
    guess = np.array([np.ones_like(x), x, y ] ).T
    coeff, r, rank, s = np.linalg.lstsq(guess, z)
    ev = (coeff[0] + coeff[1]*XX + coeff[2]*YY )
    return ev.reshape( img.shape), out2d


def integrate(R, badmask, data, gain=28, fit_bg=True):
    """
    get the crystal scatter, background scatter, and photon counting noise
    for the reflections listed in the table R
    :param R:  reflection table
    :param badmask: mask in numpy format, same shape as data
    :param data: data
    :param gain: detector gain per photon
    :return: 3 arrays, one for signal, background and noise
    """

    Nrefl = len( R)

    Rpp = spot_utils.refls_by_panelname(R)  # this is a dictionary whose key (0-63) unlock that panels reflections
    allspotmask = {}
    bgtilt = {}
    for pid in Rpp:

        # load the spot mask for all strong spots for this panel 
        allspotmask[pid] = spot_utils.strong_spot_mask( Rpp[pid], ( 185, 194))
        if fit_bg:
            bgtilt[pid], _ = tilting_plane( data[pid],
                    mask= (~allspotmask[pid])*badmask[pid], zscore=8)

    signa = np.zeros( Nrefl)
    bg = np.zeros( Nrefl)
    noise = np.zeros_like( signa)
    for i_r, refl in enumerate(R):
        pid = refl['panel']
        
        spotmask = refl['shoebox'].mask.as_numpy_array() == 5
        f1, f2, s1, s2, _,_ = refl['shoebox'].bbox   # fast scan and slow scan edges of bounding box
        
        thisspotmask = np.zeros_like( allspotmask[pid])
        thisspotmask[s1:s2,f1:f2] = spotmask
        
        #smask = (thisspotmask ^ allspotmask[pid])
     
        signa[i_r] = data[pid] [thisspotmask].sum()
        if fit_bg:
            bg[i_r] = bgtilt[pid][thisspotmask].sum()
        # else bg[i_r]  is going to remain 0
        signa[i_r] = (signa[i_r]-bg[i_r]) / gain
        noise[i_r] = np.sqrt(signa[i_r])
        
    return signa, bg, noise
