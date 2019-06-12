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
    return ev.reshape( img.shape), out2d, coeff


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
            bgtilt[pid], _,_ = tilting_plane( data[pid],
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

def integrate2(R, badmask, data, gain=28, fit_bg=True, zscore=8, sz=8):
    """
    get the crystal scatter, background scatter, and photon counting noise
    for the reflections listed in the table R
    :param R:  reflection table
    :param badmask: mask in numpy format, same shape as data
    :param data: data
    :param gain: detector gain per photon
    :return: 3 arrays, one for signal, background and noise
    """
    from dials.algorithms.shoebox import MaskCode
    fg_code = MaskCode.Foreground.real
    Nrefl = len(R)
    fs_dim = 194
    ss_dim = 185

    Rpp = spot_utils.refls_by_panelname(R)  # this is a dictionary whose key (0-63) unlock that panels reflections
    allspotmask = {}
    for pid in Rpp:
        # load the spot mask for all strong spots for this panel
        allspotmask[pid] = spot_utils.strong_spot_mask(Rpp[pid], (ss_dim, fs_dim))

    signa = np.zeros(Nrefl)
    bg = np.zeros(Nrefl)
    noise = np.zeros_like(signa)
    pix_per = np.zeros(Nrefl, int)
    for i_r, refl in enumerate(R):
        pid = refl['panel']

        spotmask = refl['shoebox'].mask.as_numpy_array() & fg_code == fg_code
        f1, f2, s1, s2, _, _ = refl['shoebox'].bbox  # fast scan and slow scan edges of bounding box
        icent,jcent,_ = refl['xyzobs.px.value']

        thisspotmask = np.zeros_like(allspotmask[pid])
        thisspotmask[s1:s2, f1:f2] = spotmask

        i1 = int(max(icent-.5-sz, 0))
        i2 = int(min(icent-.5+sz , fs_dim))
        j1 = int(max(jcent-.5-sz, 0))
        j2 = int(min(jcent-.5+sz , ss_dim))
        sub_data = data[pid][j1:j2, i1:i2]
        sub_mask =  ((~allspotmask[pid]) * badmask[pid] )[j1:j2, i1:i2]

         
        sub_thisspotmask = thisspotmask[j1:j2,i1:i2]
        Is = sub_data[sub_thisspotmask].sum()
        
        if fit_bg:
            tilt, bgmask, coeff = tilting_plane(sub_data,
                        mask=sub_mask, zscore=zscore)
            
            bg_fit_mask = np.logical_and(~bgmask, sub_mask)
            m = sub_thisspotmask.sum()
            n = bg_fit_mask.sum()
            m2n = float(m)/float(n)  # ratio of number of background to number of strong spot pixels
            
            # modifuf Is according to background plane fit
            Is = Is - tilt[sub_thisspotmask].sum()
            # store background pix according to Leslie 99
            Ibg = m2n*sub_data[bg_fit_mask].sum()
        else:
            Ibg = 0
        
        signa[i_r] = Is  # signal in the spot
        bg[i_r] = Ibg  # background around the spot
        noise[i_r] = (Is + Ibg + m2n*Ibg) / gain
        pix_per[i_r] = thisspotmask.sum()

    return signa, bg, noise, pix_per


def integrate3(Rpanel, badmask, data, gain=28, fit_bg=True, zscore=8, sz=8):
    """
    get the crystal scatter, background scatter, and photon counting noise
    for the reflections listed in the table R
    :param R:  reflection table from a single panel
    :param badmask: mask in numpy format, same shape as data from a single panel
    :param data: data, shape of a single panel
    :param gain: detector gain per photon
    :return: 4 arrays, one for signal, background and noise, and pixel per spot
    """
    from dials.algorithms.shoebox import MaskCode
    fg_code = MaskCode.Foreground.real
    Nrefl = len(Rpanel)
    ss_dim, fs_dim = data.shape

    allspotmask = spot_utils.strong_spot_mask(Rpanel, (ss_dim, fs_dim))

    signa = np.zeros(Nrefl)
    bg = np.zeros(Nrefl)
    noise = np.zeros_like(signa)
    pix_per = np.zeros(Nrefl, int)
    for i_r, refl in enumerate(Rpanel):

        spotmask = refl['shoebox'].mask.as_numpy_array() & fg_code == fg_code
        f1, f2, s1, s2, _, _ = refl['shoebox'].bbox  # fast scan and slow scan edges of bounding box
        icent,jcent,_ = refl['xyzobs.px.value']

        thisspotmask = np.zeros_like(allspotmask)
        thisspotmask[s1:s2, f1:f2] = spotmask

        i1 = int(max(icent-.5-sz, 0))
        i2 = int(min(icent-.5+sz , fs_dim))
        j1 = int(max(jcent-.5-sz, 0))
        j2 = int(min(jcent-.5+sz , ss_dim))
        sub_data = data[j1:j2, i1:i2]
        sub_mask =  ((~allspotmask) * badmask )[j1:j2, i1:i2]

         
        sub_thisspotmask = thisspotmask[j1:j2,i1:i2]
        Is = sub_data[sub_thisspotmask].sum()
        
        if fit_bg:
            tilt, bgmask, coeff = tilting_plane(sub_data,
                        mask=sub_mask, zscore=zscore)
            
            bg_fit_mask = np.logical_and(~bgmask, sub_mask)
            m = sub_thisspotmask.sum()
            n = bg_fit_mask.sum()
            m2n = float(m)/float(n)  # ratio of number of background to number of strong spot pixels
            
            # modifuf Is according to background plane fit
            Is = Is - tilt[sub_thisspotmask].sum()
            # store background pix according to Leslie 99
            Ibg = m2n*sub_data[bg_fit_mask].sum()
        else:
            Ibg = 0
        
        signa[i_r] = Is  # signal in the spot
        bg[i_r] = Ibg  # background around the spot
        noise[i_r] = (Is + Ibg + m2n*Ibg) / gain
        pix_per[i_r] = thisspotmask.sum()

    return signa, bg, noise, pix_per

