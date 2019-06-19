import numpy as np
import pylab as plt
import lmfit
from cxid9114 import fit_utils, utils
from cxid9114.mask import mask_utils


def get_gain_dists(panel_data, gain_data, mask_data=None, plot=False, norm=False,
            bins_low=None, bins_high=None):
    """
    this processes the panel data and applies common mode to the panels
    different gain sections individually.
    """
    if mask_data is None:
        mask_data = np.ones_like( gain_data)

    panel_data2 = np.zeros_like( panel_data)

    #   intensity bins for LD91 (specific)
    if bins_low is None:
        bins_low = np.linspace(-10,20, 300) # in ADU
    bc_low = .5*(bins_low[1:] + bins_low[:-1]) # bin centers
    if bins_high is None:
        bins_high = np.linspace(-20,50, 200) # in ADUs
    bc_high = .5*(bins_high[1:] + bins_high[:-1]) # bin centers

    i1_low = np.argmin( np.abs(bc_low+10))
    i2_low = np.argmin( np.abs(bc_low-10))
    i1_high = np.argmin( np.abs(bc_low+20))
    i2_high = np.argmin( np.abs(bc_low-15))
    xdata_low = bc_low[ i1_low:i2_low]
    xdata_high = bc_high[ i1_high:i2_high]

    #   these are the panel indices to use to form the dists
    low_gain_idx = [0,1,7,8,9,15,16,17,23,24,25,31]
    high_gain_idx =[0,2,3,4,5,6,7,8,10,11,12,14,15,16,
                    18,19,20,22,23,24,26,27,28,30,31]
    low_gain_dists = []
    high_gain_dists = []

    low_gain_fits = {}
    high_gain_fits ={}

    gauss_params_low = lmfit.Parameters()
    gauss_params_low.add('amp', value=0.25, min=0)
    gauss_params_low.add('wid', value=3, min=1)
    gauss_params_low.add('mu', value=0, min=-5, max=5)

    gauss_params_high = lmfit.Parameters()
    gauss_params_high.add('amp', value=0.12, min=0)
    gauss_params_high.add('wid', value=3, min=1)
    gauss_params_high.add('mu', value=0, min=-3, max=3)

    for i_pan in range(len(panel_data)):

        g = panel_data[i_pan].copy()
        is_low = gain_data[i_pan]*mask_data[i_pan]
        is_high = (~gain_data[i_pan])*mask_data[i_pan]

        Nlow = is_low.sum()
        if Nlow > 0:
            sig_low_gain = np.histogram(g[is_low].ravel(),
                                         bins=bins_low,density=True)[0]
            ydata_low = sig_low_gain[i1_low:i2_low]

            result_low_gain = lmfit.minimize(fit_utils.gauss_standard,
                                             gauss_params_low,
                                             args=(xdata_low, ydata_low ))

            low_fit = fit_utils.gauss_standard(result_low_gain.params,
                                     xdata_low, np.zeros_like( xdata_low))
            low_gain_fits[i_pan] = result_low_gain
            mu_low = result_low_gain.params['mu'].value

            mu_low = xdata_low[ np.argmax(utils.smooth(ydata_low, window_size=30))]
            if plot:
                plt.figure()
                ax=plt.gca()
                ax.plot( xdata_low, ydata_low)
                ax.plot( xdata_low, low_fit)
                plt.show()

            panel_data2[i_pan][is_low] = panel_data[i_pan][is_low]-mu_low

        Nhigh = is_high.sum()
        if Nhigh > 0:
            sig_high_gain = np.histogram(g[is_high].ravel(),
                                          bins=bins_high,density=True)[0]
            ydata_high = sig_high_gain[i1_high:i2_high]

            result_high_gain = lmfit.minimize(fit_utils.gauss_standard,
                                              gauss_params_high,
                                              args=(xdata_high, ydata_high ))

            high_fit = fit_utils.gauss_standard(result_high_gain.params,
                                      xdata_high, np.zeros_like( xdata_high))
            high_gain_fits[i_pan] = result_high_gain
            mu_high = result_high_gain.params['mu'].value
            mu_high = xdata_high[np.argmax(utils.smooth(ydata_high, window_size=30))]
            if plot:
                plt.figure()
                ax=plt.gca()
                ax.plot( xdata_high, ydata_high)
                ax.plot( xdata_high, high_fit)
                plt.show()
            panel_data2[i_pan][is_high] = panel_data[i_pan][is_high]-mu_high

        if i_pan in low_gain_idx:
            sig_low_gain2 = np.histogram( g[is_low].ravel()-mu_low,
                                          bins=bins_low,density=True)[0]
            low_gain_dists.append(sig_low_gain2)

        if i_pan in high_gain_idx:
            sig_high_gain2 = np.histogram( g[is_high].ravel()-mu_high,
                                           bins=bins_high,density=True)[0]
            high_gain_dists.append(sig_high_gain2)

    return bc_low, np.mean(low_gain_dists,0), bc_high, np.mean(high_gain_dists,0), panel_data2

def correct_panels(data, gain_map, mask,plot=False):
    xlow,ylow,xhigh,yhigh,new_data = get_gain_dists( data, gain_map, mask)

    low_g0,low_g1,fit_low = fit_utils.fit_low_gain_dist(xlow,ylow,plot=plot)
    high_g0,high_g1,fit_high = fit_utils.fit_high_gain_dist(xhigh,yhigh,plot=plot)

    low_1phot = xlow[low_g1.argmax()]
    #high_1phot = xhigh[high_g1.argmax()]
    high_1phot = xhigh[np.argmax(utils.smooth(yhigh, window_size=30)[220:300]) + 220]
    print "Low gain 1 photon peak: %.4f ADU"%low_1phot
    print "High gain 1 photon peak: %.4f ADU"%high_1phot
    gain = high_1phot / low_1phot

    print "Estimated gain: %.4f"%gain

    low_0phot_wid = fit_low.params['wid0']
    high_0phot_wid = fit_high.params['wid0']
    bg_gain = high_0phot_wid / low_0phot_wid
    print "Estimated dark-current gain: %.4f"%bg_gain

    cutoff = low_1phot - 1*fit_low.params['wid1'].value/np.sqrt(2.)
    print "Estimated low-gain dark-current cutoff ADU: %.4f"%cutoff

    #cutoff = 1.85
    #gain = 6.85
    #bg_gain = 1.95

    lowgain_photons = np.logical_and(new_data > cutoff, gain_map)
    new_data[gain_map] = new_data[gain_map] * bg_gain
    new_data[lowgain_photons] = new_data[lowgain_photons] * gain/bg_gain

    return new_data

def main():
    data =np.load("raw_peaks_img.npy")
    gain_map = np.load("gain2.npy")==2.
    mask = mask_utils.mask_small_regions(gain_map)
    new_data = correct_panels( data, gain_map, mask)

    plt.figure()
    plt.imshow( new_data[0],  vmin=-10,vmax=50,cmap='gnuplot')
    plt.show()

def main2():
    import psana
    ds = psana.DataSource("exp=cxid9114:run=62")
    events = ds.events()

    det = psana.Detector('CxiDs2.0:Cspad.0')
    dark = det.pedestals(62)
    gain_map = det.gain_mask(62) == 1
    mask = mask_utils.mask_small_regions(gain_map)
    mask2 = np.load("mask/details_mask.npy")
    mask *= mask2
    start = 0

    all_ylow = []
    all_yhigh = []
    for i in range(1200):
        if i < 1000:
            continue
        ev = events.next()
        if ev is None:
            continue
        if i < start:
            continue
        raw = det.raw( ev)
        if raw is None:
            continue
        data = raw - dark
        # new_data = correct_panels( data, gain_map, mask, plot=True)

        xlow, ylow, xhigh, yhigh, new_data = get_gain_dists(data, gain_map, mask)
        all_ylow.append( ylow)
        all_yhigh.append( yhigh)

        #plt.figure()
        #plt.imshow( new_data[0],  vmin=-10,vmax=50,cmap='gnuplot')
        #plt.show()
        print i
    np.savez("all_shot_hists",
            ylow=all_ylow, yhigh=all_yhigh, xlow=xlow, xhigh=xhigh)


def main3():
    import psana
    ds = psana.DataSource("exp=cxid9114:run=62")
    events = ds.events()

    det = psana.Detector('CxiDs2.0:Cspad.0')
    dark = det.pedestals(62)
    gain_map = det.gain_mask(62) == 1
    plt.imshow(gain_map[0])
    plt.show()
    mask = mask_utils.mask_small_regions(gain_map)
    mask2 = np.load("details_mask.npy")
    mask *= mask2
    start = 0
    for i in range(100):
        ev = events.next()
        if ev is None:
            continue
        if i < start:
            continue
        data = det.calib(ev, cmpars=(5,0,0,0,0))
        #data = det.calib(ev, cmpars=(1,25,25,100,1))
        if data is None:
            continue
        plt.imshow( gain_map[0] )
        plt.show()    
        xlow,ylow,xhigh,yhigh,new_data = get_gain_dists( data, gain_map, mask)

        low_g0,low_g1,fit_low = fit_utils.fit_low_gain_dist(xlow,ylow,plot=1)
        high_g0,high_g1,fit_high = fit_utils.fit_high_gain_dist(xhigh,yhigh,plot=1)
        
        plt.figure()
        plt.imshow( data[0],  vmin=-10,vmax=50,cmap='gnuplot')
        plt.show()


if __name__=="__main__":
    main2()

