import numpy as np
from scipy.signal import savgol_filter
import matplotlib

# @profile
def pppg(shot_, gain, mask=None, window_length=101, polyorder=5,
        low_x1=-10, low_x2 = 10, high_x1=-20, high_x2=20, Nhigh=1000,
         Nlow=500, plot_details=False, verbose=False, before_and_after=False,
         plot_metric=True, inplace=False):

    if not inplace:
        shot = shot_.copy()
    else:
        shot = shot_
    if mask is not None:
        is_low = gain*mask
        is_high = (~gain)*mask
    else:
        is_low = gain
        is_high = (~gain)

    low_gain_pid = np.where([ np.any( is_low[i] ) for i in range(32)])[0]
    high_gain_pid = np.where([ np.any( is_high[i] ) for i in range(32)])[0]

    bins_low = np.linspace(low_x1, low_x2, Nlow)
    bins_high = np.linspace(high_x1,high_x2,Nhigh)

    xdata_low = bins_low[1:]*.5 + bins_low[:-1]*.5
    xdata_high = bins_high[1:]*.5 + bins_high[:-1]*.5

    if before_and_after:
        before_low = []
        after_low = []
        before_high = []
        after_high = []

    common_mode_shifts = {}
    for i_pan in low_gain_pid:
        pixels = shot[i_pan][ is_low[i_pan] ]
        Npix = is_low[i_pan].sum()
        pix_hist = np.histogram( pixels, bins=bins_low, normed=True)[0]
        smoothed_hist = savgol_filter( pix_hist, window_length=window_length,
                                    mode='constant',polyorder=polyorder)
        pk_val = np.argmax(smoothed_hist)
        shift = xdata_low[pk_val ]
        common_mode_shifts[ (i_pan, 'low') ] = shift
        if plot_details:
            plt.figure()
            ax = plt.gca()
            ax.plot( xdata_low, pix_hist, '.')
            ax.plot( xdata_low, smoothed_hist, lw=2)
            ax.plot( xdata_low-shift, smoothed_hist, lw=2)
            ax.plot( shift, smoothed_hist[pk_val], 's', mfc=None, mec='Deeppink', mew=2 )
            ax.set_title("Panel has %d pixels, Shift amount = %.3f"%( Npix, shift))
            plt.show()
        if verbose:
            print("shifted panel %d by %.4f"% ( i_pan, shift))
        if before_and_after:
            before_low.append( pix_hist)
            pix_hist_shifted = np.histogram( pixels-shift, bins=bins_low, normed=True)[0]
            after_low.append( pix_hist_shifted)
    for i_pan in high_gain_pid:
        pixels = shot[i_pan][ is_high[i_pan] ]
        Npix = is_high[i_pan].sum()
        pix_hist = np.histogram( pixels, bins=bins_high, normed=True)[0]
        smoothed_hist = savgol_filter( pix_hist, window_length=window_length,mode='constant', polyorder=polyorder)
        pk_val=np.argmax(smoothed_hist)
        shift = xdata_high[pk_val]
        common_mode_shifts[ (i_pan, 'high') ] = shift
        if plot_details:
            plt.figure()
            ax = plt.gca()
            ax.plot( xdata_high, pix_hist, '.')
            ax.plot( xdata_high, smoothed_hist, lw=2)
            ax.plot( xdata_high-shift, smoothed_hist, lw=2)
            ax.plot( shift,  smoothed_hist[pk_val], 's', mfc=None, mec='Deeppink', mew=2 )
            ax.set_title("Panel has %d pixels, Shift amount = %.3f"%( Npix, shift))
            plt.show()
        if verbose:
            print("shifted panel %d by %.4f"%(i_pan,shift))
        if before_and_after:
            before_high.append( pix_hist)
            pix_hist_shifted = np.histogram( pixels-shift, bins=bins_high, normed=True)[0]
            after_high.append( pix_hist_shifted)

    for (i_pan,which), shift in common_mode_shifts.items():
        if which =='low':
            shot[i_pan][ is_low[i_pan]] = shot[i_pan][ is_low[i_pan]] - shift
        if which == 'high':
            shot[i_pan][ is_high[i_pan]] = shot[i_pan][ is_high[i_pan]] - shift
    if verbose:
        print("Mean shift: %.4f"%(np.mean(common_mode_shifts.values())))
    if plot_metric:
        print shot.shape, shot_.shape
        plt.figure()
        plt.plot( np.median( np.median(shot_,-1),-1), 'bo', ms=10, label='before')
        plt.plot( np.median( np.median(shot,-1),-1), 's', ms=10,color='Darkorange', label='after')
        plt.legend()
        plt.show()
    if inplace:
        return
    elif before_and_after:
        return xdata_low, before_low, after_low, xdata_high, before_high, after_high, shot
    else:
        return shot

# @profile
def main():
    from mask import mask_utils
    gain = np.load("/home/dermen/cxid9114_data/gain_mask.npy")
    mask1 = np.load("/home/dermen/cxid9114_data/corner_masks.npy")
    mask2 = np.load("/home/dermen/cxid9114_data/details_mask.npy")
    mask3 = mask_utils.mask_small_regions(gain) # masks regions too small to analyze
    mask = mask1*mask2*mask3

    Nshot = 11
    import psana
    ds = psana.DataSource("exp=cxid9114:run=62")
    det = psana.Detector("CxiDs2.0:Cspad.0")
    events = ds.events()
    dark = det.pedestals(62)
    shots = []
    for i in range( Nshot):
        print i
        ev = events.next()
        if ev is None:
            continue
        raw = det.raw(ev)
        if raw is None:
            continue
        shot = raw.astype(np.float32) - dark
        shots.append( shot)

    # shot 5 has lots of spots
    shot = shots[5].copy()
    shot_pppg_corr = pppg( shot, gain, mask=mask, low_x1=-5, low_x2=5, high_x1=-10, high_x2=10, Nlow=100, Nhigh=200,
                window_length=51, polyorder=3, plot_metric=True, verbose=True, before_and_after=False)

    # compare pppg to psana common mode to un-corrected case
    # when taking into account different gain levels...
    shot_psana_corr = shot.copy()
    det.common_mode_apply( 62, shot_psana_corr, (1,25,25,100,1))

    is_low = gain*mask
    is_high = (~gain)*mask
    meds_low = []
    meds_pppg_corr_low = []
    meds_psana_corr_low = []
    meds_high = []
    meds_pppg_corr_high = []
    meds_psana_corr_high = []

    for i_pan in range(32):
        if np.any( is_low[i_pan]):
            m = np.median( shot[i_pan][ is_low[i_pan]] )
            m_pppg = np.median( shot_pppg_corr[i_pan][ is_low[i_pan]] )
            m_psana = np.median( shot_psana_corr[i_pan][ is_low[i_pan]] )
            meds_low.append( m)
            meds_pppg_corr_low.append( m_pppg)
            meds_psana_corr_low.append( m_psana)
        if np.any( is_high[i_pan]):
            m = np.median( shot[i_pan][ is_high[i_pan]] )
            m_pppg = np.median( shot_pppg_corr[i_pan][ is_high[i_pan]] )
            m_psana = np.median( shot_psana_corr[i_pan][ is_high[i_pan]] )
            meds_high.append( m)
            meds_pppg_corr_high.append( m_pppg)
            meds_psana_corr_high.append( m_psana)

    fig, (ax1,ax2) = plt.subplots( nrows=2, ncols=1)
    ax1.plot( meds_low, 'o', color='b', label='unaltered', ms=10,mec=None)
    ax1.plot( meds_pppg_corr_low, 's', color='Darkorange', label='pppg', ms=10,mec=None)
    ax1.plot( meds_psana_corr_low, '<', color='Deeppink', label='psana', ms=10,mec=None)
    ax2.plot( meds_high, 'o', color='b', label='unaltered', ms=10,mec=None)
    ax2.plot( meds_pppg_corr_high, 's', color='Darkorange', label='pppg', ms=10,mec=None)
    ax2.plot( meds_psana_corr_high, '<', color='Deeppink', label='psana', ms=10,mec=None)
    ax1.legend(prop={"size":14})
    ax2.legend(prop={'size':14})
    ax1.set_title("low-gain regions",fontsize=14)
    ax2.set_title("high-gain regions",fontsize=14)
    plt.show()

if __name__ =="__main__":
    main()

