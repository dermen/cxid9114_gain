from scipy.special import erf
import lmfit
import numpy as np
import pylab as plt

LOW_GAIN_GAUSS_PARAMS = lmfit.Parameters()
LOW_GAIN_GAUSS_PARAMS.add('wid0', value= 2.2, min=1)
LOW_GAIN_GAUSS_PARAMS.add('amp0', value= 0.25 , min=0)
LOW_GAIN_GAUSS_PARAMS.add('mu0', value= 0, min=-0.00001, max=0.00001)
LOW_GAIN_GAUSS_PARAMS.add('wid1', value= 3.,min=0, max=3)
LOW_GAIN_GAUSS_PARAMS.add('amp1', value=0.01 , min=0)
LOW_GAIN_GAUSS_PARAMS.add('mu1', value= 4.2 , min=3.5, max=5 )
#LOW_GAIN_GAUSS_PARAMS.add('alpha1', value=-2.2, max=0.2, min=-8)

HIGH_GAIN_GAUSS_PARAMS = lmfit.Parameters()
HIGH_GAIN_GAUSS_PARAMS.add('wid0', value= 3.,)
HIGH_GAIN_GAUSS_PARAMS.add('amp0', value= 0.1 , min=0)
HIGH_GAIN_GAUSS_PARAMS.add('mu0', value= 0 ,min=-0.0001, max=0.0001 )
# HIGH_GAIN_GAUSS_PARAMS.add('alpha0', value=2 ,  )
HIGH_GAIN_GAUSS_PARAMS.add('wid1', value= 1.5, min=0,max=2)
HIGH_GAIN_GAUSS_PARAMS.add('amp1', value= 0.005 , min=0)
HIGH_GAIN_GAUSS_PARAMS.add('mu1', value= 28, min=24, max=32)
#HIGH_GAIN_GAUSS_PARAMS.add('alpha1', value=-2)

def Gauss(x,amp,mu,wid):
    """returns a Gaussian"""
    return amp*np.exp( \
        -((x - mu)/wid)**2)

def Cumu(x, mu,alpha):
    """returns a cummulative distribution"""
    return 0.5*( 1 + erf( (alpha*(x-mu))  /np.sqrt(2)) )

def skew_gauss(x,amp,mu,wid,alpha):
    """returns a skewed normal dist"""
    return Gauss(x, amp, mu,wid)*Cumu(x,mu,alpha)*2

def gauss_standard(params, xdata, ydata):
    """
    This is for fitting the 0th order peak
    """
    amp = params['amp'].value
    wid = params['wid' ].value
    mu= params['mu' ].value
    gauss_model = Gauss( xdata, amp, mu, wid)
    return gauss_model-ydata

def gauss_and_skewgauss( params, xdata, ydata):
    """ residual function for fitting sum of two skewed Gaussians"""
    #   fit the 0-photon peak Gaussian
    amp0 = params['amp0'].value
    wid0 = params['wid0' ].value
    mu0 = params['mu0' ].value
    if 'alpha0' in params.keys():
        alpha0 = params['alpha0'].value
    else:
        alpha0 = 0
    gauss_model0 = skew_gauss( xdata, amp0, mu0, wid0, alpha=alpha0)

    amp1 = params['amp1'].value
    wid1 = params['wid1' ].value
    mu1 = params['mu1' ].value
    if 'alpha1' in params.keys():
        alpha1 = params['alpha1'].value
    else:
        alpha1 = 0
    gauss_model1 = skew_gauss( xdata, amp1, mu1, wid1, alpha=alpha1)

    return gauss_model0 + gauss_model1 - ydata


def fit_low_gain_dist(xdata, ydata, plot=False):
    result = lmfit.minimize(gauss_and_skewgauss, LOW_GAIN_GAUSS_PARAMS,
                            args=(xdata, ydata ))
    rp = result.params

    amp0 = rp['amp0'].value
    mu0 = rp['mu0'].value
    wid0 = rp['wid0'].value
    if 'alpha0' in rp.keys():
        alpha0 = rp['alpha0'].value
    else:
        alpha0 = 0
    amp1 = rp['amp1'].value
    mu1 = rp['mu1'].value
    wid1 = rp['wid1'].value
    if 'alpha1' in rp.keys():
        alpha1 = rp['alpha1'].value
    else:
        alpha1 = 0

    gauss0 = skew_gauss(xdata, amp0,mu0,wid0,alpha0)
    gauss1 = skew_gauss(xdata, amp1,mu1,wid1,alpha1)

    if plot:
        plt.figure()
        plt.gca().tick_params(labelsize=14)
        plt.xlabel("ADU (dark subtracted)", fontsize=14)
        plt.ylabel("bincount", fontsize=14)
        plt.plot( xdata, ydata, '.', label="data")
        plt.gca().set_yscale("log")
        plt.ylim(0.0001,.3)
        plt.xlim(-20,50)
        plt.plot( xdata, gauss1, label="fit to 1-photon peak")
        plt.plot( xdata, gauss0, label="fit to 0-photon peak")
        plt.plot( xdata, gauss0+gauss1, label="fit")
        plt.legend( prop={'size':13})
        plt.draw()
        plt.pause(0.1)

    return gauss0,gauss1, result

def fit_high_gain_dist(xdata, ydata, plot=False):

    result = lmfit.minimize(gauss_and_skewgauss, HIGH_GAIN_GAUSS_PARAMS,
                            args=(xdata, ydata ))
    rp = result.params

    amp0 = rp['amp0'].value
    mu0 = rp['mu0'].value
    wid0 = rp['wid0'].value
    if 'alpha0' in rp.keys():
        alpha0 = rp['alpha0'].value
    else:
        alpha0 = 0
    amp1 = rp['amp1'].value
    mu1 = rp['mu1'].value
    wid1 = rp['wid1'].value
    if 'alpha1' in rp.keys():
        alpha1 = rp['alpha1'].value
    else:
        alpha1 = 0

    gauss0 = skew_gauss(xdata, amp0,mu0,wid0,alpha0)
    gauss1 = skew_gauss(xdata, amp1,mu1,wid1,alpha1)

    if plot:
        plt.figure()
        plt.gca().tick_params(labelsize=14)
        plt.xlabel("ADU (dark subtracted)", fontsize=14)
        plt.ylabel("bincount", fontsize=14)
        plt.plot( xdata, ydata, '.', label="data")
        plt.gca().set_yscale("log")
        plt.ylim(0.0001,.3)
        plt.xlim(-20,50)
        plt.plot( xdata, gauss1, label="fit to 1-photon peak")
        plt.plot( xdata, gauss0, label="fit to 0-photon peak")
        plt.plot( xdata, gauss0+gauss1, label="fit")
        plt.legend( prop={'size':13})
        plt.draw()
        plt.pause(0.1)

    return gauss0,gauss1, result
