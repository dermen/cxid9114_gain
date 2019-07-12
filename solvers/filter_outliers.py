
import numpy as np
import pandas
from scipy.optimize import curve_fit

from cxid9114.fit_utils import Gauss
from cxid9114.utils import is_outlier


def filter_outliers(dhkl, use_median=True, thresh=2.5, fit_gauss=True,
                    nsig=3):
    # reflections where A channel scattering only is present 
    dAnotB = dhkl.query("PA > 0 and PB == 0")
    I_AnotB = dAnotB.D / dAnotB.gain / dAnotB.LA / dAnotB.PA * dAnotB.K

    # reflections where B channel scattering only present
    dBnotA = dhkl.query("PA == 0 and PB > 0")
    I_BnotA = dBnotA.D / dBnotA.gain / dBnotA.LB / dBnotA.PB * dBnotA.K

    # reflections where both A and B channels present
    # here we assume half the scattering is A channel and the other half B channel, only
    # need to be approximate here as we are guessing the form factor value we will refine
    p = dhkl.query("PA > 0 and PB > 0")
    valsA = p.D/p.LA/p.PA*p.K / p.gain/2
    valsB = p.D/p.LB/p.PB*p.K / p.gain/2

    # combine all estimates of the form factor
    all_vals = np.hstack([I_AnotB, I_BnotA, valsA, valsB])
    all_vals2 = np.sqrt(all_vals)

    # remove outliers using median absolute deviation filter
    outliers = is_outlier(all_vals2, thresh)

    # now combine the not outlier rows into a new dataframe
    N_AnotB = len(I_AnotB)
    out_AnotB = outliers[:N_AnotB]

    N_BnotA = len(I_BnotA)
    out_BnotA = outliers[N_AnotB: N_AnotB + N_BnotA]

    n = N_AnotB + N_BnotA
    out_AandB_1 = outliers[n: n + len(valsA)]
    out_AandB_2 = outliers[n + len(valsA):]
    out_AandB = np.logical_or(out_AandB_1, out_AandB_2)


    # sanity check
    #n1 = out_AnotB.sum() + out_BnotA.sum() + out_AandB_1.sum() + out_AandB_2.sum()
    #n2 = outliers.sum()
    #assert (n1 == n2), "%d , %d" % (n1, n2)

    good_vals = all_vals2[~outliers]

    if use_median:
        best_val = np.median(good_vals)**2
    else:
        best_val = np.mean( good_vals)**2

    d1 = dAnotB.loc[~out_AnotB]
    I1 = I_AnotB[~out_AnotB].values

    d2 = dBnotA.loc[~out_BnotA]
    I2 = I_BnotA[~out_BnotA].values

    o3 = ~out_AandB
    d3 = p.loc[o3]
    I3 = valsA[o3].values*.5 + valsB[o3].values*.5

    dhkl_filt = pandas.concat([d1, d2, d3])
    dhkl_filt['Iestimate'] = np.concatenate([I1, I2, I3])

    if fit_gauss:
        # fit a Gaussian to good_vals

        mu = np.median(good_vals)
        sig = np.std(good_vals)
        bins = np.linspace(mu - nsig * sig, mu + nsig * sig, len(dhkl_filt) / 2)
        xdata = .5 * bins[1:] + .5 * bins[:-1]
        ydata, _ = np.histogram(good_vals, bins=bins)
        try:
            amp = ydata.max()
            pFit, cov = curve_fit(Gauss, xdata, ydata, p0=(amp, mu, sig))

            W1 = Gauss(np.sqrt(I1), *pFit)
            W2 = Gauss(np.sqrt(I2), *pFit)
            W3 = Gauss(np.sqrt(I3), *pFit)

        except (RuntimeError, TypeError, ValueError):
            pFit, cov = None, None
            W1 = np.zeros_like(I1)
            W2 = np.zeros_like(I2)
            W3 = np.zeros_like(I3)
    else:
        pFit = cov = None
        W1 = np.zeros_like(I1)
        W2 = np.zeros_like(I2)
        W3 = np.zeros_like(I3)

    dhkl_filt['weights'] = np.concatenate([W1, W2, W3])
    dhkl_filt.weights /= dhkl_filt.weights.max()

    return best_val, dhkl_filt, pFit, cov

