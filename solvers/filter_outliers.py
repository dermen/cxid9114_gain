
import numpy as np
import pandas
from scipy.optimize import curve_fit
from IPython import embed

from cxid9114.fit_utils import Gauss
from cxid9114.utils import is_outlier


def filter_outliers2(dhkl, Dall_ave, thresh=2.5):
    tops = [(25, 23, 1),
     (-12, -10, -15),
     (8, 4, 6),
     (20, 4, 3),
     (-16, -13, -14),
     (21, 17, 9),
     (20, 3, 6),
     (14, 8, 0),
     (-7, -2, -1),
     (16, 7, 8),
     (23, 2, 8),
     (-7, -3, -18),
     (-13, -4, -16),
     (10, 0, 2),
     (13, 8, 9),
     (9, 6, 10),
     (30, 8, 10),
     (34, 11, 3),
     (18, 7, 15),
     (32, 3, 3),
     (-3, -2, -7),
     (32, 2, 6),
     (12, 3, 4),
     (22, 8, 7),
     (-17, -4, -13),
     (-26, -19, -5),
     (34, 18, 3),
     (-13, -6, -13),
     (-17, -7, -10),
     (-17, -4, -3),
     (-35, -10, -1),
     (-34, -5, -1),
     (10, 7, 5),
     (24, 0, 4),
     (12, 7, 1),
     (-17, -1, -8),
     (28, 26, 0),
     (30, 15, 9),
     (18, 5, 1),
     (-22, -6, -14),
     (-23, -10, -14),
     (-16, -11, -6),
     (33, 7, 2),
     (19, 15, 14),
     (22, 16, 1),
     (-36, -11, -1),
     (18, 7, 14),
     (-28, -23, -4),
     (24, 11, 3),
     (-26, -23, -7),
     (-28, -8, -12),
     (13, 3, 8),
     (-9, -7, -11),
     (-17, -11, -11),
     (-10, -1, -7),
     (31, 4, 9),
     (-2, -1, -10),
     (22, 2, 9),
     (11, 5, 12),
     (-23, -6, -9),
     (10, 10, 0),
     (34, 6, 7),
     (10, 8, 4),
     (-22, -5, -9),
     (19, 13, 0),
     (36, 10, 6),
     (1, 1, 3),
     (-23, -18, -9),
     (31, 13, 8),
     (13, 4, 8),
     (-23, -4, -6),
     (26, 20, 10),
     (20, 18, 2),
     (29, 3, 6),
     (-31, -4, -4),
     (19, 1, 4),
     (16, 0, 0),
     (-26, -22, -3),
     (22, 8, 2),
     (16, 12, 5),
     (-19, -16, -1),
     (-4, -3, -6),
     (-11, -3, -12),
     (7, 7, 10),
     (-27, -4, -8),
     (-33, -1, -2),
     (22, 12, 3),
     (35, 4, 4),
     (26, 19, 4),
     (25, 3, 12),
     (-31, -6, -3),
     (-36, -3, -3),
     (-29, -22, -7),
     (-9, -5, -13),
     (13, 12, 14),
     (25, 14, 5),
     (-24, -14, -1),
     (-36, -5, -1),
     (27, 5, 1),
     (32, 13, 4)]

    h = tuple(dhkl[["h2", "k2", "l2"]].values[0])

    gain_guess = dhkl.lhs.mean() / Dall_ave

    # reflections where A channel scattering only is present
    dAnotB = dhkl.query("PA > 0 and PB == 0")
    I_AnotB = dAnotB.lhs / gain_guess / dAnotB.LA / dAnotB.PA * dAnotB.K

    # reflections where B channel scattering only present
    dBnotA = dhkl.query("PA == 0 and PB > 0")
    I_BnotA = dBnotA.lhs / gain_guess / dBnotA.LB / dBnotA.PB * dBnotA.K

    # reflections where both A and B channels present
    # here we assume half the scattering is A channel and the other half B channel, only
    # need to be approximate here as we are guessing the form factor value we will refine
    p = dhkl.query("PA > 0 and PB > 0")
    valsA = p.lhs / p.LA / p.PA * p.K / gain_guess / 2
    valsB = p.lhs / p.LB / p.PB * p.K / gain_guess / 2

    if h in tops:
        embed()

    return np.median(valsA), dhkl, 1, 1


def filter_outliers(dhkl, use_median=True, thresh=2.5, fit_gauss=True,
                    nsig=3, gain_key="gain"):

    # reflections where A channel scattering only is present
    dAnotB = dhkl.query("PA > 0 and PB == 0")
    I_AnotB = dAnotB.D / dAnotB[gain_key] / dAnotB.LA / dAnotB.PA * dAnotB.K

    # reflections where B channel scattering only present
    dBnotA = dhkl.query("PA == 0 and PB > 0")
    I_BnotA = dBnotA.D / dBnotA[gain_key] / dBnotA.LB / dBnotA.PB * dBnotA.K

    # reflections where both A and B channels present
    # here we assume half the scattering is A channel and the other half B channel, only
    # need to be approximate here as we are guessing the form factor value we will refine
    p = dhkl.query("PA > 0 and PB > 0")
    valsA = p.D/p.LA/p.PA*p.K / p[gain_key]/2
    valsB = p.D/p.LB/p.PB*p.K / p[gain_key]/2

    # combine all estimates of the form factor
    all_vals = np.hstack([I_AnotB, I_BnotA, valsA, valsB])
    all_vals2 = np.sqrt(all_vals)

    # remove outliers using median absolute deviation filter
    outliers = is_outlier(all_vals2, thresh)

    # now combine the inlier rows into a new dataframe
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
        best_val = np.mean(good_vals)**2

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

            _, muFit, sigFit = pFit
            W1 = abs(np.sqrt(I1) - muFit) / sigFit
            W2 = abs(np.sqrt(I2) - muFit) / sigFit
            W3 = abs(np.sqrt(I3) - muFit) / sigFit
            from IPython import embed
            embed()

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
    #dhkl_filt.weights /= dhkl_filt.weights.max()

    return best_val, dhkl_filt, pFit, cov

