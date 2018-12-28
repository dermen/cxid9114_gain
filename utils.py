import numpy as np
import cPickle
import os
from dials.array_family import flex

def psana_geom_splitter(cspad, event=None, returned_units='mm'):
    """
    Splits the psana geometry from 32 panels to 64 panels
    because of the non-uniform pixel gap
    This creates a list of 64 origins, 64 slow-scan directions and 64 fast-scan directions
    though slow and fast scan directions are assumed parallel within
    manufacturing error
    :param cspad:  psana.Detector instantance for a CSPAD
    :param event: psana.Event instatance, can usually just leave it as None`
    :param returned_units: string of 'mm', 'um', or 'pixels'
    :return: PSF vectors, 64 long
    """
    geom = cspad.geometry(event)
    origin_64 = np.zeros((64, 3))
    FS_64 = np.zeros_like(origin_64)
    SS_64 = np.zeros_like(origin_64)

    origin_32, SS_32, FS_32 = map(np.array, zip(*geom.get_psf()))
    for i in range(32):
        # create the origins of each sub asic
        origin_A = origin_32[i]
        shift = 194. * 109.92 + (274.8 - 109.92) * 2.
        unit_f = FS_32[i] / np.linalg.norm(FS_32[i])
        origin_B = origin_A + unit_f * shift

        # save two sub-asics per each of the 32 actual asics
        idx_A = 2 * i
        idx_B = 2 * i + 1
        origin_64[idx_A] = origin_A
        origin_64[idx_B] = origin_B
        FS_64[idx_A] = FS_64[idx_B] = FS_32[i]
        SS_64[idx_A] = SS_64[idx_B] = SS_32[i]

    if returned_units == "mm":  # dials convention
        return origin_64 / 1000., SS_64 / 1000., FS_64 / 1000.,
    elif returned_units == "um":  # psgeom convention
        return origin_64, SS_64, FS_64
    elif returned_units == "pixels":  # crystfel convention
        return origin_64 / 109.92, SS_64 / 109.92, FS_64 / 109.92

def psana_data_to_aaron64_data(data):
    """
    :param data:  32 x 185 x 388 cspad data
    :return: 64 x 185 x 194 cspad data:w
    """
    return [sub_asic
            for split_asic in
                [(asic[:, :194], asic[:, 194:]) for asic in data]
            for sub_asic in split_asic]

def psana_mask_to_aaron64_mask(mask_32panels, pickle_name, force=False):
    """
    FormatXTCCspad divides CSPAD ASICS (panels) into 2 due to the gap that is not an
    integer multiple of 109.92 microns on each CSPAD ASIC,
    This does the same for masks that are made in the psana format.

    :param mask_32panels:  psana-style mask (False means "mask this pixel")
    :param pickle_name: string , where to store the flex mask to use for FormatXTCCspad
    :param force: bool, whether to force an overwrite of pickle_name
    :return: None
    """
    flex_mask = []
    for panelmask in mask_32panels:
        flex_mask += [panelmask[:, :194], panelmask[:, 194:]]

    # or this one liner for interactive ease
    # flex_mask = [ l for sl in [(m[:, :194], m[:, 194:]) for m in mask_32panels] for l in sl]

    # this just maps to flex.bool but also ensures arrays are contiguous as
    # that is a requirement for flex (I think)
    flex_mask = tuple(map(lambda x: flex.bool(np.ascontiguousarray(x)), flex_mask))
    if not force:
        if os.path.exists(pickle_name):
            raise OSError("The file %s exists, use force=True to overwrite" % pickle_name)
    with open(pickle_name, "w") as out:
        cPickle.dump(flex_mask, out)


def smooth(x, beta=10.0, window_size=11):
    """
    https://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html
    
    Apply a Kaiser window smoothing convolution.

    Parameters
    ----------
    x : ndarray, float
        The array to smooth.

    Optional Parameters
    -------------------
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.

    Returns
    -------
    smoothed : ndarray, float
        A smoothed version of `x`.
    """

    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # apply the smoothing function
    s = np.r_[x[window_size - 1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')

    # remove the extra array length convolve adds
    b = int((window_size - 1) / 2)
    smoothed = y[b:len(y) - b]

    return smoothed


def is_outlier(points, thresh=3.5):
    """
    http://stackoverflow.com/a/22357811/2077270

    Returns a boolean array with True if points are outliers and False
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def write_cxi_peaks(h5, peaks_path, pkX, pkY, pkI):
    """
    adds peaks to an hdf5 file in the CXI format
    :param h5:  h5 file handle
    :param peaks_path:  peaks group name
    :param pkX: X-coordinate of  peaks (list of lists)
    :param pkY: Y-coordinate of peaks (list of lists like pkX)
    :param pkI: Intensity of peaks (list of float
    """
    npeaks = np.array([len(x) for x in pkX])
    max_n = max(npeaks)
    Nimg = len(pkX)

    data_x = np.zeros((Nimg, max_n), dtype=np.float32)
    data_y = np.zeros_like(data_x)
    data_I = np.zeros_like(data_x)

    for i in xrange(Nimg):
        n = npeaks[i]
        data_x[i, :n] = pkX[i]
        data_y[i, :n] = pkY[i]
        data_I[i, :n] = pkI[i]

    peaks = h5.create_group(peaks_path)
    peaks.create_dataset('nPeaks', data=npeaks)
    peaks.create_dataset('peakXPosRaw', data=data_x)
    peaks.create_dataset('peakYPosRaw', data=data_y)
    peaks.create_dataset('peakTotalIntensity', data=data_I)
