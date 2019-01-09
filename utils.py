import bisect
import os
import cPickle
import numpy as np
import h5py
from dials.array_family import flex
from cxid9114.spots import count_spots, spot_utils

try:
    import psana
    has_psana = True
except ImportError:
    has_psana=False


def open_flex(filename):
    """unpickle the flex file which requires flex import"""
    with open(filename, "r") as f:
        data = cPickle.load(f)
    return data

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
    import numpy as np
    npeaks = np.array([len(x) for x in pkX])
    max_n = max(npeaks)
    Nimg = len(pkX)

    data_x = np.zeros((Nimg, max_n), dtype=np.float32)
    data_I = np.zeros_like(data_x)
    data_y = np.zeros_like(data_x)

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


def make_event_time(sec, nanosec, fid):
    if not has_psana:
        print("No psana")
        return
    time = int((sec<<32)|nanosec)
    et = psana.EventTime(time, fid)
    return time, et


class GetSpectrum:
    """
    The spectrum data is processed separately and stored in
    hdf5 files where there is one spectrum per event time

    Then, when we process the CSPAD images we keep a copy of each
    CSPAD images event time, and this class is designed to retrieve
    the spectrum from the hdf5 file for a given event time
    """
    def __init__(self,
               spec_file="/home/dermen/cxid9114/spec_trace/traces.62.h5",
               spec_file_times="event_time",
               spec_file_data="line_mn",
               spec_is_1d = True):
        """
        :param spec_file:  path to the spectrum file which contans data and event times
        :param spec_file_times: dataset path of the times in the hdf5 file
        :param spec_file_data: dataset path of the spectrum data in the hdf5 file
        :param spec_is_1d: is the spectrum data 1d? If not process it as if it were 2d
        """
        self._f = h5py.File(spec_file, 'r')
        times = self._f[spec_file_times][()]
        self.spec_traces = self._f[spec_file_data]
        self.order = times.argsort()
        self.spec_sorted_times = times[self.order]
        self.spec_is_1d = spec_is_1d

    def get_spec(self, time):
        """
        Retrieve the spectrum!
        :param time: psana event time combo of sec and nanosec
        :return: the spectrum data if found, else None
        """
        if time not in self.spec_sorted_times:
            print "No spectrum for given time %d" % time
            return None
        sorted_pos = bisect.bisect_left(self.spec_sorted_times, time)
        trace_idx = self.order[sorted_pos]
        data = self.spec_traces[trace_idx]

        if self.spec_is_1d:
            return data
        #else:
        #    return self.project_fee_img(data)

def images_and_refls_to_simview(prefix, imgs, refls):

    refls_concat = spot_utils.combine_refls(refls)
    refl_info = count_spots.group_refl_by_shotID(refls_concat)
    refl_shotIds = refl_info.keys()
    Nrefl = len( refl_shotIds)
    Nimg = len( imgs)
    assert(Nimg==Nrefl)
    assert( all([ i in range(Nrefl) for i in refl_shotIds]))

    with open("%s_strong.pkl" % prefix, "w") as strong_f:
        cPickle.dump(refls_concat, strong_f)
        print "Wrote %s" % strong_f.name

    with h5py.File( "%s.h5" % prefix, "w") as img_f:
        for i_img in range(Nimg):
            if imgs[i].dtype != np.float32:
                imgs[i] = imgs[i].astype(np.float32)
        img_f.create_dataset("simulated_d9114_images",
                             data=imgs)
        print "Wrote %s" % img_f.filename
