"""
This script takes the indexing solutions, then
appends the spectrum information to each
indexing solution.

This information is needed for the refinement
script which is based on the simualation
"""
import sys
import numpy as np
from scipy.ndimage import maximum_filter1d

from cxid9114 import utils
import dxtbx


if __name__ == "__main__":
    dials_img_file = sys.argv[1]
    spec_file = sys.argv[2]
    data_file = sys.argv[3]
    output_name = sys.argv[4]

    max_Filt_cut = 20
    thresh = 0.8
    # dials_img_file = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
    # spec_file ="/Users/dermen/cxid9114/spec_trace/traces.62.h5"
    # data_file = "/Users/dermen/cxid9114_gain/run62_idx_-2processed.pkl"

    print "Loading image getter"
    loader = dxtbx.load(dials_img_file)
    hit_ev_times = loader._h5_handle["event_times"][()]

    print "loading spectrum analyzer"
    # this is the spectrum data file
    spec_data = "line_mn"

    get_spec = utils.GetSpectrum(spec_file=spec_file,
                                 spec_file_data=spec_data)

    print "Loading indexing results data"
    # this is the pickle produced by index/ddi.py
    # it contains crystals, reflections, and rmsd score
    # and the outer-most dictionary keys are the shot indices
    # corresponding to the position in the image hdf5 file
    data = utils.open_flex(data_file)
    Ndata = len(data)
    hit_dset_idx = data.keys()  # the dials shot index (also the hdf5 dset index)

    print "Removing bad data"
    # iterate through the indexed events, and
    # delete events where the spectrum is None
    for i in hit_dset_idx:
        ev_t = hit_ev_times[i]
        spec = get_spec.get_spec(ev_t)
        data[i]["spectrum"] = spec

    # print out some useful info, maximum value in spectrum
    slcA = slice(124,149)  # low energy
    slcB = slice(740,779)  # high energy

    print "Analyzing spectra"
    some_good_hits = []
    hits_idx = data.keys()
    for h in hits_idx:
        spec = data[h]["spectrum"]
        rmsd_score = data[h]['best']
        if spec is None:
            data[h]['fracA'] = 0.5  # not sure what to do here...
            data[h]['fracB'] = 0.5
            data[h]['can_analyze'] = False
            print "Shot %d has no spectrum, setting frac =0.5" % h
        else:
            # find the good spectrums
            spec_filt = maximum_filter1d(spec, max_Filt_cut)
            data[h]["spec_filt"] = spec_filt

            spec_bkgrnd = np.median(spec_filt)
            spec_filt -= spec_bkgrnd

            specA = spec_filt[slcA]
            specB = spec_filt[slcB]
            sigA = specA.max()
            sigB = specB.max()
            data[h]["sigA"] = sigA
            data[h]["sigB"] = sigB

            can_analyze_spec = sigA > thresh or sigB > thresh

            if can_analyze_spec:
                if sigA < 0:
                    sigA = 0.0001
                if sigB < 0:
                    sigB = 0.0001
                fracA,fracB = sigA / (sigA + sigB), sigB / (sigA + sigB)
                data[h]["fracA"] = fracA
                data[h]["fracB"] = fracB
                data[h]['can_analyze'] = True
                print "Shot %d  has 2 color fractions %f" % (h, fracA)
            else:
                data[h]['fracA'] = 0.5  # NOTE: not sure what to do here..
                data[h]['fracB'] = 0.5
                data[h]['can_analyze'] = False

    utils.save_flex(data, output_name)
    N_has_spec = np.sum( [1 for k in data if data[k]['spectrum'] is not None])
    N_weak_spec = np.sum( [1 for k in data if not data[k]['can_analyze']
                          and data[k]['spectrum'] is not None])

    print( "%d / %d hits had spectrum data!" % (N_has_spec, Ndata ) )
    print( "%d / %d hits had spectrum data but it was too weak!" % (N_weak_spec, N_has_spec) )
    print("Saved pickle %s. Done. " % output_name)
