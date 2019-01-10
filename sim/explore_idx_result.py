import os
from cxid9114 import utils
from cxid9114.spots import spot_utils
from cxid9114.sim import sim_utils
import dxtbx
import numpy as np
from scitbx.matrix import col
#import glob
#import json
#from dxtbx.model import DetectorFactory
#from dxtbx.model import CrystalFactory
#from dxtbx.model import BeamFactory
from cxid9114.sim.sim_crystal import PatternFactory
from cxid9114.sim import scattering_factors
from cxid9114 import parameters
from copy import deepcopy

make_output = False
Nout = 1
test_zRot = False
spec_compare = False
XYscan = False
XYscan_multi = False
n_jobs = 6
#expers = [json.load(open(f)) for f in glob.glob("exps/*json")]

print "Loading image getter"
# this was the image file used:
img_file = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
loader = dxtbx.load(img_file)
hit_ev_times = loader._h5_handle["event_times"][()]

print "loading spectrum analyzer"
# this is the spectrum data file
spec_data = "line_mn"
spec_file ="/Users/dermen/cxid9114/spec_trace/traces.62.h5"
get_spec = utils.GetSpectrum(spec_file=spec_file,
                             spec_file_data=spec_data)

print "Loading index results data"
# this is the pickle produced by index/ddi.py
# it contains crystals, reflections, and rmsd score
# and the outer-most dictionary keys are the shot indices
# corresponding to the position in the image hdf5 file
data_file = "/Users/dermen/cxid9114_gain/run62_idx_-2processed.pkl"
data = utils.open_flex(data_file)
hit_dset_idx = data.keys()  # the dials shot index (also the hdf5 dset index)

print "Removing bad data"
# iterate through the indexed events, and
# delete events where the spectrum is None
for i in hit_dset_idx:
    ev_t = hit_ev_times[i]
    spec = get_spec.get_spec(ev_t)
    if spec is None:
        _ = data.pop(i)
        print "No spec for shot %d" % i
    else:
        data[i]['spectrum'] = spec

# print out some useful info, maximum value in spectrum
from scipy.ndimage import maximum_filter1d
max_Filt_cut = 20
slcA = slice(124,149)  # low energy
slcB = slice(740,779)  # high energy
thresh = 2

print "Analyzing spectra"
some_good_hits = []
hits_idx = data.keys()
for h in hits_idx:
    spec = data[h]["spectrum"]
    rmsd_score = data[h]['best']

    # find the good spectrums
    spec_filt = maximum_filter1d(spec, max_Filt_cut)
    specA = spec_filt[slcA]
    specB = spec_filt[slcB]
    spec_bkgrnd = np.median(spec_filt)

    sigA = specA.max() - spec_bkgrnd
    sigB = specB.max() - spec_bkgrnd

    if spec_filt[slcA].max() > thresh or spec_filt[slcB].max() > thresh:
        some_good_hits.append(h)
        sigA = specA.max() - spec_bkgrnd
        sigB = specB.max() - spec_bkgrnd

    data[h]["spec_filt"] = spec_filt
    data[h]["sigA"] = sigA
    data[h]["sigB"] = sigB
    if sigA < 0:
        sigA = 0.0001
    if sigB < 0:
        sigB = 0.0001
    fracA,fracB = sigA / (sigA + sigB), sigB / (sigA + sigB)
    data[h]["fracA"] = fracA
    data[h]["fracB"] = fracB
    # ideally max(spec) has a high value
    # above 0 and best is small between 0 and 3
    print h, rmsd_score, max( spec)

# initialize the pattern simulato
Patts = PatternFactory()
Patts.adjust_mosaicity(mosaic_domains=2, mosaic_spread=0.05)

i2 = np.argmin( abs(scattering_factors.interp_energies \
                    - parameters.ENERGY_HIGH))
i1 = np.argmin( abs(scattering_factors.interp_energies \
                    - parameters.ENERGY_LOW))
fcalcs_at_en = {0:scattering_factors.fcalc_at_wavelen[i1], 1:scattering_factors.fcalc_at_wavelen[i2]}
energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]
sim_utils.save_fcalc_file(energies, fcalcs_at_en, filename="fcalc_slim.pkl")


# for each hit we will simulate
# the pattern given that hits
# spectrum
# we will make an image file and reflection table
# such that it can be loaded into the dials
# image viewer the order of the imageset will be
#
# 0. simulated_image,
# 1. corresponding data image
# 2. next simulated image..
# 3. ..
#
# so we can compare original with simulated indexed visually

if make_output:
    output_imgs = []
    output_refls = []
    output_pref = "output"
    for hit_idx in some_good_hits[:Nout]:
        #sim_img = Patts.make_pattern(
        #    crystal=data[hit_idx]['crystals'][0],
        #    spectrum=data[hit_idx]['spectrum'],
        #    show_spectrum=False)

        cryst = data[hit_idx]["crystals"][0]
        fracA = data[hit_idx]['fracA']
        fracB = data[hit_idx]['fracB']
        flux_per_en = [ fracA * 1e14, fracB*1e14]

        sim_patt = Patts.make_pattern2( crystal=cryst,
                                flux_per_en=flux_per_en,
                                energies_eV=energies,
                                fcalcs_at_energies=fcalcs_at_en,
                                mosaic_spread=None,
                                mosaic_domains=None,
                                ret_sum=True)
        actual_img = loader.get_raw_data(hit_idx).as_numpy_array().astype(np.float32)
        refl = data[hit_idx]['refl']
        output_imgs.extend([sim_patt, actual_img])
        output_refls.extend([refl,refl])

    utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)
    #os.system("dials.image_viewer %s %s" % \
    #          (output_pref+".h5", output_pref+"_strong.pkl"))

from scitbx.matrix import sqr
x = col((1.,0.,0.))
y = col((0.,1.,0.))
z = col((0.,0.,1.))
xRot = x.axis_and_angle_as_r3_rotation_matrix
yRot = y.axis_and_angle_as_r3_rotation_matrix
zRot = z.axis_and_angle_as_r3_rotation_matrix
degs = np.linspace(-5.,5.,50)  # not sure here..
degs = np.arange( -0.2, 0.2, 0.025)  # still not sure about ranges here.
degs = np.arange( -0.4, 0.4, 0.025)  # feeling more confident about this range
rotXY_series = [ (xRot(i, deg=True), yRot(j, deg=True)) for i in degs for j in degs]


#
# TEST rotation about beam!
#
if test_zRot:
    output_imgs = []
    output_refls = []
    output_pref = "zRot_output"
    hit_idx = some_good_hits[0]
    for i in np.linspace(0,2,20):
        cryst = data[hit_idx]["crystals"][0]
        fracA = data[hit_idx]['fracA']
        fracB = data[hit_idx]['fracB']
        refl = data[hit_idx]["refl"]
        flux_per_en = [ fracA * 1e12, fracB*1e12]
        sim_patt = Patts.make_pattern2( crystal=deepcopy(cryst),
                                flux_per_en=flux_per_en,
                                energies_eV=energies,
                                fcalcs_at_energies=fcalcs_at_en,
                                mosaic_spread=None,
                                mosaic_domains=None,
                                ret_sum=True,
                                Op=zRot(i, deg=True))
        output_imgs.extend([sim_patt])
        output_refls.extend([refl])

    utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)

#
# TEST Full spectrum vs simple two color
#
if spec_compare:

    output_imgs = []
    output_refls = []
    output_pref = "spec_compare"
    for i in range(0,1):
        hit_idx = some_good_hits[i]
        cryst = data[hit_idx]["crystals"][0]
        refl = data[hit_idx]["refl"]
        fracA = data[hit_idx]['fracA']
        fracB = data[hit_idx]['fracB']
        flux_per_en = [ fracA * 1e12, fracB*1e12]

        sim_patt = Patts.make_pattern2( crystal=cryst,
                                        flux_per_en=flux_per_en,
                                        energies_eV=energies,
                                        fcalcs_at_energies=fcalcs_at_en,
                                        mosaic_spread=None,
                                        mosaic_domains=None,
                                        ret_sum=True,
                                        Op=None)

        spec =  deepcopy(data[hit_idx]["spectrum"])
        spec -= np.median(spec)
        spec[spec < 1 ] = 0
        sim_patt2 = Patts.make_pattern(crystal=cryst,
                                       spectrum=spec)

        spec_filt = deepcopy(data[hit_idx]["spec_filt"])
        spec_filt -= np.median( spec_filt)
        spec_filt[ spec_filt < 0.5] = 0
        sim_patt3 = Patts.make_pattern(crystal=cryst,
                                       spectrum=spec_filt)


        output_imgs.extend([sim_patt, sim_patt2, sim_patt3])
        output_refls.extend([refl]*3)

    utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)


#
# XY scan
#

if XYscan:
    output_imgs =[]
    output_refls = []
    hit_idx = some_good_hits[0]
    cryst = data[hit_idx]["crystals"][0]
    refl = data[hit_idx]["refl"]
    fracA = data[hit_idx]['fracA']
    fracB = data[hit_idx]['fracB']
    flux_per_en = [fracA * 1e14, fracB * 1e14]
    output_pref = "xyscan_xtal_fine_%d" % hit_idx
    for i_rot, (rX,rY) in enumerate(rotXY_series):
        sim_patt = Patts.make_pattern2( crystal=deepcopy(cryst),
                                        flux_per_en=flux_per_en,
                                        energies_eV=energies,
                                        fcalcs_at_energies=fcalcs_at_en,
                                        mosaic_spread=None,
                                        mosaic_domains=None,
                                        ret_sum=True,
                                        Op=rX*rY)
        output_imgs.append( sim_patt)
        output_refls.append( refl)
        print "Rot %d / %d" % (i_rot+1, len(rotXY_series))
    utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)


def save_results(data_img, crystal, refl, fcalc_file, Xang, Yang, filename):
    x = col((1,0,0))
    y = col((0,1,0))
    rX = x.axis_and_angle_as_r3_rotation_matrix(Xang, deg=True)
    rY = y.axis_and_angle_as_r3_rotation_matrix(Yang, deg=True)

    energies, fcalc_at_en = sim_utils.load_fcalc_file(fcalc_file)
    Patts = PatternFactory()
    Patts.adjust_mosaicity(2,0.05)
    #flux_per_en = [fracA*Patts.SIM2.flux, fracB*Patts.SIM2.flux]
    flux_per_en = [fracA*1e14, fracB*1e14]

    sim_patt = Patts.make_pattern2(crystal=deepcopy(crystal),
                               flux_per_en=flux_per_en,
                               energies_eV=energies,
                               fcalcs_at_energies=fcalcs_at_en,
                               mosaic_spread=None,
                               mosaic_domains=None,
                               ret_sum=True,
                               Op=None)
    sim_patt_R = Patts.make_pattern2(crystal=deepcopy(crystal),
                                     flux_per_en=flux_per_en,
                                     energies_eV=energies,
                                     fcalcs_at_energies=fcalcs_at_en,
                                     mosaic_spread=None,
                                     mosaic_domains=None,
                                     ret_sum=True,
                                     Op=rX * rY)
    refls = [refl]*3
    imgs = [sim_patt, data_img, sim_patt_R]
    utils.images_and_refls_to_simview(filename, imgs, refls)


def xyscan(crystal_file, fcalcs_file, fracA, fracB, refl_file, rotxy_series):
    """

    :param crystal_file:
    :param energies:
    :param fcalcs_at_en:
    :param fracA:
    :param fracB:
    :param refl_file:
    :param rotxy_series:
    :return:
    """
    crystal = utils.open_flex(crystal_file)
    strong_spots = utils.open_flex(refl_file)
    energies, fcalcs_at_en = sim_utils.load_fcalc_file(fcalcs_file)
    Patts = PatternFactory()
    Patts.adjust_mosaicity(2, 0.05)  # defaults
    flux_per_en = [fracA*1e14, fracB*1e14]

    img_size = Patts.detector.to_dict()['panels'][0]['image_size']
    found_spot_mask = spot_utils.strong_spot_mask(strong_spots, img_size)

    overlaps = []
    for rX, rY in rotxy_series:
        sim_patt = Patts.make_pattern2(crystal=deepcopy(crystal),
                                       flux_per_en=flux_per_en,
                                       energies_eV=energies,
                                       fcalcs_at_energies=fcalcs_at_en,
                                       mosaic_spread=None,
                                       mosaic_domains=None,
                                       ret_sum=True,
                                       Op=rX * rY)
        sim_sig_mask = sim_patt > 0
        overlaps.append( sum(sim_sig_mask * found_spot_mask))

    return overlaps

def xyscan_multi(crystal_file, fcalcs_file, fracA, fracB,
                 strong_file, rotxy_series, n_jobs):
    """

    :param crystal_file:
    :param energies:
    :param fcalcs_file:
    :param fracA:
    :param fracB:
    :param strong_file:
    :param rotxy_series:
    :param n_jobs:
    :return:
    """

    from joblib import Parallel, delayed

    rotxy_split = np.array_split(rotxy_series, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(xyscan)\
                (crystal_file, fcalcs_file, fracA, fracB, strong_file, rotxy_split[jid]) \
                for jid in range(n_jobs))

    return np.concatenate(results)

from IPython import embed
embed()