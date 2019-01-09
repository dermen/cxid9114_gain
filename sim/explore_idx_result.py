import os
from cxid9114 import utils
import dxtbx
import numpy as np
#import glob
#import json
#from dxtbx.model import DetectorFactory
#from dxtbx.model import CrystalFactory
#from dxtbx.model import BeamFactory
from cxid9114.sim.sim_crystal import PatternFactory

#expers = [json.load(open(f)) for f in glob.glob("exps/*json")]

# this was the image file used:
img_file = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
loader = dxtbx.load(img_file)
hit_ev_times = loader._h5_handle["event_times"][()]

# this is the spectrum data file
spec_data = "line_mn"
spec_file ="/Users/dermen/cxid9114/spec_trace/traces.62.h5"
get_spec = utils.GetSpectrum(spec_file=spec_file,
                             spec_file_data=spec_data)

# this is the pickle produced by index/ddi.py
# it contains crystals, reflections, and rmsd score
# and the outer-most dictionary keys are the shot indices
# corresponding to the position in the image hdf5 file
data_file = "/Users/dermen/cxid9114_gain/run62_idx_-2processed.pkl"
data = utils.open_flex(data_file)
hit_dset_idx = data.keys()  # the dials shot index (also the hdf5 dset index)

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
slcA = slice(124,149)
slcB = slice(740,779)
thresh = 2

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



# these indices in particular had low rmsd
#some_good_hits = hits_idx[:5] + [142, 143, 94,  130 , 60]
#some_good_hits = np.random.permutation(hits_idx)[:10] # lets explore 10
from IPython import embed
embed()

##########



# initialize the pattern simulato
Patts = PatternFactory()




from cxid9114.sim import scattering_factors
from cxid9114 import parameters

i2 = np.argmin( abs(scattering_factors.interp_energies \
                    - parameters.ENERGY_HIGH))
i1 = np.argmin( abs(scattering_factors.interp_energies \
                    - parameters.ENERGY_LOW))
fcalcs_at_en = {0:scattering_factors.fcalc_at_wavelen[i1], 1:scattering_factors.fcalc_at_wavelen[i2]}
energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]

#for h in some_good_hits:



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

output_imgs = []
output_refls = []
output_pref = "output"
for hit_idx in some_good_hits:
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
                            mosaic_spread=0.2,
                            mosaic_domains=3,
                            ret_sum=True)
    actual_img = loader.get_raw_data(hit_idx).as_numpy_array().astype(np.float32)
    refl = data[hit_idx]['refl']
    output_imgs.extend([sim_patt, actual_img])
    output_refls.extend([refl,refl])

utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)


from scitbx.matrix import col
from scitbx.matrix import sqr
x = col((1.,0.,0.))
y = col((0.,1.,0.))
z = col((0.,0.,1.))
xRot = x.axis_and_angle_as_r3_rotation_matrix
yRot = y.axis_and_angle_as_r3_rotation_matrix
zRot = z.axis_and_angle_as_r3_rotation_matrix
degs = linspace(-5.,5.,50)  # not sure here..
rot_series = [ (xRot(i, deg=True), yRot(j, deg=True)) for i in degs for j in degs]


output_imgs = []
output_refls = []
output_pref = "zRot_output"
hit_idx = some_good_hits[0]
for i in linspace(0,2,20):
    cryst = data[hit_idx]["crystals"][0]
    fracA = data[hit_idx]['fracA']
    fracB = data[hit_idx]['fracB']
    flux_per_en = [ fracA * 1e14, fracB*1e14]
    sim_patt = Patts.make_pattern2( crystal=cryst,
                            flux_per_en=flux_per_en,
                            energies_eV=energies,
                            fcalcs_at_energies=fcalcs_at_en,
                            mosaic_spread=0.2,
                            mosaic_domains=3,
                            ret_sum=True,
                            Op=zRot(i, deg=True))
    output_imgs.extend([sim_patt])
    output_refls.extend([refl])

utils.images_and_refls_to_simview(output_pref, output_imgs, output_refls)



