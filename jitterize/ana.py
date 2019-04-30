from cxid9114 import utils
from cxid9114.spots import spot_utils

from cxid9114 import parameters
from copy import deepcopy
from cxid9114.index.ddi import params as mad_index_params
from cxi_xdr_xes.two_color.two_color_indexer \
    import indexer_two_color

index = False
shot_idx = 924
PID = 23
D = utils.open_flex("ref1_det.pkl")
B = utils.open_flex("ref3_beam.pkl")
BeamA = deepcopy(B)
BeamB = deepcopy(B)
mask = np.load("mask64.npz")["mask"] 
BeamA.set_wavelength(parameters.WAVELEN_LOW)
BeamB.set_wavelength(parameters.WAVELEN_HIGH)

img_data = utils.open_flex('some_imgs.pkl')

dblock = utils.datablock_from_numpyarrays( 
    [i.as_numpy_array() for i in img_data["img%d" % shot_idx]], 
    detector=D, beam=BeamA, mask=mask )
dump = utils.open_flex("dump_%d.pkl" % shot_idx)
refls = dump['refls_strong']
spot_utils.as_single_shot_reflections( refls)

iset = dblock.extract_imagesets()[0]

if index:
    orientAB = indexer_two_color(
        reflections=refls,
        imagesets=[iset],
        params=mad_index_params)
    orientAB.index()

    C = orientAB.refined_experiments.crystals()[0]
    utils.save_flex({"C":C}, "C%d.pkl" % shot_idx)
    from dxtbx.model.experiment_list import Experiment, ExperimentList
    e = Experiment()
    e.detector = D
    e.beam = B
    e.crystal = C
    e.imageset = iset
    el = ExperimentList()
    el.append(e)
    orientAB.export_as_json(el, "exp%d.json" % shot_idx)

##############
#############
##############
##############
##############
##############
##############
#############
#############
#############
#############
#############
import sys
from cxid9114 import utils
from dxtbx.model.experiment_list import  ExperimentListFactory
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import numpy as np
from cxid9114.refine import jitter_refine
import scipy.ndimage
from cxid9114.refine import metrics


ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [5000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e14, 1e14]  # fluxes of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
crystalAB = utils.open_flex("C%d.pkl" % shot_idx)['C']

detector = D
beamA = BeamA
beamB = BeamB

refls_strong = refls
reflsPP = spot_utils.refls_by_panelname(refls_strong)
pids = [i for i in reflsPP if len(reflsPP[i]) > 0]  # refine on these panels only
pan_imgs = {pid: 
            iset.get_raw_data(0)[pid].as_numpy_array()
            for pid in pids}

# helper wrapper for U-matrix grid search based refinement
# `scanZ = ...` can also be passed as an argument, to jitter rotation
# about the Z (beam) axis
jitt_out = jitter_refine.jitter_panels(
    panel_ids=pan_imgs.keys(), 
       crystal=crystalAB,
       refls=refls_strong,
       det=detector,
       beam=iset.get_beam(0),
       FF=FF,
       en=ENERGIES,
       data_imgs=pan_imgs.values(),
       flux=FLUX,
       ret_best=False,
       Ncells_abc=(5,5, 5),
       oversample=1,
       Gauss=True,
       verbose=0,
       mos_dom=1,
       mos_spread=0.0,
       scanX=np.arange(-.4, .4, .05), 
       scanY=np.arange(-.4, .4, .05))

# select the refined matrix based on overlap superposition
# overlap is a metric used in the JitterFactory (wrapped into jitter_panels)
# which checks agreement between data panels and simulated panels
overlap = np.sum([jitt_out[pid]['overlaps'] for pid in jitt_out], axis=0)
max_pos = np.argmax(overlap)
optA = jitt_out[jitt_out.keys()[0]]["A_seq"][
    max_pos]  # just grab the first A_seq cause same sequence is tested on all panels

optCrystal = deepcopy(crystalAB)
optCrystal.set_A(optA)
utils.save_pickle( 'optC_%d.pkl' % shot_idx, optCrystal)

simsAB1 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[1], Gauss=False, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB50 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[50], Gauss=False, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=False, oversample=2,
    Ncells_abc=(5, 5, 5), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB2 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=False, oversample=3,
    Ncells_abc=(15, 15, 15), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB3 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=False, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0, verbose=0)

simsAB4 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB5 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=3,
    Ncells_abc=(12, 12, 12), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB6 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(30, 30, 30), mos_dom=20, mos_spread=0.0, verbose=2)

simsAB7 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(30, 30, 30), mos_dom=100, mos_spread=.3, verbose=2)

simsAB8 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(25, 25, 25), mos_dom=20, mos_spread=.3, verbose=2)

simsAB9 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(25, 25, 25), mos_dom=20, mos_spread=.05, verbose=2)


simsAB10 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(22, 22, 22), mos_dom=20, mos_spread=.05, verbose=2)


simsAB11 = sim_utils.sim_twocolors2(
    optCrystal, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=.05, verbose=2)



def norm_sim(sims):
    simg = .5*(sims[0][0] + sims[1][0]).copy()
    simg -= simg.min()
    simg /= simg.max()
    return simg

def norm_img(img):
    dimg = img.copy()/28.
    dimg -= dimg.min()
    dimg /= dimg.max()
    return dimg

def cscale(img, contrast=0.1):
    m90 = np.percentile( img, 90)
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)
    
dimg = norm_img(pan_imgs[PID])

#fig,axs = subplots(2,2)
figure()
m9 = percentile(dimg[8:80,52:60],90)

shargs = dict(vmax=1, vmin=0.0, cmap='Greys')
subplot(221)
imshow( cscale(dimg,0.06), **shargs)
#xlim(52,160);ylim(80,8)
grid(1)
gca().set_xticklabels([])

subplot(222)
imshow( norm_sim(simsAB112), **shargs)
#xlim(52,160);ylim(80,8)
gca().set_xticklabels([])
gca().set_yticklabels([])
grid(1)

subplot(223)
imshow( norm_sim(simsAB1122), **shargs)
#xlim(52,160);ylim(80,8)
grid(1)

subplot(224)
im=imshow( norm_sim(simsAB11), **shargs)
#xlim(52,160);ylim(80,8)
gca().set_yticklabels([])
grid(1)

cax = gcf().add_axes([0.9, 0.1, 0.025, 0.8])
gcf().colorbar(im, cax=cax)


gcf().set_figheight(4.76)
gcf().set_figwidth(6.89)
subplots_adjust(hspace=0, wspace=.05,right=.95, bottom=.05, left=.05,top=.95 )

##################
#################
jitt_out2 = jitter_refine.jitter_panels(
    panel_ids=[PID],
       crystal=optCrystal,
       refls=refls_strong,
       det=detector,
       beam=iset.get_beam(0),
       FF=FF,
       en=ENERGIES,
       data_imgs=[pan_imgs[PID]],
       flux=FLUX,
       ret_best=False,
       Ncells_abc=(10,10, 10),
       oversample=2,
       Gauss=True,
       verbose=0,
       mos_dom=1,
       mos_spread=0.0,
       scanX=np.arange(-.2, .2, .025),  # these seemed to be sufficient ranges
       scanY=np.arange(-.2, .2, .025))


overlap2 = np.sum([jitt_out2[pid]['overlaps'] for pid in jitt_out2], axis=0)
max_pos2 = np.argmax(overlap2)
optA2 = jitt_out2[jitt_out2.keys()[0]]["A_seq"][
    max_pos2]  # just grab the first A_seq cause same sequence is tested on all panels

optCrystal2 = deepcopy(crystalAB)
optCrystal2.set_A(optA2)


######

simsAB52 = sim_utils.sim_twocolors2(
    optCrystal2, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=3,
    Ncells_abc=(12, 12, 12), mos_dom=20, mos_spread=0.0, verbose=2)
simsAB112 = sim_utils.sim_twocolors2(
    optCrystal2, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=.05, verbose=2)
simsAB1122 = sim_utils.sim_twocolors2(
    optCrystal2, detector, iset.get_beam(0), [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=[PID], Gauss=False, oversample=5,
    Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=.05, verbose=2)


np.savez_compressed("simVdat_%d_pan%d" % (shot_idx, PID),
    #simsAB=simsAB,
    #simsAB2=simsAB2,
    #simsAB3 = simsAB3,
    #simsAB4 = simsAB4,
    #simsAB5 = simsAB5,
    #simsAB6 = simsAB6,
    #simsAB7 = simsAB7,
    #simsAB8 = simsAB8,
    #simsAB9 = simsAB9,
    #simsAB10 = simsAB10,
    simsAB11 = simsAB11,
    simsAB112 = simsAB112,
    simsAB1122 = simsAB1122,
    dimg= dimg)

np.savez_compressed("shot%d_pan%d" % (shot_idx, PID),
    #simsAB=simsAB,
    #simsAB2=simsAB2,
    #simsAB3 = simsAB3,
    #simsAB4 = simsAB4,
    #simsAB5 = simsAB5,
    #simsAB6 = simsAB6,
    #simsAB7 = simsAB7,
    #simsAB8 = simsAB8,
    #simsAB9 = simsAB9,
    #simsAB10 = simsAB10,
    simsAB11 = simsAB11,
    simsAB112 = simsAB112,
    simsAB1122 = simsAB1122,
    dimg= dimg, crystal=optCrystal, crystal2=optCrystal2, detector=D, 
    beamB=beamB,beamA=beamA)

# ~~~~~~~~~~~
# ~~~~~~~~~~~
# ~~~~~~~~~~~
# ~~~~~~~~~~~
# ~~~~~~~~~~~


spot_dataA = spot_utils.get_spot_data_multipanel(
    simsAB[0], detector=detector,
    beam=beamA, crystal=optCrystal, thresh=0,
    filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

spot_dataB = spot_utils.get_spot_data_multipanel(
    simsAB[1], detector=detector,
    beam=beamB, crystal=optCrystal, thresh=0,
    filter=scipy.ndimage.filters.gaussian_filter, sigma=0.2)

spot_resid, dvecs, best = \
    metrics.indexing_residuals_twocolor(spot_dataA, spot_dataB, refls_strong, detector)

HA, HiA = spot_utils.refls_to_hkl(refls_strong, detector, beamA, optCrystal)
HB, HiB = spot_utils.refls_to_hkl(refls_strong, detector, beamB, optCrystal)

HAres = np.sqrt( np.sum((HA-HiA)**2, 1))
HBres = np.sqrt( np.sum((HB-HiB)**2, 1))
Hres = np.min( zip(HAres, HBres), axis=1)

hkl_tol = 0.15
d_idx = spot_resid[Hres < hkl_tol]
dvecs_idx = dvecs[Hres < hkl_tol]

dump = {"crystalAB": optCrystal,
        # "res_opt": res_opt,
        # "color_opt": color_opt,
        # "resAB": resAB,
        # "colorAB": colorAB,
        "beamA": beamA,
        "beamB": beamB,
        "overlap": overlap,
        "detector": detector,
        "spot_dataA": spot_dataA,
        "spot_dataB": spot_dataB,
        "d": spot_resid,
        "d_idx": d_idx,
        "dvecs_idx": dvecs_idx,
        "Hres": Hres,
        "dvecs": dvecs,
        "best": best,
        "rmsd": data['rmsd'],
        # "dist_vecs": dist_vecs,
        # "dists": dists,
        # "spot_data_combo": spot_data_combo,
        "refls_strong": refls_strong}

dump_name = data_name.replace(".pkl", "_ref.pkl")
utils.save_flex(dump, dump_name)

#sim_fname = data_name.replace(".pkl", "_ref_sim64.h5")
#sim_utils.save_twocolor(simsAB, iset, sim_fname, force=0)


