
import sys

from cxid9114.refine.jitter_refine import JitterFactory
import os
from cxid9114 import utils
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from dxtbx.model.experiment_list import ExperimentListFactory
import numpy as np
import pandas
from itertools import izip
from scitbx.array_family import flex
from copy import deepcopy

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  

outlier_cutoff = 0.6  # probability to not be an outlier..
szx = szy = 8
Nmos_dom = 1
exp_name = sys.argv[1]
data_name = sys.argv[2]  
cuda = int( sys.argv[3])
outdir = sys.argv[4]
Ntrial = int( sys.argv[5])

if not os.path.exists( outdir):
    os.makedirs( outdir)

output_basename = os.path.basename(data_name).replace(".pkl", "_jitt.pkl")
outputname = os.path.join( outdir,output_basename) 

exp_lst = ExperimentListFactory.from_json_file(exp_name) #, check_format=False)
iset = exp_lst.imagesets()[0]
data = utils.open_flex( data_name)

waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e12, 1e12]  # fluxes of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]

data = utils.open_flex( data_name)
beamA = data["beamA"]
beamB = data["beamB"]
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

refls_strong = data["refls_strong"]
print ("###\n %d reflections\n" % len(refls_strong))

crystalAB = data["crystalAB"]
detector = data["detector"]
reflsPP = spot_utils.refls_by_panelname(refls_strong)
pids = reflsPP.keys()
raw_dat = iset.get_raw_data(0)
pan_imgs = [raw_dat[pid].as_numpy_array()
            for pid in pids]

roi_pp = []
counts_pp =[]
for pid, img in izip(pids, pan_imgs):
    panel = detector[pid]
    rois = spot_utils.get_spot_roi(
        reflsPP[pid],
        dxtbx_image_size=panel.get_image_size(),
        szx=szx, szy=szy)
    counts = spot_utils.count_roi_overlap(rois, img_size=img.shape)

    roi_pp.append(rois)
    counts_pp.append(counts)


def test(N=150):
    import time
    res = []
    Ncells_abc, mos_doms, mos_spread, xtal_shapes, ucell_a, ucell_b, ucell_c \
        = [], [], [], [], [], [], []
    new_crystals  = []
    for i in range(N):
        # add a jittered unit cell and a jittered U matrix ( in all 3 dimensions)
        new_crystal = JitterFactory.jitter_crystal(crystalAB)

        # jitter the size, shape and mosaicity of the crystal
        new_shape = JitterFactory.jitter_shape(
            min_Ncell=20, max_Ncell=100, min_mos_spread=0.005, max_mos_spread=0.1)

        print "### CRYSTAL SIMULATION %d / %d ###" % (i+1, N)
        print new_crystal.get_unit_cell().parameters()
        print new_shape
        print
        t = time.time()
        simsAB = sim_utils.sim_twocolors2(
            new_crystal,
            detector,
            beamA,
            FF,
            [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
            FLUX,
            pids=pids,
            profile='gauss', #new_shape['shape'],
            oversample=0,  # this should let nanoBragg decide how to oversample!
            Ncells_abc=new_shape['Ncells_abc'],
            mos_dom=Nmos_dom, 
            mos_spread=new_shape['mos_spread'],
            roi_pp=roi_pp,
            counts_pp=counts_pp,
            cuda=cuda)
        print("\t... took %.4f seconds" % (time.time() - t))
        Ncells_abc.append(new_shape['Ncells_abc'])
        mos_doms.append(Nmos_dom)
        mos_spread.append(new_shape['mos_spread'])
        xtal_shapes.append(new_shape['shape'])
        a, b, c, _, _, _ = new_crystal.get_unit_cell().parameters()
        ucell_a.append(a)
        ucell_b.append(b)
        ucell_c.append(c)
        new_crystals.append( new_crystal)
        res.append(np.array(simsAB[0]) + np.array(simsAB[1]))

    return [res, Ncells_abc, mos_doms, mos_spread, xtal_shapes,
                ucell_a, ucell_b, ucell_c, new_crystals]


data = test(Ntrial)
results = data.pop(0)
new_crystals = data.pop()

results = np.array( results)
# the shape is (Ntrial, Npanel, 185, 194)

pan_img_idx = {pid: idx for idx, pid in enumerate(pids)}


def overlay_imgs(imgA, imgB):
    """overlaye 2 images (np arrays) to get a measure of agreement"""
    return np.sum(imgA*imgB) / np.sqrt(np.sum(imgA**2) * np.sum(imgB**2))


Nrefl = len(refls_strong)
overlaps = np.zeros((Nrefl, Ntrial))

for i_refl, refl in enumerate(refls_strong):
    print i_refl
    pid = refl['panel']
    mask = spot_utils.get_single_refl_spot_mask(refl, (185, 194))
    data_img = pan_imgs[pan_img_idx[pid]][mask]
    for i_trial in range( Ntrial):
        sim_img = results[i_trial][pan_img_idx[pid]][mask]
        overlaps[i_refl, i_trial] = overlay_imgs(sim_img, data_img)

overlaps[overlaps > 1] = 0
overlaps = np.nan_to_num(overlaps)

overlaps -= overlaps.min()
overlaps /= overlaps.max()

score = overlaps.mean(1)

winners = score > outlier_cutoff

print sum( winners)

good_refls = refls_strong.select(flex.bool(winners))

Ncells_abc, mos_doms, mos_spread, xtal_shapes, ucell_a, ucell_b, ucell_c = data
Na, Nb, Nc = zip(*Ncells_abc)

data_dict = dict(
    Na=Na, Nb=Nb, Nc=Nc, mos_spread=mos_spread, Nmos_domain=mos_doms,
    xtals_shape=xtal_shapes, a=ucell_a, b=ucell_b, c=ucell_c)

# for the non-outlier reflections, store the overlap values for each trial
# then we can do a global clustering to see if there is a preferred simulation parameter
data_dict["scores"] = list(map(list, overlaps[winners].T))

Amats = [ list(C.get_A()) for C in new_crystals]
data_dict["Amat"] = Amats

df = pandas.DataFrame(data_dict)
df['ave_score'] = np.vstack(df.scores.values).mean(1)
df.to_pickle(outputname)

idx_max = df.ave_score.idxmax()
best_model = df.iloc[idx_max].drop([ 'scores', "Amat"])

best_Amat = tuple(df.iloc[idx_max]['Amat'])

with open(outputname.replace(".pkl", "_best.txt"), "w") as o:
    o.write(best_model.to_string())

dump_name = outputname.replace(".pkl", "_best_dump.pkl")
best_cryst = deepcopy(new_crystals[0])
best_cryst.set_A(best_Amat)
utils.save_flex({"good_refls": good_refls, "best_cryst":best_cryst}, dump_name )

