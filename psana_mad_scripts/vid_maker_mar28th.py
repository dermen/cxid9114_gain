
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
import pylab as plt
from itertools import izip
from scitbx.array_family import flex
from copy import deepcopy
from IPython import embed

shot_idx = int( sys.argv[1])

run = 63
exp_name = "results/run%d/exp_%d_feb8th.json" % (run, shot_idx)
data_name = "results/run%d/dump_%d_feb8th.pkl" % (run, shot_idx)
outdir = "videos/run%d/shot%d" % (run, shot_idx)
Ntrial = 5 
outlier_cutoff = 0.1  # probability to not be an outlier..
szx = szy = 8
Nmos_dom = 1
save_figs = True

if not os.path.exists( outdir):
    os.makedirs( outdir)

output_basename = os.path.basename(data_name).replace(".pkl", "_jitt.pkl")
outputname = os.path.join( outdir,output_basename) 

exp_lst = ExperimentListFactory.from_json_file(exp_name) #, check_format=False)
iset = exp_lst.imagesets()[0]
data = utils.open_flex( data_name)

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]

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
img_sh = (185, 194)

# these are the axis specific details, e.g. boundaries and 
# patches outlining the strong spot masks
xlims = {}
ylims = {}
patches = {}

def get_spot_patches(Masks, fc='none', ec='C1', lw=1):
    """ a list of strong spot masks"""
    patches = []
    for M in Masks:
        y,x = np.where( M)
        
        pts = [ [(i-.5, j-.5), 
                ( i-.5, j+.5), 
                (i+.5, j+.5), 
                (i+.5,j-.5), 
                (i-.5, j-.5)] for i,j in zip(x,y) ]

        for p in pts:
            path = plt.mpl.path.Path( p)
            patch = plt.mpl.patches.PathPatch( path, fc=fc, ec=ec, lw=lw)
            patches.append( patch)

    return patches

Malls = {}
for pid, img in izip(pids, pan_imgs):
    panel = detector[pid]
    rois = spot_utils.get_spot_roi(
        reflsPP[pid],
        dxtbx_image_size=panel.get_image_size(),
        szx=szx, szy=szy)
    counts = spot_utils.count_roi_overlap(rois, img_size=img.shape)

    roi_pp.append(rois)
    counts_pp.append(counts)

    spot_masks = spot_utils.strong_spot_mask(
        reflsPP[pid], img_sh, as_composite=False)

    # composite mask
    Mall = np.any( spot_masks, axis=0)
    
    yall,xall = np.where( Mall)  # use comp mask to determine the boundary
    
    xlims[pid] = (xall.min()-szx, xall.max()+szx )
    ylims[pid] = ( yall.max()+szy, yall.min()-szy)
    patches[pid] = get_spot_patches(spot_masks)

    Malls[pid] =spot_masks 

pan_img_idx = {pid: idx for idx, pid in enumerate(pids)}

Nrefl = len(refls_strong)
overlaps = np.zeros((Nrefl, Ntrial))

def overlay_imgs(imgA, imgB):
    """overlaye 2 images (np arrays) to get a measure of agreement"""
    return np.sum(imgA*imgB) / np.sqrt(np.sum(imgA**2) * np.sum(imgB**2))

#
#@profile
def test(N=150):
    import time
    Ncells_abc, mos_doms, mos_spread, xtal_shapes, ucell_a, ucell_b, ucell_c \
        = [], [], [], [], [], [], []
    new_crystals  = []
    for i_trial in range(N):
        # add a jittered unit cell and a jittered U matrix ( in all 3 dimensions)
        new_crystal = JitterFactory.jitter_crystal(crystalAB)

        # jitter the size, shape and mosaicity of the crystal
        new_shape = JitterFactory.jitter_shape(
            min_Ncell=20, max_Ncell=100, min_mos_spread=0.005, max_mos_spread=0.1)

        print "### CRYSTAL SIMULATION %d / %d ###" % (i_trial+1, N)
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
            cuda=False)
        
        sim_imgs = np.array(simsAB[0]) + np.array(simsAB[1])
        print("\t... took %.4f seconds" % (time.time() - t))
        
        print "Saving..."
        t = time.time()

        fig = plt.figure()
        #ax = fig.add_axes( ([0,0,1,1]))
        ax = plt.gca()
        #ax.axis('off')
        #ax.set_aspect('auto')
        #axim = ax.imshow(np.random.random((10,10)), cmap='Blues')
        for ipid,pid in enumerate(pids):
            patches = get_spot_patches(Malls[pid])
            vals = sim_imgs[ipid] > 1e-10
            m = vals.mean()
            s = vals.std()
            vmax = m+3*s
            #axim.set_data( sim_imgs[ipid])
            #axim.set_clim( 0, vmax)
            ax.imshow(sim_imgs[ipid], vmin=0,vmax=vmax,
                    cmap='Blues')
            ax.set_ylim( ylims[pid])
            ax.set_xlim( xlims[pid])

            #i1,i2 = ax.get_xlim()
            #j2,j1 = ax.get_ylim()
            #dy = j2 - j1
            #dx = i2 - i1
            #fig.set_size_inches((16,8)) #9, 9.*dy / dx))
            for p in patches:
                ax.add_patch(p)
            
            # save the figure
            figdir = os.path.join( outdir, "panel%d"%pid)
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            figname = "pan%d_trial%d.png" % (pid, i_trial)
            figpath = os.path.join( figdir, figname)
            ax.set_title("Panel %d; Model Trial %d" % (pid, i_trial))
            plt.savefig(figpath, dpi=150)#bbox_inches='tight')
            while ax.patches:
                ax.patches.pop()
            ax.images.pop()
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
        #res.append(np.array(simsAB[0]) + np.array(simsAB[1]))

        for i_refl, refl in enumerate(refls_strong):
            pid = refl['panel']
            mask = spot_utils.get_single_refl_spot_mask(refl, (185, 194))
            data_img = pan_imgs[pan_img_idx[pid]][mask]
            sim_img = sim_imgs[pan_img_idx[pid]][mask]
            overlaps[i_refl, i_trial] = overlay_imgs(sim_img, data_img)

    return [Ncells_abc, mos_doms, mos_spread, xtal_shapes,
                ucell_a, ucell_b, ucell_c, new_crystals]


data = test(Ntrial)
#results = data.pop(0)
new_crystals = data.pop()

#results = np.array( results)
# the shape is (Ntrial, Npanel, 185, 194)

overlaps[overlaps > 1] = 0
overlaps = np.nan_to_num(overlaps)

overlaps -= overlaps.min()
overlaps /= overlaps.max()

score = overlaps.mean(1)

winners = score > outlier_cutoff

print sum( winners)

good_refls = refls_strong.select(flex.bool(winners))

#####3

good_reflsPP = spot_utils.refls_by_panelname(good_refls)

fig = plt.figure()
ax = plt.gca()
for ipid,pid in enumerate(pids):
    
    ax.set_ylim( ylims[pid])
    ax.set_xlim( xlims[pid])
    
    patches = get_spot_patches( Malls[pid])
    for p in patches:
        ax.add_patch(p)
    
    #if pid not in good_reflsPP.keys():
    # default
    ax.set_title("Panel %d; No winners" % pid)
  
    if pid in good_reflsPP.keys(): 
        good_R_lst = list(good_reflsPP[pid])

        kept_sel = []
        for i_r,r in enumerate(reflsPP[pid]):
            kept_sel.append( r in good_R_lst)
        
        if np.any( kept_sel): 
            kept_refls = reflsPP[pid].select( flex.bool(kept_sel) )
            spot_masks = spot_utils.strong_spot_mask(
                kept_refls, img_sh, as_composite=False)

            # color the kept reflections green! 
            winner_patches = get_spot_patches( spot_masks, ec='C2')

            for p in winner_patches:
                ax.add_patch(p)
        
            ax.set_title("Panel %d; Some Winners!" % pid )
        
    # save the figure
    figdir = os.path.join( outdir, "panel%d"%pid)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    figname = "pan%d_trial%d.jpg" % (pid, Ntrial)
    figpath = os.path.join( figdir, figname)
    plt.savefig(figpath, dpi=150)
    while ax.patches:
        ax.patches.pop()

#####

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

