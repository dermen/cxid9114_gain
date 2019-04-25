
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
import h5py
from scipy.ndimage import label


shot_idx = int(sys.argv[1])
tag = sys.argv[2]
run =int( sys.argv[3])
idxdir_tag = sys.argv[4]
dumpname_tag = sys.argv[5]

cell_fwhm=0.05
rot_width=0.2
min_Ncell=20
max_Ncell=70
Ntrial = 20 

basedir = "/Users/dermen/jitterize/"
#exp_name = "%s/results/run%d%s/exp_%d%s.json" % \
#    (basedir,run,idxdir_tag ,shot_idx, dumpname_tag)
data_name = "%s/results/run%d%s/dump_%d%s.pkl" % \
    (basedir,run,idxdir_tag ,shot_idx, dumpname_tag)
outdir = "%s/videos/run%d/shot%d%s" % (basedir,run, shot_idx,tag)
szx = szy = 11
Nmos_dom = 1
save_figs = True
save_sims= True
min_prob = 0.2 

skip_existing = False

if not os.path.exists( outdir):
    os.makedirs( outdir)
elif skip_existing:
    print "EXISITS!"
    exit()

output_basename = os.path.basename(data_name).replace(".pkl", "_jitt.pkl")
outputname = os.path.join( outdir,output_basename) 

if save_sims:
    h5_fname = outputname.replace(".pkl",".h5py")
    h5 = h5py.File(h5_fname, "w")

#exp_lst = ExperimentListFactory.from_json_file(exp_name, check_format=False)
#iset = exp_lst.imagesets()[0]

data = utils.open_flex(data_name)
beamA = data["beamA"]
beamB = data["beamB"]
enA = parameters.ENERGY_CONV / beamA.get_wavelength()
enB = parameters.ENERGY_CONV / beamB.get_wavelength() 

print enA,enB
#enB = 9034.7
print enA,enB
ENERGIES = [enA, enB]

FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors
FLUX = [1e12, 1e12]  # fluxes of the beams


refls_strong = data["refls_strong"]
print ("###\n %d reflections\n" % len(refls_strong))

crystalAB = data["crystalAB"]
detector = data["detector"]
reflsPP = spot_utils.refls_by_panelname(refls_strong)
pids = reflsPP.keys()
#raw_dat = iset.get_raw_data(0)
#pan_imgs = [raw_dat[pid].as_numpy_array()
#            for pid in pids]

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
    import pylab as plt
    import matplotlib as mpl
    patches = []
    for M in Masks:
        y,x = np.where( M)
        
        pts = [ [(i-.5, j-.5), 
                ( i-.5, j+.5), 
                (i+.5, j+.5), 
                (i+.5,j-.5), 
                (i-.5, j-.5)] for i,j in zip(x,y) ]

        for p in pts:
            path = mpl.path.Path( p)
            patch = mpl.patches.PathPatch( path, fc=fc, ec=ec, lw=lw)
            patches.append( patch)

    return patches

Malls = {}
for pid in pids:
    panel = detector[pid]
    rois = spot_utils.get_spot_roi(
        reflsPP[pid],
        dxtbx_image_size=panel.get_image_size(),
        szx=szx, szy=szy)
    counts = spot_utils.count_roi_overlap(rois, img_size=img_sh)

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

if save_sims:
    h5.create_dataset("pids", data=pids)
    #h5.create_dataset("pan_imgs", data=pan_imgs)


#@profile
def test(N=150):
    import time
    thresh=1e-3
    Ncells_abc, mos_doms, mos_spread, xtal_shapes, ucell_a, ucell_b, ucell_c \
        = [], [], [], [], [], [], []
    new_crystals  = []
    master_img = None
    for i_trial in range(N):
        # add a jittered unit cell and a jittered U matrix ( in all 3 dimensions)
        new_crystal = JitterFactory.jitter_crystal(
            crystalAB, 
            cell_jitter_fwhm=cell_fwhm, 
            rot_jitter_width=rot_width)

        # jitter the size, shape and mosaicity of the crystal
        new_shape = JitterFactory.jitter_shape(
            min_Ncell=min_Ncell, max_Ncell=max_Ncell, 
            min_mos_spread=0.005, max_mos_spread=0.1)
        print "### CRYSTAL SIMULATION %d / %d ###" % (i_trial+1, N)
        print new_crystal.get_unit_cell().parameters()
        print new_shape
        print ENERGIES
        t = time.time()
        simsAB = sim_utils.sim_twocolors2(
            new_crystal,
            detector,
            beamA,
            FF,
            ENERGIES, 
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
        if save_sims:
            h5.create_dataset("sim_imgs_%d"% i_trial, data=sim_imgs)
       
        if master_img is None:
            master_img = (sim_imgs > thresh ).astype(int)
        else:
            master_img += (sim_imgs > thresh).astype(int)

        print("\t... took %.4f seconds" % (time.time() - t))
        
        if save_figs:
            print "Saving..."
            t = time.time()
            fig = plt.figure()
            ax = plt.gca()
            for ipid,pid in enumerate(pids):
                patches = get_spot_patches(Malls[pid])
                vals = sim_imgs[ipid] > 1e-10
                m = vals.mean()
                s = vals.std()
                vmax = m+3*s
                ax.imshow(sim_imgs[ipid], vmin=0,vmax=vmax,
                        cmap='Blues')
                ax.set_ylim( ylims[pid])
                ax.set_xlim( xlims[pid])

                for p in patches:
                    ax.add_patch(p)
                
                # save the figure
                figdir = os.path.join(outdir, "panel%d"%pid)
                if not os.path.exists(figdir):
                    os.makedirs(figdir)
                figname = "pan%d_trial%d.jpg" % (pid, i_trial)
                figpath = os.path.join( figdir, figname)
                ax.set_title("Panel %d; Model Trial %d" % (pid, i_trial))
                plt.savefig(figpath, dpi=50)
                
                # remove patches and image so can replot 
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

    return master_img


master_img = test(Ntrial)
master_img = master_img.astype(float) / Ntrial

Ntotal_bad = 0
winners = []
for refl in refls_strong:
    pid = refl['panel']
    pidx = pan_img_idx[pid]

    proba_im = master_img[pidx]

    mask = spot_utils.get_single_refl_spot_mask(refl,  (185,194))

    proba_pix = mask*proba_im
    tota_proba = np.nan_to_num(proba_pix[ proba_pix > 0].mean())
    
    is_reject = tota_proba < min_prob
    
    if is_reject:
        Ntotal_bad += 1

    winners.append( np.logical_not(is_reject))

winners = np.array( winners)
print "Ntotal bad = %d " % Ntotal_bad
print "Ntotal winners = %d that will be saved to a new refl table" % sum( winners)

good_refls = refls_strong.select(flex.bool(winners))
good_reflsPP = spot_utils.refls_by_panelname(good_refls)

if save_figs:
    fig = plt.figure()
    ax = plt.gca()
    for ipid,pid in enumerate(pids):
        
        ax.set_ylim( ylims[pid])
        ax.set_xlim( xlims[pid])
        
        patches = get_spot_patches( Malls[pid])
        for p in patches:
            ax.add_patch(p)
        
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
        plt.savefig(figpath, dpi=50)
        while ax.patches:
            ax.patches.pop()

dump_name = outputname.replace(".pkl", "_probable_refls.pkl")
utils.save_flex(good_refls, dump_name )

if save_sims:
    h5.close()

