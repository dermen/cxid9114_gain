#!/usr/bin/env libtbx.python
from __future__ import division, print_function

# these can change according to input args
verbose = 0
overwrite = False
thresh=1e-2
force_twocolor=False
Node = 0

# these cannot change according to input args
beamsize_mm=0.001
exposure_s=1
smi_stride = 5

#@profile
def run_paramList(Ntrials, odir, tag, rank, n_jobs, pkl_file):

  import os
  import sys
  from copy import deepcopy  
  
  import numpy as np
  import h5py
  from IPython import embed
  from scipy.ndimage.morphology import binary_dilation
  
  import scitbx
  from scitbx.array_family import flex
  from scitbx.matrix import sqr,col
  from simtbx.nanoBragg import shapetype
  from simtbx.nanoBragg import nanoBragg
  import libtbx.load_env  # possibly implicit
  from libtbx.development.timers import Profiler
  from dxtbx.model.crystal import CrystalFactory 
  from cctbx import crystal,crystal_orientation
  
  from cxid9114 import utils
  from cxid9114.sim import sim_utils
  from cxid9114.spots import spot_utils
  from cxid9114.bigsim.bigsim_geom import DET,BEAM
  from cxid9114.parameters import ENERGY_CONV, ENERGY_HIGH, ENERGY_LOW
  from cxid9114.refine.jitter_refine import make_param_list
  from cxid9114.bigsim import sim_spectra
  
  from LS49.sim.step4_pad import microcrystal

  data_pack = utils.open_flex(pkl_file)
  CRYST = data_pack['crystalAB']

  mos_spread_deg=0.015
  mos_doms=1000
  beam_size_mm=0.001
  exposure_s=1
  use_microcrystal=True 
  Deff_A = 2200
  length_um = 2.2
  timelog = False
 
  crystal = microcrystal(Deff_A = Deff_A, length_um = length_um, 
        beam_diameter_um = beam_size_mm*1000, verbose=False) 
  spec_file =  h5py.File("simMe_data_run62.h5", "r")
  spec_data = spec_file["hist_spec"]
  Umat_data = spec_file["Umats"]
  en_chans = spec_file["energy_bins"][()]
  ilow = np.abs(en_chans - ENERGY_LOW).argmin()
  ihigh = np.abs(en_chans - ENERGY_HIGH).argmin()
  wave_chans = ENERGY_CONV/en_chans
  sfall_main = sim_spectra.load_spectra("test_sfall.h5")
  
  refls_strong = data_pack['refls_strong'] 
  strong_mask_img = spot_utils.strong_spot_mask(
                refls_strong, (1800,1800) ) 

  # edge detection in the ground truth strong mask image
  reference_img = (binary_dilation(strong_mask_img, iterations=1).astype(int) - 
                strong_mask_img.astype(int) ).astype(bool)
  
  param_fileout = os.path.join( odir, "rank%d_%s.pkl" % (rank, tag))

  param_list = make_param_list(
            CRYST, DET, BEAM, Ntrials, 
              rot=0.09, cell=0.1, eq=(1,1,0), 
            min_Ncell=20, max_Ncell=40, 
              min_mos_spread=0.005, max_mos_spread=0.02)

  for p in param_list:
      print(p['crystal'].get_unit_cell().parameters())
  shot_idx = int(data_pack['img_f'].split("_")[-1].split(".")[0])
  Fluxes = spec_data[2] #shot_idx]
  Pmax = param_list[0]
  F1max = 0
  for i_trial in range(Ntrials):
    print ("<><><><><><><><><><><><><><>")
    print ("Job %d; Trial %d / %d" % (rank, i_trial+1, Ntrials))
    print ("<><><><><><><><><><><><><><>")
    
    if (rank==0 and i_trial % smi_stride==0):
      print("GPU status")
      os.system("nvidia-smi")
      
      print("\n\n")
      print("CPU memory usage")
      mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
      os.system(mem_usg)
    
 
    assert (len(wave_chans)==len(Fluxes)==len(sfall_main))
    if np.sum(Fluxes)==0:
        print ("Cannot simulate with an all-zeros spectrum!")
        sys.exit()
    
    N = crystal.number_of_cells(sfall_main[0].unit_cell())
    Ncells_abc = (N,N,N)  
    
    if force_twocolor: 
        Fluxes *= 0
        Fluxes[ilow] = 1e12
        Fluxes[ihigh]=1e12
    
    P = param_list[i_trial]
    simsAB = sim_utils.sim_twocolors2(
        P['crystal'],
        DET,
        BEAM,
        sfall_main,
        en_chans,
        Fluxes,
        pids = None,
        profile="gauss",
        oversample=0,
        Ncells_abc = Ncells_abc, #P['Ncells_abc'],
        mos_dom=mos_doms,
        verbose=verbose,
        mos_spread=mos_spread_deg, 
        #@mos_spread=P['mos_spread'],
        cuda=True, 
        device_Id=rank,
        beamsize_mm=beamsize_mm,
        exposure_s=exposure_s,
        boost=crystal.domains_per_crystal)
    
    out = np.sum( [ simsAB[i][0] for i in simsAB.keys() if simsAB[i]], axis=0)
    if out.shape==():
        print("This simsAB output has an empty shape, something is wrong!")
        sys.exit()
    
    trial_refls = spot_utils.refls_from_sims([out], DET, BEAM, thresh=thresh)

    trial_spotmask = spot_utils.strong_spot_mask(
                 trial_refls, (1800,1800) ) 

    trial_img = (binary_dilation(trial_spotmask, iterations=1).astype(int) - 
                trial_spotmask.astype(int) ).astype(bool)

    comp_img = trial_img.astype(int) + reference_img.astype(int)*2

    Nfalse_neg = (comp_img==2).sum()  # reference has signal but trial doesnt
    Nfalse_pos = (comp_img==1).sum()  # trial has signal but reference doesnt
    Ntrue_pos = (comp_img==3).sum()  #

    Precision = float(Ntrue_pos) / (Ntrue_pos + Nfalse_pos)
    Recall = float(Ntrue_pos) / (Ntrue_pos + Nfalse_neg)
    
    F1 = 2.*(Precision*Recall) / (Precision+Recall)
    if F1 > F1max:
        Pmax = {'crystal': P['crystal'], 
                'mos_spread': P['mos_spread'], 
                'Ncells_abc': P['Ncells_abc'], "F1": F1}
        F1max = F1

    print("Rank %d, Trial %d: F1score = %.5f" % (rank, i_trial, F1))
    
  utils.save_flex(Pmax, param_fileout)
  return F1max 

if __name__=="__main__":
  from joblib import Parallel, delayed
  import sys
  import glob
  import os
  from argparse import ArgumentParser 
  import numpy as np
  parser = ArgumentParser("gpu sim pad")
  parser.add_argument("-g", dest="n_gpu",default=1, type=int, help="number of GPUs")
  parser.add_argument("-t", dest="tag", type=str, default="run62", help="tag")
  #parser.add_argument("-o", dest="odir", type=str,default='.', help="output dir")
  parser.add_argument("-v", dest="verbose", type=int,default=0, help="verbosity level (0-10)" )
  parser.add_argument("--overwrite", dest="overwrite", 
                    action='store_true', help="whether to overwrite" )
  parser.add_argument("--on-axis", dest="onaxis", 
                    action='store_true', help="whether to apply rotation mat (debugging)" )
  parser.add_argument("--force-twocolor", dest="force2", 
                    action='store_true', help="whether to force two colors" )
  parser.add_argument('-n', dest='n', type=int, default=1, help="Total number of trials across all jobs (GPUs)")
  parser.add_argument('-N', dest='nodes', type=int, default=[1,0], nargs=2, help="Number of nodes, and node id")
  parser.add_argument("-i",dest='i',type=str, required=True, help="input glob of pickle files from the indexing script (dump*.pkl)" )
  parser.add_argument("-thresh", dest='thresh', default=1e-2, type=float, help="ADU threshold for simulation spot finding")
  
  args = parser.parse_args()

  num_nodes, node_id = args.nodes  
  thresh = args.thresh
  n_jobs = args.n_gpu
  Ntrials = args.n
  assert( Ntrials >= 1)
        
  trials_per_gpu = [len(trials) for trials in 
    np.array_split(range(Ntrials), args.n_gpu)]

  print("Trials per gpu:")
  print (trials_per_gpu) 
  
  force_twocolor = args.force2
  overwrite = args.overwrite
  verbose = args.verbose
  pkl_files = glob.glob(args.i)

  pkl_files = pkl_files[node_id::num_nodes]  # divide up if working on multiple nodes
  for pkl in pkl_files:
      odir = pkl.replace(".pkl", "_refine")
      if not os.path.exists(odir):
          os.makedirs(odir)
      results = Parallel(n_jobs=n_jobs)( \
        delayed(run_paramList)(trials_per_gpu[jid],odir=odir, tag=args.tag, \
                rank=jid, n_jobs=n_jobs, pkl_file=pkl) \
        for jid in range(n_jobs) )
       
      for rank,F1 in enumerate(results):
        print ("PKL_OCTOPUS: %s Rank %d: F1score=%.5f"% (pkl, rank, F1))

