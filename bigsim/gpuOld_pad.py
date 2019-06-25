#!/usr/bin/env libtbx.python
from __future__ import division, print_function

npout = None
verbose = 0
beamsize_mm=0.001
exposure_s=1
overwrite = False
add_background=False 
add_noise=False 
idx_path = None
on_axis=False
force_twocolor=False
force_index=None
model_file = None

#@profile
def run_sim2smv(Nshot_max, odir, tag, rank, n_jobs, save_bragg=False, 
            save_smv=True, save_h5 =False, return_pixels=False):

  import os
  import h5py
  import math
  import sys
  import numpy as np
  from IPython import embed
  from cxid9114.bigsim.bigsim_geom import DET,BEAM
  import scitbx
  from scitbx.array_family import flex
  from scitbx.matrix import sqr,col
  from simtbx.nanoBragg import shapetype
  from simtbx.nanoBragg import nanoBragg
  import libtbx.load_env # possibly implicit
  from libtbx.development.timers import Profiler
  from cctbx import crystal,crystal_orientation
  from LS49.sim.step4_pad import microcrystal
  from cxid9114 import utils
  from cxid9114.sim import sim_utils
  from cxid9114.parameters import ENERGY_CONV, ENERGY_HIGH, ENERGY_LOW
  from cxid9114.bigsim import sim_spectra
  from dxtbx.model.crystal import CrystalFactory 
  
  odir_j = os.path.join( odir, "job%d" % rank)
  if not os.path.exists(odir_j):
      os.makedirs(odir_j)

  cryst_descr = {'__id__': 'crystal',
              'real_space_a': (79, 0, 0),
              'real_space_b': (0, 79, 0),
              'real_space_c': (0, 0, 38),
              'space_group_hall_symbol': '-P 4 2'} 
  Crystal = CrystalFactory.from_dict(cryst_descr)

  #offset_adu=30
  mos_spread_deg=0.015
  mos_doms=1000
  beam_size_mm=0.001
  exposure_s=1
  use_microcrystal=True 
  Deff_A = 2200
  length_um = 2.2
  timelog = False
  if add_background:
    background = utils.open_flex("background")
 
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
    
  Nshot = spec_data.shape[0]
    
  idx_range = np.array_split(np.arange(Nshot), n_jobs)
    
  Nshot_per_job = len(idx_range[rank])
  if Nshot_max  > 0 :
    Nshot_per_job = min( Nshot_max, Nshot_per_job)
  
  print ("Job %d: Simulating %d shots" % (rank, Nshot_per_job))
 
  istart = idx_range[rank][0]
  istop = istart + Nshot_per_job
  smi_stride = 10
  for idx in range( istart, istop): 
    print ("<><><><><><><><><><><><><><>")
    print ("Job %d; Image %d (%d - %d)" % (rank, idx+1, istart, istop))
    print ("<><><><><><><><><><><><><><>")
    
    smv_fileout = os.path.join( odir_j, "%s_%d.img" % (tag,idx))
    h5_fileout = smv_fileout + ".h5"
    
    if os.path.exists(smv_fileout) and not overwrite and save_smv:
        print("Shot %s exists: moving on" % smv_fileout)
        continue
    
    if os.path.exists(h5_fileout) and not overwrite and save_h5:
        print("Shot %s exists: moving on" % h5_fileout)
        continue
    
    if (rank==0 and idx % smi_stride==0):
      print("GPU status")
      os.system("nvidia-smi")
      
      print("\n\n")
      print("CPU memory usage")
      mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
      os.system(mem_usg)
    if force_index is not None:
        spec = spec_data[force_index]
        rotation = sqr(Umat_data[force_index])
    else:
        spec = spec_data[idx]
        rotation = sqr(Umat_data[idx])
    wavelength_A = np.mean(wave_chans)
  
    spectra = iter([(wave_chans, spec, wavelength_A)])
    wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    assert wavelength_A > 0
    assert (len(wavlen)==len(flux)==len(sfall_main))
    if np.sum(flux)==0:
        continue
    N = crystal.number_of_cells(sfall_main[0].unit_cell())
    Ncells_abc = (N,N,N)  
    if not on_axis:
        Crystal.set_U(rotation)
    
    if idx_path is not None:
        assert( model_file is None)
        Crystal = utils.open_flex(idx_path)['crystalAB']
    
    if model_file is not None:
        assert( idx_path is None)
        P = utils.open_flex(model_file)
        Crystal = P['crystal']
        #mos_spread_deg = P['mos_spread']
        #Ncells_abc = P['Ncells_abc']
    
    if force_twocolor: 
        flux *= 0
        flux[ilow] = 1e12
        flux[ihigh]=1e12
    
    simsAB = sim_utils.sim_twocolors2(
        Crystal,
        DET,
        BEAM,
        sfall_main,
        en_chans,
        flux,
        pids = None,
        profile="gauss",
        oversample=0,
        Ncells_abc = Ncells_abc,
        mos_dom=mos_doms,
        verbose=verbose,
        mos_spread=mos_spread_deg,
        cuda=True, 
        device_Id =rank,
        beamsize_mm=beamsize_mm,
        exposure_s=exposure_s,
        boost=crystal.domains_per_crystal)
    
    out = np.sum( [ simsAB[i][0] for i in simsAB.keys() if simsAB[i]], axis=0)
    if out.shape==():
        continue

    if add_background:
        out = out + background.as_numpy_array() #.reshape( out.shape)
    if add_noise:
        SIM = Patt.SIM2
        SIM.raw_pixels = flex.double(out.ravel())
        SIM.detector_psf_kernel_radius_pixels=5;
        SIM.detector_psf_type=shapetype.Unknown # for CSPAD
        SIM.detector_psf_fwhm_mm=0
        SIM.quantum_gain = 1 
        SIM.add_noise()
        out = SIM.raw_pixels.as_numpy_array()

    f = h5py.File(h5_fileout, "w")
    f.create_dataset("bigsim_d9114", 
        data=out, 
        compression="lzf")

    ua,ub,uc = Crystal.get_real_space_vectors()
    f.create_dataset("real_space_a", data=ua)
    f.create_dataset("real_space_b", data=ub)
    f.create_dataset("real_space_c", data=uc)
    f.create_dataset("space_group_hall_symbol", 
                data=Crystal.get_space_group().info().type().hall_symbol())
    f.create_dataset("Umatrix", data=Crystal.get_U())
    f.create_dataset("fluxes",data=flux)
    f.create_dataset("energies", data=en_chans)
    f.close()
 
    if npout is not None: 
        np.save(npout, out) # SIM.raw_pixels.as_numpy_array())


if __name__=="__main__":
  from joblib import Parallel, delayed
  import sys
  from argparse import ArgumentParser 
  parser = ArgumentParser("gpu sim pad")
  parser.add_argument("-g", dest="n_gpu",default=1, type=int, help="number of GPUs")
  parser.add_argument("-tag", dest="tag", type=str, default="run62", help="tag")
  parser.add_argument("-m", dest="Nshot_max", type=int, default=-1, help="max num of shot to process per job")
  parser.add_argument("-o", dest="odir", type=str,default='.', help="output dir")
  parser.add_argument("-n", dest="npout", type=str,default=None, help="numpy output file")
  parser.add_argument("-v", dest="verbose", type=int,default=0, help="verbosity level (0-10)" )
  parser.add_argument("-idx-path", dest="idx_path", default=None, type=str, 
                    help="path to a dump pickle for debugging" )
  parser.add_argument("--add-bg", dest="add_bg",action='store_true',help="add background" )
  parser.add_argument("--add-noise", dest="add_noise",action='store_true',help="add noise" )
  parser.add_argument("--overwrite", dest="overwrite", 
                    action='store_true', help="whether to overwrite" )
  parser.add_argument("--on-axis", dest="onaxis", 
                    action='store_true', help="whether to apply rotation mat (debugging)" )
  parser.add_argument("--force-twocolor", dest="force2", 
                    action='store_true', help="whether to force two colors" )
  parser.add_argument('-force-index', dest='force_idx', type=int, default=None, 
                help="use specific index for spectrum and rotation")
  parser.add_argument('-force-model', dest='force_model', type=str, default=None, 
                help="force loading a model from parameters list stored in a pkl file")
  
  args = parser.parse_args()

  model_file = args.force_model 
  force_index=args.force_idx 
  force_twocolor = args.force2
  on_axis = args.onaxis
  idx_path = args.idx_path
  add_background=args.add_bg
  add_noise = args.add_noise
  overwrite = args.overwrite
  verbose = args.verbose
  n_jobs = args.n_gpu
  Nshot_max = args.Nshot_max
  odir = args.odir
  npout = args.npout
  Parallel(n_jobs=n_jobs)( \
    delayed(run_sim2smv)(Nshot_max=Nshot_max,odir=odir, tag=args.tag, \
            save_smv=False, save_h5=True, rank=jid, n_jobs=n_jobs) \
    for jid in range(n_jobs) )
  

