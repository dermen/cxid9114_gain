from __future__ import division, print_function

npout = None
verbose = 0
beamsize_mm=0.001
exposure_s=1
#@profile
def run_sim2smv(Nshot_max, odir, prefix, rank, n_jobs, save_bragg=False, 
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
 
  
  odir_j = os.path.join( odir, "job%d" % rank)
  if not os.path.exists(odir_j):
      os.makedirs(odir_j)

  add_noise = False
  add_background = False
  overwrite = True #$False
  offset_adu=30
  mos_spread_deg=0.015
  mos_doms=1000
  beam_size_mm=0.001
  exposure_s=1
  use_microcrystal=True #False
  Ncells_abc=(120,120,120)
  Deff_A = 2200
  length_um = 2.2
  timelog = False
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
    
    smv_fileout = os.path.join( odir_j, prefix % idx + ".img")
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
    spec = spec_data[2]
    rotation = sqr(Umat_data[2])
    wavelength_A = np.mean(wave_chans)
  
    spectra = iter([(wave_chans, spec, wavelength_A)])
    wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    assert wavelength_A > 0
    assert (len(wavlen)==len(flux)==len(sfall_main))

    N = crystal.number_of_cells(sfall_main[0].unit_cell())
    Ncells_abc = (N,N,N)  
    
    idxpath = "try3_idx2/job0/dump_0_data.pkl"
    idxpath = "try5_idx/job0/dump_0_data.pkl"
    idxpath = "try7_idx/job0/dump_0_data.pkl"
    Crystal = utils.open_flex(idxpath)["crystalAB"]
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
        oversample=1,
        Ncells_abc = Ncells_abc,
        mos_dom=mos_doms,
        verbose=verbose,
        mos_spread=mos_spread_deg,
        cuda=True, device_Id =rank,
        beamsize_mm=beamsize_mm,
        exposure_s=exposure_s,
        boost= crystal.domains_per_crystal) 
    
    out = np.sum( [ simsAB[i][0] for i in simsAB.keys() if simsAB[i]], axis=0)
    print()

    f = h5py.File(h5_fileout, "w")
    f.create_dataset("bigsim_d9114", 
        data=out, 
        compression="lzf")
    f.close()
 
    if npout is not None: 
        np.save(npout, out) # SIM.raw_pixels.as_numpy_array())


if __name__=="__main__":
  from joblib import Parallel, delayed
  import sys
  from argparse import ArgumentParser 
  prefix = "run62_%06d"
  parser = ArgumentParser("gpu sim pad")
  parser.add_argument("-g", dest="n_gpu",default=1, type=int, help="number of GPUs")
  parser.add_argument("-t", dest="tag", type=str, default="+-+", help="string tag")
  parser.add_argument("-m", dest="Nshot_max", type=int, default=-1, help="max num of shot to process per job")
  parser.add_argument("-o", dest="odir", type=str,default='.', help="output dir")
  parser.add_argument("-n", dest="npout", type=str,default=None, help="numpy output file")
  parser.add_argument("-v", dest="verbose", type=int,default=0, help="verbosity level (0-10)" )
  args = parser.parse_args()

  verbose = args.verbose
  n_jobs = args.n_gpu
  tag = args.tag
  Nshot_max = args.Nshot_max
  odir = args.odir
  npout = args.npout
  Parallel(n_jobs=n_jobs)( \
    delayed(run_sim2smv)(Nshot_max=Nshot_max,odir=odir, prefix=prefix, \
            save_smv=False, save_h5=True, rank=jid, n_jobs=n_jobs) \
    for jid in range(n_jobs) )
  

