from __future__ import division, print_function

npout = None
verbose = 0
#@profile
def run_sim2smv(Nshot_max, odir, prefix, rank, n_jobs, save_bragg=False, 
            save_smv=True, save_h5 =False, return_pixels=False):

  from six.moves import range, StringIO
  from six.moves import cPickle as pickle
  from cxid9114.sim import sim_utils
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
  from cxid9114.parameters import ENERGY_CONV, ENERGY_HIGH, ENERGY_LOW
  from cxid9114.bigsim import sim_spectra
 
  
  odir_j = os.path.join( odir, "job%d" % rank)
  if not os.path.exists(odir_j):
      os.makedirs(odir_j)

  add_noise = False
  add_background = False
  overwrite = True #$False
  sample_thick_mm = 0.005  # 50 micron GDVN nozzle makes a ~5ish micron jet
  air_thick_mm =0  # mostly vacuum, maybe helium layer of 1 micron
  flux_ave=2e11
  add_spots_algorithm="cuda"
  big_data = "." # directory location for reference files
  detpixels_slowfast = (1800,1800)
  pixsize_mm=0.11
  distance_mm = 125
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
  
    direct_algo_res_limit = 1.7

    wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    assert wavelength_A > 0
    assert (len(wavlen)==len(flux)==len(sfall_main))

    N = crystal.number_of_cells(sfall_main[0].unit_cell())
    if use_mcrocrystal:
      Ncells_abc = (N,N,N)  
    
    flux *= 0
    flux[ilow] = 1e12
    flux[ihigh]=1e12
    
    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=mos_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mos_doms)
    for m in mosaic_rotation:
      site = col(mersenne_twister.random_double_point_on_sphere())
      UMAT_nm.append( site.axis_and_angle_as_r3_rotation_matrix(m,deg=False) )
    
    Amatrix_rot = (rotation * sqr(sfall_main[0].unit_cell().orthogonalization_matrix())).transpose()
    
    #SIM = nanoBragg(detpixels_slowfast=detpixels_slowfast,
    #      pixel_size_mm=pixsize_mm,Ncells_abc=Ncells_abc,
    #      wavelength_A=wavelength_A,verbose=verbose)
    SIM = nanoBragg(detector=DET, beam=BEAM, panel_id=0,verbose=verbose)

    SIM.adc_offset_adu = offset_adu # Do not offset by 40
    SIM.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = mos_doms
    SIM.distance_mm=distance_mm
    SIM.set_mosaic_blocks(UMAT_nm)
    SIM.seed = 1
    SIM.oversample=1
    SIM.wavelength_A = wavelength_A
    SIM.polarization=1
    SIM.default_F=0
    SIM.Fhkl=sfall_main[0].amplitudes()
    SIM.progress_meter=False
    SIM.flux=flux_ave
    SIM.exposure_s=exposure_s 
    SIM.beamsize_mm=beam_size_mm 
    SIM.Ncells_abc=Ncells_abc
    SIM.xtal_shape=shapetype.Gauss 
    
    idxpath = "try3_idx2/job0/dump_0_data.pkl"
    Amat = sim_utils.Amatrix_dials2nanoBragg(utils.open_flex(idxpath)["crystalAB"])
    #SIM.Umatrix = rotation.elems
    #SIM.unit_cell_Adeg = sfall_main[0].unit_cell()
    
    #SIM2 = nanoBragg(detector=DET, beam=BEAM, panel_id=0,verbose=verbose)

    #SIM2.adc_offset_adu = offset_adu # Do not offset by 40
    #SIM2.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
    #SIM2.mosaic_domains = mos_doms
    #SIM2.distance_mm=distance_mm
    #SIM2.set_mosaic_blocks(UMAT_nm)
    #SIM2.seed = 1
    #SIM2.oversample=1
    #SIM2.wavelength_A = wavelength_A
    #SIM2.polarization=1
    #SIM2.default_F=0
    #SIM2.Fhkl=sfall_main[0].amplitudes()
    #SIM2.progress_meter=False
    #SIM2.flux=flux_ave
    #SIM2.exposure_s=exposure_s 
    #SIM2.beamsize_mm=beam_size_mm 
    #SIM2.Ncells_abc=Ncells_abc
    #SIM2.xtal_shape=shapetype.Gauss 
    #SIM2.Umatrix = rotation.elems
    #SIM2.unit_cell_Adeg = sfall_main[0].unit_cell()
    
    #SIM.Amatrix_RUB = Amatrix_rot
    #SIM.Amatrix = Amatrix_rot.inverse()
    
    raw_pixel_sum = flex.double(len(SIM.raw_pixels))
    Nflux = len(flux)
    print("Beginning the loop")
    for x in range(Nflux):
      if x % 10==0:
        print("+++++++++++++++++++++++++++++++++++++++ Wavelength %d / %d" % (x+1, Nflux) , end="\r")
      if flux[x] ==0:
        continue
      #SIM.Amatrix = Amatrix_rot.inverse() 
      print (SIM.Ncells_abc)
      SIM.wavelength_A=wavlen[x]
      SIM.flux=flux[x]
      SIM.Fhkl=sfall_main[x].amplitudes()
      SIM.Amatrix = Amat#Amatrix_rot.inverse()
      
      #sim_utils.compare_sims(SIM, SIM2)
      #SIM.Ncells_abc=Ncells_abc
      #SIM.adc_offset_adu = offset_adu
      #SIM.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
      #SIM.mosaic_domains = mos_doms  #
      #SIM.distance_mm=distance_mm
      #SIM.set_mosaic_blocks(UMAT_nm)
      #SIM.seed = 1
      #SIM.polarization=1
      #SIM.default_F=0
      #SIM.xtal_shape=shapetype.Gauss 
      #SIM.progress_meter=False 
      #SIM.exposure_s = exposure_s
      #SIM.beamsize_mm=beam_size_mm 

      SIM.timelog=timelog
      SIM.device_Id=rank
      SIM.raw_pixels *= 0  # just in case!
      SIM.add_nanoBragg_spots_cuda()
    
      if use_microcrystal:
        raw_pixel_sum += SIM.raw_pixels * crystal.domains_per_crystal

    print()

    SIM.raw_pixels = raw_pixel_sum
    if add_background:
        SIM.raw_pixels = SIM.raw_pixels + background 
    
    SIM.detector_psf_kernel_radius_pixels=5;
    #SIM.detector_psf_fwhm_mm=0.08;
    #SIM.detector_psf_type=shapetype.Fiber # rayonix=Fiber, CSPAD=None (or small Gaussian)
    SIM.detector_psf_type=shapetype.Unknown # for CSPAD
    SIM.detector_psf_fwhm_mm=0
    SIM.quantum_gain = 28.
    #SIM.apply_psf()
    if add_noise:
        SIM.add_noise() #converts phtons to ADU.
    extra = "PREFIX=%s;\nRANK=%d;\n"%(prefix,rank)
  
    out =  SIM.raw_pixels.as_numpy_array() 
  
    if save_smv:
      SIM.to_smv_format_py(fileout=smv_fileout,intfile_scale=1,rotmat=True,extra=extra,gz=True)
    elif save_h5:
      f = h5py.File(h5_fileout, "w")
      f.create_dataset("bigsim_d9114", 
        data=SIM.raw_pixels.as_numpy_array().astype(np.uint16).reshape(detpixels_slowfast), 
        compression="lzf")
      f.close()
 
    if npout is not None: 
        np.save(npout, SIM.raw_pixels.as_numpy_array())
    SIM.free_all()


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
  

