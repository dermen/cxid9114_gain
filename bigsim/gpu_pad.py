from __future__ import division, print_function

#@profile
def run_sim2smv(Nshot_max,odir, prefix, rank, n_jobs, save_bragg=False, 
            save_smv=True, save_h5 =False, return_pixels=False):

  from six.moves import range, StringIO
  from six.moves import cPickle as pickle
  import os
  import h5py
  import math
  import sys
  import numpy as np
  from IPython import embed

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
  from cxid9114.parameters import ENERGY_CONV
  from cxid9114.bigsim import sim_spectra
 
  
  odir_j = os.path.join( odir, "job%d" % rank)
  if not os.path.exists(odir_j):
      os.makedirs(odir_j)

  overwrite = False
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
  verbose=0
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
  wave_chans = ENERGY_CONV/en_chans
  sfall_main = sim_spectra.load_spectra("test_sfall.h5")
    
  Nshot = spec_data.shape[0]
  if Nshot_max is not None:
    Nshot = min( Nshot_max, Nshot)
  print ("Job %d: Simulating %d shots" % (rank, Nshot))
    
  idx_range = np.array_split(np.arange(Nshot), n_jobs)
    
  Nshot_per_job = len(idx_range[rank])
 
  #if save_h5:
  #    h5_fname = os.path.join(odir , prefix % idx+ "_%Job%d.h5" % rank)
  #    h5 = h5py.File(h5_fname, "w")
    
  istart = idx_range[rank][0]
  istop = idx_range[rank][-1]
  smi_stride = 10
  for idx in idx_range[rank]:
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
      mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB"}'"""
      os.system(mem_usg)

    spec = spec_data[idx]
    rotation = sqr(Umat_data[idx])
    wavelength_A = np.mean(wave_chans)
    
  
    spectra = iter([(wave_chans, spec, wavelength_A)])
  
  
    direct_algo_res_limit = 1.7

    wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    assert wavelength_A > 0
    assert (len(wavlen)==len(flux)==len(sfall_main))

    N = crystal.number_of_cells(sfall_main[0].unit_cell())
    if use_microcrystal:
      Ncells_abc = (N,N,N)  
    SIM = nanoBragg(detpixels_slowfast=detpixels_slowfast,
          pixel_size_mm=pixsize_mm,Ncells_abc=Ncells_abc,
          wavelength_A=wavelength_A,verbose=verbose)
    SIM.adc_offset_adu = offset_adu # Do not offset by 40
    SIM.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = mos_doms
    SIM.distance_mm=distance_mm
    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=SIM.mosaic_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(SIM.mosaic_domains)
    for m in mosaic_rotation:
      site = col(mersenne_twister.random_double_point_on_sphere())
      UMAT_nm.append( site.axis_and_angle_as_r3_rotation_matrix(m,deg=False) )
    SIM.set_mosaic_blocks(UMAT_nm)

    SIM.seed = 1
    SIM.oversample=1
    SIM.wavelength_A = wavelength_A
    SIM.polarization=1
    SIM.default_F=0
    SIM.Fhkl=sfall_main[0].amplitudes()
    Amatrix_rot = (rotation * sqr(sfall_main[0].unit_cell().orthogonalization_matrix())).transpose()

    SIM.Amatrix_RUB = Amatrix_rot
    Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
    Ori = crystal_orientation.crystal_orientation(Amat, crystal_orientation.basis_type.reciprocal)

    SIM.xtal_shape=shapetype.Gauss 
    SIM.progress_meter=False
    SIM.flux=flux_ave
    SIM.exposure_s=exposure_s 
    SIM.beamsize_mm=beam_size_mm 
    temp=SIM.Ncells_abc
    SIM.Ncells_abc=temp

    raw_pixel_sum = flex.double( len( SIM.raw_pixels))
    Nflux = len(flux)
    for x in range(Nflux):

      if x %10==0:
        print("+++++++++++++++++++++++++++++++++++++++ Wavelength %d / %d" % (x+1, Nflux) , end="\r")
      if flux[x] ==0:
        continue
     
      SIM.wavelength_A=wavlen[x]
      SIM.flux=flux[x]
      SIM.Ncells_abc=Ncells_abc
      SIM.adc_offset_adu = offset_adu
      SIM.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
      SIM.mosaic_domains = mos_doms  #
      SIM.distance_mm=distance_mm
      SIM.set_mosaic_blocks(UMAT_nm)
      SIM.seed = 1
      SIM.polarization=1
      SIM.default_F=0
      SIM.Fhkl=sfall_main[x].amplitudes()
      SIM.Amatrix_RUB = Amatrix_rot
      SIM.xtal_shape=shapetype.Gauss 
      SIM.progress_meter=False 
      SIM.exposure_s = exposure_s
      SIM.beamsize_mm=beam_size_mm 

      SIM.timelog=timelog
      SIM.device_Id=rank
      SIM.raw_pixels *= 0  # just in case!
      SIM.add_nanoBragg_spots_cuda()
    
      if use_microcrystal:
        raw_pixel_sum += SIM.raw_pixels * crystal.domains_per_crystal

    print()
    
    SIM.raw_pixels = raw_pixel_sum + background
  
    SIM.detector_psf_kernel_radius_pixels=5;
    #SIM.detector_psf_fwhm_mm=0.08;
    #SIM.detector_psf_type=shapetype.Fiber # rayonix=Fiber, CSPAD=None (or small Gaussian)
    SIM.detector_psf_type=shapetype.Unknown # for CSPAD
    SIM.detector_psf_fwhm_mm=0
    SIM.quantum_gain = 28.
    #SIM.apply_psf()
    SIM.add_noise() #converts phtons to ADU.
    extra = "PREFIX=%s;\nRANK=%d;\n"%(prefix,rank)
  
    out =  SIM.raw_pixels.as_numpy_array() 
  
    if save_smv:
      SIM.to_smv_format_py(fileout=smv_fileout,intfile_scale=1,rotmat=True,extra=extra,gz=True)
    elif save_h5:
      f = h5py.File(h5_fileout, "w")
      f.create_dataset("bigsim_d9114", data=SIM.raw_pixels.as_numpy_array().astype(np.uint16).reshape(detpixels_slowfast), 
        compression="lzf")
      f.close()
  
    SIM.free_all()


if __name__=="__main__":
  from joblib import Parallel, delayed
  import sys
  prefix = "run62_%06d"
  odir = "/global/project/projectdirs/lcls/dermen/bigsim"
  tag = sys.argv[2]
  odir = odir + tag
  n_jobs = int(sys.argv[1])
  Parallel(n_jobs=n_jobs)( \
    delayed(run_sim2smv)(Nshot_max=None,odir=odir, prefix=prefix, \
            save_smv=False, save_h5=True, rank=jid, n_jobs=n_jobs) \
    for jid in range(n_jobs) )
  

