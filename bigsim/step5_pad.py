from __future__ import division,print_function
from six.moves import range
from six.moves import StringIO
from scitbx.array_family import flex
from scitbx.matrix import sqr,col
from simtbx.nanoBragg import shapetype
from simtbx.nanoBragg import nanoBragg
import libtbx.load_env # possibly implicit
from cctbx import crystal
from libtbx.development.timers import Profiler
import math
from cctbx import crystal_orientation
from six.moves import cPickle as pickle
import scitbx
import sys
from LS49.sim.step4_pad import microcrystal
from LS49.sim.util_fmodel import gen_fmodel
import os
from cxid9114 import utils
import h5py

# GLOBALS
sample_thick_mm = 0.005  # 50 micron GDVN nozzle makes a ~5ish micron jet
air_thick_mm = 0  # mostly vacuum but can make a helium layer of 1 micron maybe.. 
flux=2e11
add_spots_algorithm = "NKS"
big_data = "." # directory location for reference files
detpix_slowfast = (1800,1800)
pixsize_mm=0.11
distance_mm = 125
offset_adu=10
mos_spread_deg=0.05
mos_doms=25
beam_size_mm = 0.001
exposure_s = 1

# --------------------

def full_path(filename):
  return os.path.join(big_data,filename)

def raw_to_pickle(raw_pixels, fileout):
  with open(fileout, "wb") as F:
    pickle.dump(raw_pixels, F)

def write_safe(fname):
  # make sure file or compressed file is not already on disk
  return (not os.path.isfile(fname)) and (not os.path.isfile(fname+".gz"))

#usg below: channel_pixels(wavlen[x], flux[x], N, UMAT_nm, Amatrix_rot, rank, sf_all[x])
def channel_pixels(wavelength_A,flux, N, UMAT_nm, Amatrix_rot, rank, 
                sfall_channel):

  SIM = nanoBragg(detpixels_slowfast=detpix_slowfast,pixel_size_mm=pixsize_mm,Ncells_abc=(N,N,N),
    wavelength_A=wavelength_A,verbose=verbose)
  SIM.adc_offset_adu = offset_adu
  SIM.mosaic_spread_deg = mos_spread_deg # interpreted by UMAT_nm as a half-width stddev
  SIM.mosaic_domains = mos_doms  # 77 seconds.  With 100 energy points, 7700 seconds (2 hours) per image
  SIM.distance_mm=distance_mm
  SIM.set_mosaic_blocks(UMAT_nm)

  # get same noise each time this test is run
  SIM.seed = 1
  SIM.oversample=1
  SIM.wavelength_A = wavelength_A
  SIM.polarization=1
  SIM.default_F=0
  SIM.Fhkl=sfall_channel
  SIM.Amatrix_RUB = Amatrix_rot
  SIM.xtal_shape=shapetype.Gauss # both crystal & RLP are Gaussian
  SIM.progress_meter=True # False
  # flux is always in photons/s
  SIM.flux=flux
  SIM.exposure_s = exposure_s
  # assumes round beam
  SIM.beamsize_mm=beam_size_mm 
  temp=SIM.Ncells_abc
  print("Ncells_abc=",SIM.Ncells_abc)
  SIM.Ncells_abc=temp

  P = Profiler("nanoBragg C++ rank %d"%(rank))
  if add_spots_algorithm is "NKS":
    from boost.python import streambuf # will deposit printout into dummy StringIO as side effect
    SIM.add_nanoBragg_spots_nks(streambuf(StringIO()))
  elif add_spots_algorithm is "JH":
    SIM.add_nanoBragg_spots()
  elif add_spots_algorithm is "cuda":
    SIM.add_nanoBragg_spots_cuda()
  else: raise Exception("unknown spots algorithm")
  del P
  return SIM

def run_sim2smv(prefix,crystal,spectra,rotation,rank,quick=False,save_bragg=False):
  smv_fileout = prefix + ".img"
  if not quick:
    if not write_safe(smv_fileout):
      print("File %s already exists, skipping in rank %d"%(smv_fileout,rank))
      return

  direct_algo_res_limit = 1.7

  wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
  assert wavelength_A > 0
  if quick:
    wavlen = flex.double([wavelength_A]);
    flux = flex.double([flex.sum(flux)])
    print("Quick sim, lambda=%f, flux=%f"%(wavelength_A,flux[0]))

  sfall_main = GF.get_amplitudes()
  
  # use crystal structure to initialize Fhkl array
  sfall_main.show_summary(prefix = "Amplitudes used ")
  N = crystal.number_of_cells(sfall_main.unit_cell())

  SIM = nanoBragg(detpixels_slowfast=detpixels_slowfast,
        pixel_size_mm=pixsize_mm,Ncells_abc=(N,N,N),
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

  # get same noise each time this test is run
  SIM.seed = 1
  SIM.oversample=1
  SIM.wavelength_A = wavelength_A
  SIM.polarization=1
  # this will become F000, marking the beam center
  SIM.default_F=0
  #SIM.missets_deg= (10,20,30)
  print("mosaic_seed=",SIM.mosaic_seed)
  print("seed=",SIM.seed)
  print("calib_seed=",SIM.calib_seed)
  print("missets_deg =", SIM.missets_deg)
  SIM.Fhkl=sfall_main
  print("Determinant",rotation.determinant())
  Amatrix_rot = (rotation * sqr(sfall_main.unit_cell().orthogonalization_matrix())).transpose()
  print("RAND_ORI", prefix, end=' ')
  for i in Amatrix_rot: print(i, end=' ')
  print()

  SIM.Amatrix_RUB = Amatrix_rot
  #workaround for failing init_cell, use custom written Amatrix setter
  print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
  print("unit_cell_tuple=",SIM.unit_cell_tuple)
  Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
  Ori = crystal_orientation.crystal_orientation(Amat, crystal_orientation.basis_type.reciprocal)
  print("Python unit cell from SIM state",Ori.unit_cell())

  SIM.xtal_shape=shapetype.Gauss 
  SIM.progress_meter=False
  SIM.show_params()
  SIM.flux=flux
  SIM.exposure_s=exposure_s 
  SIM.beamsize_mm=beam_size_mm 
  temp=SIM.Ncells_abc
  print("Ncells_abc=",SIM.Ncells_abc)
  SIM.Ncells_abc=temp
  print("Ncells_abc=",SIM.Ncells_abc)
  print("xtal_size_mm=",SIM.xtal_size_mm)
  print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
  print("unit_cell_tuple=",SIM.unit_cell_tuple)
  print("missets_deg=",SIM.missets_deg)
  print("Amatrix=",SIM.Amatrix)
  print("beam_center_mm=",SIM.beam_center_mm)
  print("XDS_ORGXY=",SIM.XDS_ORGXY)
  print("detector_pivot=",SIM.detector_pivot)
  print("xtal_shape=",SIM.xtal_shape)
  print("beamcenter_convention=",SIM.beamcenter_convention)
  print("fdet_vector=",SIM.fdet_vector)
  print("sdet_vector=",SIM.sdet_vector)
  print("odet_vector=",SIM.odet_vector)
  print("beam_vector=",SIM.beam_vector)
  print("polar_vector=",SIM.polar_vector)
  print("spindle_axis=",SIM.spindle_axis)
  print("twotheta_axis=",SIM.twotheta_axis)
  print("distance_meters=",SIM.distance_meters)
  print("distance_mm=",SIM.distance_mm)
  print("close_distance_mm=",SIM.close_distance_mm)
  print("detector_twotheta_deg=",SIM.detector_twotheta_deg)
  print("detsize_fastslow_mm=",SIM.detsize_fastslow_mm)
  print("detpixels_fastslow=",SIM.detpixels_fastslow)
  print("detector_rot_deg=",SIM.detector_rot_deg)
  print("curved_detector=",SIM.curved_detector)
  print("pixel_size_mm=",SIM.pixel_size_mm)
  print("point_pixel=",SIM.point_pixel)
  print("polarization=",SIM.polarization)
  print("nopolar=",SIM.nopolar)
  print("oversample=",SIM.oversample)
  print("region_of_interest=",SIM.region_of_interest)
  print("wavelength_A=",SIM.wavelength_A)
  print("energy_eV=",SIM.energy_eV)
  print("fluence=",SIM.fluence)
  print("flux=",SIM.flux)
  print("exposure_s=",SIM.exposure_s)
  print("beamsize_mm=",SIM.beamsize_mm)
  print("dispersion_pct=",SIM.dispersion_pct)
  print("dispsteps=",SIM.dispsteps)
  print("divergence_hv_mrad=",SIM.divergence_hv_mrad)
  print("divsteps_hv=",SIM.divsteps_hv)
  print("divstep_hv_mrad=",SIM.divstep_hv_mrad)
  print("round_div=",SIM.round_div)
  print("phi_deg=",SIM.phi_deg)
  print("osc_deg=",SIM.osc_deg)
  print("phisteps=",SIM.phisteps)
  print("phistep_deg=",SIM.phistep_deg)
  print("detector_thick_mm=",SIM.detector_thick_mm)
  print("detector_thicksteps=",SIM.detector_thicksteps)
  print("detector_thickstep_mm=",SIM.detector_thickstep_mm)
  print("***mosaic_spread_deg=",SIM.mosaic_spread_deg)
  print("***mosaic_domains=",SIM.mosaic_domains)
  print("indices=",SIM.indices)
  print("amplitudes=",SIM.amplitudes)
  print("Fhkl_tuple=",SIM.Fhkl_tuple)
  print("default_F=",SIM.default_F)
  print("interpolate=",SIM.interpolate)
  print("integral_form=",SIM.integral_form)

  # simulated crystal is only 125 unit cells (25 nm wide)
  # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)
  print(crystal.domains_per_crystal)
  SIM.raw_pixels *= crystal.domains_per_crystal; # must calculate the correct scale!

  for x in range(len(flux)):
    P = Profiler("nanoBragg Python and C++ rank %d"%(rank))

    print("+++++++++++++++++++++++++++++++++++++++ Wavelength",x)
    CH = channel_pixels(wavlen[x], flux[x], N, UMAT_nm, Amatrix_rot, rank, sf_all[x])
    SIM.raw_pixels += CH.raw_pixels * crystal.domains_per_crystal
    CH.free_all()

    del P

  # image 1: crystal Bragg scatter
  if quick or save_bragg:  SIM.to_smv_format(fileout=prefix + "_intimage_001.img")

  if save_bragg: raw_to_pickle(SIM.raw_pixels, fileout=prefix + "_dblprec_001.pickle")

  # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
  bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
  SIM.Fbg_vs_stol = bg
  SIM.amorphous_sample_thick_mm = sample_thick_mm 
  SIM.amorphous_density_gcm3 = 1
  SIM.amorphous_molecular_weight_Da = 18
  SIM.flux=flux 
  SIM.beamsize_mm=beam_size_mm
  SIM.exposure_s=exposure_s
  SIM.add_background()
  if quick:  SIM.to_smv_format(fileout=prefix + "_intimage_002.img")

  # rough approximation to air
  bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
  SIM.Fbg_vs_stol = bg
  SIM.amorphous_sample_thick_mm = air_thick_mm # between beamstop and collimator
  SIM.amorphous_density_gcm3 = 1.2e-3
  SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
  print("amorphous_sample_size_mm=",SIM.amorphous_sample_size_mm)
  print("amorphous_sample_thick_mm=",SIM.amorphous_sample_thick_mm)
  print("amorphous_density_gcm3=",SIM.amorphous_density_gcm3)
  print("amorphous_molecular_weight_Da=",SIM.amorphous_molecular_weight_Da)
  SIM.add_background()

  #apply beamstop mask here

  # set this to 0 or -1 to trigger automatic radius.  could be very slow with bright images
  # settings for CCD
  SIM.detector_psf_kernel_radius_pixels=5;
  #SIM.detector_psf_fwhm_mm=0.08;
  #SIM.detector_psf_type=shapetype.Fiber # rayonix=Fiber, CSPAD=None (or small Gaussian)
  SIM.detector_psf_type=shapetype.Unknown # for CSPAD
  SIM.detector_psf_fwhm_mm=0
  SIM.quantum_gain = 28.
  #SIM.apply_psf()
  print("One pixel-->",SIM.raw_pixels[500000])

  # at this point we scale the raw pixels so that the output array is on an scale from 0 to 50000.
  # that is the default behavior (intfile_scale<=0), otherwise it applies intfile_scale as a multiplier on an abs scale.
  if quick:  SIM.to_smv_format(fileout=prefix + "_intimage_003.img")

  print("quantum_gain=",SIM.quantum_gain) #defaults to 1. converts photons to ADU
  print("adc_offset_adu=",SIM.adc_offset_adu)
  print("detector_calibration_noise_pct=",SIM.detector_calibration_noise_pct)
  print("flicker_noise_pct=",SIM.flicker_noise_pct)
  print("readout_noise_adu=",SIM.readout_noise_adu) # gaussian random number to add to every pixel (0 for PAD)
  # apply Poissonion correction, then scale to ADU, then adc_offset.
  # should be 10 for most Rayonix, Pilatus should be 0, CSPAD should be 0.

  print("detector_psf_type=",SIM.detector_psf_type)
  print("detector_psf_fwhm_mm=",SIM.detector_psf_fwhm_mm)
  print("detector_psf_kernel_radius_pixels=",SIM.detector_psf_kernel_radius_pixels)
  SIM.add_noise() #converts phtons to ADU.

  print("raw_pixels=",SIM.raw_pixels)
  extra = "PREFIX=%s;\nRANK=%d;\n"%(prefix,rank)
  SIM.to_smv_format_py(fileout=smv_fileout,intfile_scale=1,rotmat=True,extra=extra,gz=True)
  
  SIM.free_all()

def tst_one(quick=False,prefix="step5",save_bragg=False):
  
  spec_file =  h5py.File(full_path("test_data.h5"), "r")
  idx = 0
  wave_chans = spec_file["energy_bins"][()]
  spec = spec_file["raw_spec"][idx]
  C_ori = sqr(spec_file["Umats"][idx])
  wavelength_A = np.mean(wave_change)

  C = microcrystal(Deff_A = 1000, length_um = 2., beam_diameter_um = beam_size_mm*1000) 
 
  iterator = iter([wave_chans, spec, wavelength_A ])
      
  run_sim2smv(prefix = "tst_one_test_data_%d" % idx,
            crystal = C,
            spectra=iterator,
            rotation=rand_ori,
            quick=False,
            rank=0,
            save_bragg=True)

if __name__=="__main__":
  tst_one()
  print("OK")
