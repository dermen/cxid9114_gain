# coding: utf-8
import pylab as plt
get_ipython().magic(u'pylab')
imshow( sims[0], vmin=0, vmax=3e2)
imshow( sims[0])
imshow( sims[0]*117)
imshow( sims[0]*117, vmax=10)
imshow( sims[0]*117, vmax=9)
imshow( sims[0]*117, vmax=1)
imshow( sims[0]*117, vmax=10)
imshow( sims[0]*117, vmax=20)
imshow( sims[0]*117, vmax=20)
#crystal
#C = microcrystal(Deff_A=4000, length_um=5, beam_diameter_um=1)
get_ipython().magic(u'whos ')
SIM=Patt.SIM2
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
SIM.Fbg_vs_stol = bg
SIM.amorphous_sample_thick_mm = 0.1
SIM.amorphous_density_gcm3 = 1
SIM.amorphous_molecular_weight_Da = 18
SIM.flux=1e12
SIM.beamsize_mm=0.003 # square (not user specified)
SIM.exposure_s=1.0 # multiplies flux x exposure
SIM.add_background()
SIM.raw_pixels.as_numpy_array()
imshow(SIM.raw_pixels.as_numpy_array())
water = SIM.raw_pixels.as_numpy_array()
imshow( water)
figure()
imshow( water)
SIM.raw_pixels *=0
bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
SIM.Fbg_vs_stol = bg
#SIM.amorphous_sample_thick_mm = 35 # between beamstop and collimator
SIM.amorphous_sample_thick_mm = 10 # between beamstop and collimator
SIM.amorphous_density_gcm3 = 1.2e-3
SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
print("amorphous_sample_size_mm=",SIM.amorphous_sample_size_mm)
print("amorphous_sample_thick_mm=",SIM.amorphous_sample_thick_mm)
print("amorphous_density_gcm3=",SIM.amorphous_density_gcm3)
print("amorphous_molecular_weight_Da=",SIM.amorphous_molecular_weight_Da)
SIM.add_background()
air = SIM.raw_pixels.as_numpy_array()
imshow( air)
imshow( air, vmax=400)
imshow( air, vmax=200)
imshow( water, vmax=200)
close()
water.max()
water.mni()
water.min()
imshow( sims[0] + water, vmax=200)
close()
imshow( sims[0] + water)
imshow( sims[0]*117 + water)
imshow( sims[0]*117 + water+air)
imshow( sims[0]*117 +air)
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
SIM.Fbg_vs_stol = bg
SIM.amorphous_sample_thick_mm = 0.004
SIM.amorphous_density_gcm3 = 1
SIM.amorphous_molecular_weight_Da = 18
SIM.flux=1e12
SIM.beamsize_mm=0.003 # square (not user specified)
SIM.exposure_s=1.0 # multiplies flux x exposure
SIM.add_background()
SIM.raw_pixels *=0
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
SIM.Fbg_vs_stol = bg
SIM.amorphous_sample_thick_mm = 0.004
SIM.amorphous_density_gcm3 = 1
SIM.amorphous_molecular_weight_Da = 18
SIM.flux=1e12
SIM.beamsize_mm=0.003 # square (not user specified)
SIM.exposure_s=1.0 # multiplies flux x exposure
SIM.add_background()
water = SIM.raw_pixels.as_numpy_array()
imshow( sims[0]*117 + water+air)
SIM print("quantum_gain=",SIM.quantum_gain) #defaults to 1. converts photons to ADU
  print("adc_offset_adu=",SIM.adc_offset_adu)
  print("detector_calibration_noise_pct=",SIM.detector_calibration_noise_pct)
  print("flicker_noise_pct=",SIM.flicker_noise_pct)
  print("readout_noise_adu=",SIM.readout_noise_adu) # 
print("quantum_gain=",SIM.quantum_gain) #defaults to 1. converts photons to ADU
print("adc_offset_adu=",SIM.adc_offset_adu)
print("detector_calibration_noise_pct=",SIM.detector_calibration_noise_pct)
print("flicker_noise_pct=",SIM.flicker_noise_pct)
print("readout_noise_adu=",SIM.readout_noise_adu) # 
print("detector_psf_type=",SIM.detector_psf_type)
print("detector_psf_fwhm_mm=",SIM.detector_psf_fwhm_mm)
print("detector_psf_kernel_radius_pixels=",SIM.detector_psf_kernel_radius_pixels)
SIM.adc_offset_adu=10
SIM.raw_pixels
SIM.raw_pixels*=0
#SIM.raw_pixels = flex.double()
from scitbx.array_family import flex
flex.double(water)
flex.double(water+air+sims[0])
sig = flex.double(water+air+sims[0])
SIM.raw_pixels = sig
SIM.add_noise()
sig_n = SIM.raw_pixels()
sig_n = SIM.raw_pixels
img = sig.as_numpy_array()
img_n = sig_n.as_numpy_array()
imshow( img)
imshow( img_n)
sig = water+air+sims[0]
imshow( water+air_sims[0])
imshow( water+air+sims[0])
sims[0]
imshow( sims[0])
imshow( water+air+sims[0]*117)
img = water+air+sims[0]*117
imshow( img ) 
#img ) 
def cscale(img, contrast=0.1):
    m90 = np.percentile(img, 90) 
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)
imshow( img ) 
imshow( cscale(img) ) 
imshow( cscale(img,0.02) ) 
close()
imshow( cscale(img,0.02) ) 
imshow( cscale(img,0.2) ) 
imshow( cscale(img,0.1) ) 
imshow( cscale(img,0.001) ) 
imshow( cscale(img,0.99) ) 
imshow( cscale(img,0.9) ) 
imshow( cscale(img,0.7) ) 
imshow( cscale(img,0.5) ) 
imshow( cscale(img,0.4) ) 
get_ipython().magic(u'save noises 1-87')
get_ipython().magic(u'pwd ')
sig = flex.double(water+air+sims[0]*117)
SIM.raw_pixels*=0
SIM.raw_pixels = sig
SIM.add_noise()
sig_n = SIM.raw_pixels
img_n = sig_n.as_numpy_array()
imshow( cscale(img_n,0.4) ) 
imshow( cscale(img_n,0.5) ) 
imshow( cscale(img_n,0.8) ) 
imshow( cscale(img_n,0.7) ) 
get_ipython().magic(u'save noises 1-87')
