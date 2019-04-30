# coding: utf-8
SIM = Patt.SIM2
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
 SIM.Fbg_vs_stol = bg
 SIM.amorphous_sample_thick_mm = 0.1
 SIM.amorphous_density_gcm3 = 1
 SIM.amorphous_molecular_weight_Da = 18
 SIM.flux=1e12
 SIM.beamsize_mm=0.1
 SIM.exposure_s=0.1
 SIM.add_background()
get_ipython().magic(u'paste')
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
SIM.Fbg_vs_stol = bg
SIM.amorphous_sample_thick_mm = 0.1
SIM.amorphous_density_gcm3 = 1
SIM.amorphous_molecular_weight_Da = 18
SIM.flux=1e12
SIM.beamsize_mm=0.1
SIM.exposure_s=0.1
SIM.add_background()
from scitbx.array_family import fle
from scitbx.array_family import flex
bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
SIM.Fbg_vs_stol = bg
SIM.amorphous_sample_thick_mm = 0.1
SIM.amorphous_density_gcm3 = 1
SIM.amorphous_molecular_weight_Da = 18
SIM.flux=1e12
SIM.beamsize_mm=0.1
SIM.exposure_s=0.1
SIM.add_background()
SIM.raw_pixels()
SIM.raw_pixels
SIM.raw_pixels.as_numpy_array()
imshow(SIM.raw_pixels.as_numpy_array())
get_ipython().magic(u'pylab')
imshow(SIM.raw_pixels.as_numpy_array())
#imshow(SIM.raw_pixels.as_numpy_array() + simDa)
get_ipython().magic(u'whos ')
imshow(SIM.raw_pixels.as_numpy_array() + simsDataSum)
imshow(SIM.raw_pixels.as_numpy_array() + simsDataSum[0])
#imshow(SIM.raw_pixels.as_numpy_array() + simsDataSum[0])
SIM.raw_pixels.as_numpy_array()
SIM.raw_pixels.as_numpy_array().shape
simsDataSum
simsDataSum.sh
simsDataSum.size
array(simsDataSum)
array(simsDataSum).shape
sims = array(simsDataSum)
sims.mean()
bg_noise = random.normal(sims.mean(), sims.std()*10, sims.shape)
bg_noise
#imshow(b)
img2 = sims+bg_noise
imshow( img2[0])
bg_noise = random.normal(sims.mean()*0.01, sims.std()*10, sims.shape)
img2 = sims+bg_noise
imshow( img2[0])
bg_noise = random.normal(sims.mean()*0.001, sims.std()*10, sims.shape)
img2 = sims+bg_noise
imshow( img2[0])
bg_noise = random.normal(sims.mean()*0.001, sims.std()*0.1, sims.shape)
img2 = sims+bg_noise
imshow( img2[0])
bg_noise = random.normal(sims.mean()*0.1, sims.std()*0.1, sims.shape)
img2 = sims+bg_noise
imshow( img2[0])
bg_noise = random.normal(sims.mean()*0.1, sims.std()*1, sims.shape)
img2 = sims+bg_noise
imshow( img2[0])
imshow( img2[0], cmap='gray')
imshow( img2[0], cmap='gray_r')
bg_noise = random.normal(sims.mean()*0.1, sims.std()*2, sims.shape)
img2 = sims+bg_noise
imshow( img2[0], cmap='gray_r')
pval = img2 / img2.max()
pval
pval = img2.astype(float64) / (img2.max())
pval.dtu
pval.dtype
get_ipython().magic(u'pinfo np.random.multinomial')
get_ipython().magic(u'pinfo np.random.multinomial')
np.random.multinomial( 10,pval)
np.random.multinomial( 10,pval.ravel())
np.random.multinomial( 10, pval.ravel())
pval.sum()
pval = img2.astype(float64) / (img2.sum())
pval.sum()
np.random.multinomial( 10, pval.ravel())
np.random.multinomial( 1000, pval.ravel()).reshape( img2.shape)
np.random.multinomial( 10000, pval.ravel()).reshape( img2.shape)
np.random.multinomial( 100000, pval.ravel()).reshape( img2.shape)
np.random.multinomial( 1000000, pval.ravel()).reshape( img2.shape)
np.random.multinomial( 1e9, pval.ravel()).reshape( img2.shape)
np.random.multinomial( 1e12, pval.ravel()).reshape( img2.shape)
img3 = np.random.multinomial( 1e12, pval.ravel()).reshape( img2.shape)
figure()
imshow( img3 ) 
imshow( img3[0] ) 
imshow( img3[1] ) 
#np.savez("data")
get_ipython().magic(u'save sim_data_mono_wNoise.py 1079')
