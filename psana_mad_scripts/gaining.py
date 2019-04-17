# coding: utf-8
import dxtbx
loader = dxtbx.load('_autogen_run62.loc')
data = [x.as_numpy_array() for x in loader.get_raw_data(0)]
get_ipython().magic(u'pylab')
imshow(data[0])
imshow(data[0], vin=-20, vmax=20)
imshow(data[0], vmin=-20, vmax=20)
loader.get_instance
loader.get_instance()
get_ipython().magic(u'pinfo loader.get_instance')
loader.get_instance( loader.get_image_file())
imshow(data[0], vmin=-20, vmax=120)
loader.nominal_gain_val
imshow(data[0], vmin=-20, vmax=30)
loader.gain
loader.gain.shape
data32 = array([hstack([data[i*2], data[i*2+1]]) for i un range(32)])
data32 = array([hstack([data[i*2], data[i*2+1]]) for i in range(32)])
imshow(data32[0], vmin=-20, vmax=30)
loader.cspad_mask
loader.cspad_mask.shape
figure()
imshow(loader.cspad_mask[0])
from cxid9114 import gain_utils
get_ipython().magic(u'pinfo gain_utils.get_gain_dists')
out = gain_utils.get_gain_dists(data32, loader.gain, loader.cspad_mask, plot=True, norm=True)
close()
close()
close()
close()
close()
close()
close()
close()
close()
close()
close()
while 1:
    close()
    
out[0]
plot(out[0], out[1])
plot(out[2], out[3])
ax = gca()
ax.set_yscale('log')
data2 = data32.copy()
data2[loader.gain] /= loader.nominal_gain_val
figure()
imshow(data32[0], vmin=-20, vmax=30)
figure()
imshow(data[0], vmin=-20, vmax=30)
imshow(data2[0], vmin=-20, vmax=30)
figure()
out2 = gain_utils.get_gain_dists(data2, loader.gain, loader.cspad_mask, plot=False, norm=True)
figure(1)
plot(out[0], out[1])
plot(out2[0], out2[1])
from cxid9114 import fit_utils
import lmfit
fit2 = fit_utils.fit_low_gain_dist(out2[0], out2[1], plot=1)
get_ipython().magic(u'pwd ')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'cd results')
get_ipython().magic(u'cd run62')
get_ipython().magic(u'ls ')
from dxtbx.model.experiment_list import ExperimentListFactory
data = [x.as_numpy_array() for x in loader.get_raw_data(54598)]
data32 = array([hstack([data[i*2], data[i*2+1]]) for i in range(32)])
data2 = data32.copy()
data2[loader.gain] /= loader.nominal_gain_val
imshow(data2[0], vmin=-20, vmax=30)
figure(1)
figure()
imshow(data2[0], vmin=-20, vmax=30)
figure();imshow(data32[0], vmin=-20, vmax=30)
out2 = gain_utils.get_gain_dists(data2, loader.gain, loader.cspad_mask, plot=False, norm=True)
fit2 = fit_utils.fit_low_gain_dist(out2[0], out2[1], plot=1)
fit2H = fit_utils.fit_high_gain_dist(out2[2], out2[3], plot=1)
fit2H
fit2H[0]
fit2H[1]
fit2H[2]
fit2H[2].x
fit2H[2].param
fit2H[2].params
fit2H[2].params['mu1']
fit2H[2].params['mu1'].value
fit2H[2].params['mu1'].value
fit2[2].params['mu1'].value
27.402076353130738 / 4.0863837771690239
data3 = data2.copy()
data3[loader.gain] *= 6.70570309774341
figure();imshow(data3[0], vmin=-20, vmax=30)
get_ipython().magic(u'save gaining.py 1-92')
