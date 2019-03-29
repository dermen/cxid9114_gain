# coding: utf-8
idx = 998
from cxid9114.spots import spot_utils, integrate
#spot_utils.open
from cxid9114 import utils
r = utils.open_flex('refl_%d_feb8th.pkl'% idx)
r
get_ipython().magic(u'pinfo integrate.integrate')
mask = utils.open_flex("../../dials_mask_64panels_2.pkl")
#loader = dxtbx.load("")
import dxtbx
loader = dxtbx.load("../../image_files/_autogen_run62.loc")
get_ipython().magic(u'cp ../../d9114_32pan_mask.npy .')
loader = dxtbx.load("../../image_files/_autogen_run62.loc")
ISET = loader.get_imageset(loader.get_image_file())
iset = ISET[idx]
iset
iset = ISET[idx:idx+1]
iset
get_ipython().magic(u'pinfo integrate.integrate')
integrate.integrate( r, mask, iset)
integrate = reload( integrate)
integrate.integrate( r, mask, iset)
len(r)
integrate.integrate( r, mask, iset)
integrate = reload( integrate)
integrate.integrate( r, mask, iset)
integrate.integrate( r, mask, iset, gain=7.5)
integrate.integrate( r, mask, iset, gain=28.)
integrate.integrate( r, mask, iset, gain=28.)[0][:10]
integrate.integrate( r, mask, iset, gain=26.)[0][:10]
integrate.integrate( r, mask, iset, gain=27.)[0][:10]
get_ipython().magic(u'pwd ')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'pwd ')
get_ipython().magic(u'save integrate_example.py 1-36')
