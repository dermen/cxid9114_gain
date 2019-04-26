# coding: utf-8
l s-lrt OUTS/
get_ipython().magic(u'ls -lrt OUTS/')
# ,cmap='Blues')
img = simsDataSum
imshow( img, vmin=0, vmax=None ,cmap='Blues')
get_ipython().magic(u'pylab')
imshow( img, vmin=0, vmax=None ,cmap='Blues')
imshow( img[0], vmin=0, vmax=None ,cmap='Blues')
np.mean(img, 0)
np.mean(img[0], 0)
imshow( img[0], vmin=0, vmax=None ,cmap='Blues')
imshow( img[1], vmin=0, vmax=None ,cmap='Blues')
clf()
close()
close()
close()
close()
imshow( img[1], vmin=0, vmax=None ,cmap='Blues')
imshow( img[0], vmin=0, vmax=None ,cmap='Blues')
imshow( img[2], vmin=0, vmax=None ,cmap='Blues')
imshow( img[3], vmin=0, vmax=None ,cmap='Blues')
np.mean(img[0], 0)
np.mean(img, 0)
np.mean(img, -1).mean(-1)
#pval = img2.astype(float64) / (img2.sum())
bg_noise = random.normal(sims.mean()*0.1, sims.std()*1.3, sims.shape)
sims = array(simsDataSum)
bg_noise = random.normal(sims.mean()*0.1, sims.std()*1.3, sims.shape)
img2 = sims+bg_noise
imshow( img2[3], vmin=0, vmax=None ,cmap='Blues')
close()
imshow( img2[3], vmin=0, vmax=None ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=1e5 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e5 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e20 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e10 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e14 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e15 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e9 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e4 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e6 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e7 ,cmap='Blues')
imshow( img2[3], vmin=0, vmax=6e8 ,cmap='Blues')
imshow( img2[0], vmin=0, vmax=6e8 ,cmap='Blues')
imshow( img2[1], vmin=0, vmax=6e8 ,cmap='Blues')
imshow( img2[32], vmin=0, vmax=6e8 ,cmap='Blues')
imshow( img2[36], vmin=0, vmax=6e8 ,cmap='Blues')
#imshow( img2[36], vmin=0, vmax=6e8 ,cmap='Blues')
pvals = 
pval = img2.astype(float64) / (img2.sum())
img3 = np.random.multinomial( 1e12, pval.ravel()).reshape( img2.shape)
pval = img2.astype(float64) / (img2.sum())
pval.sum()
img3 = np.random.multinomial( 1e12, pval.ravel()).reshape( img2.shape)
np.savez("noise_img3", img=img2)
imshow( img2[36], vmin=0, vmax=6e8 ,cmap='Blues')
for i in range(64):
    cla()
    imshow( img2[i], vmin=0, vmax=6e8 ,cmap='Blues')
    
    pause(0.4)
    
for i in range(64):
    cla()
    imshow( img2[i], vmin=0, vmax=6e8 ,cmap='gray')
    
    pause(0.4)
    
close()
get_ipython().magic(u'pwd ')
img = load("noise_img3.npz")
img
img = load("noise_img3.npz")["img"]
img
imshow( img[0] ) 
clf()
imshow( img[0] ) 
get_ipython().magic(u'pylab')
imshow( img[0] ) 
figure()
imshow( img[0] ) 
imshow( img[1] ) 
close()
imshow( img[1] ) 
#imshow( img[1] ) 
get_ipython().magic(u'pylab')

def cscale(img, contrast=0.1):
    m90 = np.percentile( img, 90) 
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)
cscale(img)
imshow(cscale(img))
imshow(cscale(img[1]))
figure()
imshow(img[1])
figure()
imshow(cscale(img[1], 0.01))
imshow(cscale(img[1], 0.5))
imshow(cscale(img[1], 0.1))
imshow(cscale(img[1], 0.05))
imshow(cscale(img[1], 0.05), vmin=0, vmax=1)
imshow(cscale(img[1], 0.05), vmin=0, vmax=0.9)
imshow(cscale(img[1], 0.05), vmin=0, vmax=0.5)
imshow(cscale(img[1], 0.05))
imshow(cscale(img[1], 0.05))
close()
close()
close()
im = imshow(cscale(img[1], 0.05))
im.get_clim()
imshow(cscale(img[1], vmin=-0.15, vmax=1))
imshow(cscale(img[1],0.05), vmin=-0.15, vmax=1)
imshow(cscale(img[1],0.05), vmin=-0.15, vmax=1)
imshow(cscale(img[1],0.05), vmin=-0.15, vmax=2)
imshow(cscale(img[1],0.05), vmin=-0.15, vmax=1)
cscale( img[1], 0.05)
get_ipython().magic(u'pwd ')
get_ipython().magic(u'clear ')
get_ipython().magic(u'ls ')
cryst
Cabc = []
for aa,bb,cc in zip(a_var, b_var, c_var):
    Bnew = sqr( (aa,0,0, 0,bb,0, 0,0,cc)).inverse()
    C = deepcopy(cryst)
    C.set_B(Bnew)
    Cabc.append(C)
    
a
a,b,c,_,_,_ = cryst.get_unit_cell().parameters()
a
b
c
#a_var = linspace(a-0., 100+3,25)
1.1*a
1.01*a
a
1.05*a
1.02*a
a
a_var[0]
a_var = linspace(100-3, 100+3,25)
a_var = linspace(a-3, a+3,25)
a_var
1.03*a
0.97*a
0.975*a
linspace( 0.975*a, a*1.025, 25)
linspace( 0.975*a, a*1.025, 25)
a_var = linspace( 0.975*a, a*1.025, 25)
b_var = linspace( 0.975*b, b*1.025, 25)
c_var = linspace( 0.975*c, c*1.025, 25)
Cabc = []
for aa,bb,cc in zip(a_var, b_var, c_var):
    Bnew = sqr( (aa,0,0, 0,bb,0, 0,0,cc)).inverse()
    C = deepcopy(cryst)
    C.set_B(Bnew)
    Cabc.append(C)
    
from scitbx.array_family import flex
from scitbx.matrix import sqr
Cabc = []
for aa,bb,cc in zip(a_var, b_var, c_var):
    Bnew = sqr( (aa,0,0, 0,bb,0, 0,0,cc)).inverse()
    C = deepcopy(cryst)
    C.set_B(Bnew)
    Cabc.append(C)
    
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['ENERGIES'] = [parameters.ENERGY_LOW] #, parameters.ENERGY_HIGH]
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    params.append( param)
    
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    #print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['ENERGIES'] = [parameters.ENERGY_LOW] #, parameters.ENERGY_HIGH]
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    params.append( param)
    
utils.save_flex(params, "increase_abc_mono3.pkl")
get_ipython().magic(u'pwd ')
8944 *0.05
8944 *0.003
8944 *0.0015
dlam = 8944 *0.0015
(8944 - dlam, 8944+dlam, 7)
(8944 - dlam, 8944+dlam, 6)
linspace(8944 - dlam, 8944+dlam, 6)
linspace(8944 - dlam, 8944+dlam, 7)
spec = linspace(8944 - dlam, 8944+dlam, 5)
spec
en_spec = linspace(8944 - dlam, 8944+dlam, 5)
flux_spec = array([0.2, 0.5, 1, 0.5, 0.2])
flux_spec
bar( en_spec, flux_spec, width=.9*(en_spec[1] - en_spec[0]))
figure()
close()
close()
close()
close()
close()
close()
close()
figure()
bar( en_spec, flux_spec, width=.9*(en_spec[1] - en_spec[0]))
#np.s
#bartlett
np.savez("spec_alpha.npz", en=en_spec, flux=flux_spec)
#img2 = sims+bg_noise
#bg_noise = random.normal(sims.mean()*0.1, sims.std()*1.3, sims.shape)
#sims = array(simsDataSum)
#img2 = sims+bg_noise
en_spec = linspace(8944 - 5*dlam, 8944+5*dlam, 25)
[exp(-(x-12)*(x-12)) for x in range(25)]
[exp(-(x-12)*(x-12)/2) for x in range(25)]
[exp(-(x-12)*(x-12)/6) for x in range(25)]
[exp(-(x-12)*(x-12)/10) for x in range(25)]
[exp(-(x-12)*(x-12)/100) for x in range(25)]
[exp(-(x-12)*(x-12)/70) for x in range(25)]
flux = [exp(-(x-12)*(x-12)/70) for x in range(25)]
bar( en_spec, flux, width=.9*(en_spec[1] - en_spec[0]))
figure()
bar( en_spec, flux, width=.9*(en_spec[1] - en_spec[0]))
np.savez("spec_beta.npz", en=en_spec, flux=flux_spec)
len( en_spec)
len( flux_spec)
np.savez("spec_beta.npz", en=en_spec, flux=flux)
en_spec = linspace(8944 - 3*dlam, 8944+3*dlam, 25)
#flux = [exp(-(x-12)*(x-12)/70) for x in range(25)]
np.savez("spec_gamma.npz", en=en_spec, flux=flux)
6*dlam
6*dlam/8944
10*dlam/8944
en_spec = linspace(8944 - 1.5*dlam, 8944+1.5*dlam, 25)
3*dlam/8944
#np.savez("spec_gamma3.npz", en=en_spec, flux=flux)
load("spec_gamma.npz")
load("spec_gamma.npz")['en_spec']
load("spec_gamma.npz")['en']
load("spec_gamma.npz")['en'][0,-1]
load("spec_gamma.npz")['en'][0,24]
load("spec_gamma.npz")['en'][[0,-1]]
a= load("spec_gamma.npz")['en'][[0,-1]]
a[1] - a[0]
(a[1] - a[0]) / 8944
np.savez("spec_gamma3.npz", en=en_spec, flux=flux)
a= load("spec_gamma3.npz")['en'][[0,-1]]
(a[1] - a[0]) / 8944
#np.s
en_spec = linspace(8944 - dlam, 8944+dlam, 25)
#flux = [exp(-(x-12)*(x-12)/70) for x in range(25)
np.savez("spec_gamma4.npz", en=en_spec, flux=flux)
a= load("spec_gamma4.npz")['en'][[0,-1]]
(a[1] - a[0]) / 8944
close()
en_spec = linspace(8944 - 5*dlam, 8944+5*dlam, 25)
np.savez("spec_gamma2.npz", en=en_spec, flux=flux)
#en_spec = linspace(8944 - 3dlam, 8944+3dlam, 25)
en_spec = linspace(8944 - 1.5*dlam, 8944+1.5*dlam, 25)
en_spec
(en_spec[-1] - en_spec[0])
(en_spec[-1] - en_spec[0]) / 8944.
en_spec = linspace(8944 - 1.5*dlam, 8944+1.5*dlam, 7)
en_spec
flux = [exp(-(x-3.5)*(x-3.5)/70) for x in range(7)]
flux
flux = [exp(-(x-3.5)*(x-3.5)/50) for x in range(7)]
[exp(-(x-3.5)*(x-3.5)/50) for x in range(7)]
[exp(-(x-3.5)*(x-3.5)/20) for x in range(7)]
[exp(-(x-3.5)*(x-3.5)/10) for x in range(7)]
[exp(-(x-3)*(x-3)/10) for x in range(7)]
flux = [exp(-(x-3)*(x-3)/10) for x in range(7)]
np.savez("spec_gamma5.npz", en=en_spec, flux=flux)
en_spec
en_spec.shape
len(flux)
get_ipython().magic(u'pwd ')
get_ipython().magic(u'clear ')
#load("spec_gamma.npz")['en_spec']
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    #print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['ENERGIES'] = [parameters.ENERGY_LOW] #, parameters.ENERGY_HIGH]
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    params.append( param)
    
en_spec
flux
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    #print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    params.append( param)
    
a_var = linspace( 0.975*a, a*1.025, 25)
a
#a = (fpA*fpA + fdpA*fdpA) / (Yb_f0[0]*Yb_f0[0])
a,b,c,_,_,_ = cryst.get_unit_cell().parameters()
a_var = linspace( 0.975*a, a*1.025, 25)
b_var = linspace( 0.975*b, b*1.025, 25)
c_var = linspace( 0.975*c, c*1.025, 25)
c_var
b_var
a_var
Cabc = []
for aa,bb,cc in zip(a_var, b_var, c_var):
    Bnew = sqr( (aa,0,0, 0,bb,0, 0,0,cc)).inverse()
    C = deepcopy(cryst)
    C.set_B(Bnew)
    Cabc.append(C)
    
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    params.append( param)
    
utils.save_flex(params, "increase_abc_spec4.5.pkl")
utils.save_flex(params, "increase_abc_spec4p5.pkl")
get_ipython().magic(u'hist')
en_spec
flux
spotdata = np.load("crystR.spotdata.pkl.npz")
roi_pp = spotdata['roi_pp'][()]
counts_pp = spotdata["counts_pp"][()]
counts_pp.keys()
counts_pp
len(counts_pp)
counts_pp.shape
roit_pp
roi_pp
roi_pp[0]
roi_pp.keys()
roi_pp[0]
roi_pp.sj
roi_pp.shape
roi_pp[0]
roi_pp[1]
roi_pp[2]
get_ipython().magic(u'clear ')
det
detector
detector.to_dict
detector.to_dict()
det
det = detector
det
det[0]
#@det.add_panel
#det.rotate_around_origin
dir(det)
det.to_dict()
D = det.to_dict()
D.keys()
D['hierarchy']
from dxtbx.model.detector import Detector
Detector()
D2 = Detector()
for i in [32,34,50]:
    D2.add_panel( detector[i])
    
D2
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 500
    param['mos_spread'] = 0.125
    params.append( param)
    
    
    
utils.save_flex(params, "increase_abc_spec4p5_mos.pkl")
en_spec
flux
en_spec
flux
get_ipython().magic(u'clear ')
get_ipython().magic(u'ls ')
#Ncells_abc = [(i,i,i) for i in range(2,75)]
250 * 100
#Ncells_abc = [(i,i,i) for i in range(5,80)]
@#load("")
a= load("spec_gamma4.npz")['en'][[0,-1]]
(a[1] - a[0]) / 8944
a= load("spec_gamma2.npz")['en'][[0,-1]]
(a[1] - a[0]) / 8944
a= load("spec_gamma.npz")['en'][[0,-1]]
(a[1] - a[0]) / 8944
len(a)
a
a= load("spec_gamma2.npz")['en']
len(a)
get_ipython().magic(u'clear ')
#Ncells_abc = [(i,i,i) for i in [arange(]
arange(2,20,3)
arange(2,23,3)
arange(5,33,3)
arange(15,25,1)
arange(10,30,2)
arange(10,31,2)
Ncells_abc = [(i,i,i) for i in arange(10,31,2)]
a
a= load("spec_gamma2.npz")['en']
a
a[::3]
a[::3].shape
en_spec = a[::3]
#flux = flux[::3]
flux
b= load("spec_gamma2.npz")['flux']
b
flux[::3]
b[::3]
b[::3]
flux = b[::3]
flux
en_spec
np.savez("spec_gamma6.npz", en=en_spec, flux=flux)
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
    
utils.save_flex(params, "increase_Ncells_gamm6_mos18.pkl")
a_var
#a_var = linspace( 0.975*a, a*1.025, 25)
en_spec
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
    
utils.save_flex(params, "increase_Ncells_gamm6_mos18.pkl")
#a_var = linspace( 0.975*a, a*1.025, 25)
en_spec*0.975
linspace(0.975,1.025,25)
scale_facs = linspace(0.975,1.025,25)
scale_facs
[en_spec*s for s in scale_facs]
dlambda = [en_spec*s for s in scale_facs]
params = []
for i,spec in enumerate(dlambda):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
    
    
utils.save_flex(params, "increase_lambda_gamm6_mos18.pkl")
CrotZ, CrotY, CrotX = [],[],[]
from scitbx.matrix import col, sqr
x = col(1,0,0)
y = col(0,1,0)
z = col(0,0,1)

for deg in linspace(-0.3,0.3,25):
    Bnew = sqr( (aa,0,0, 0,bb,0, 0,0,cc)).inverse()
    
    C = deepcopy(cryst)
    C.set_B(Bnew)
    Cabc.append(C)
    
xR = x.axis_and_angle_as_r3_rotation_matrix
yR = y.axis_and_angle_as_r3_rotation_matrix
zR = z.axis_and_angle_as_r3_rotation_matrix
x = col(1,0,0)
from scitbx.matrix import sqr,col
x = col(1,0,0)
x = col((1,0,0))
y = col((0,1,0))
z = col((0,0,1))
#x.axis_and_angle_as_r3_rotation_matrix
CrotZ, CrotY, CrotX = [],[],[]
for deg in linspace(-0.3,0.3,25):
    Rx = x.axis_and_angle_as_r3_derivative_wrt_angle(deg,deg=True)
    Ry = y.axis_and_angle_as_r3_derivative_wrt_angle(deg,deg=True)
    Rz = z.axis_and_angle_as_r3_derivative_wrt_angle(deg,deg=True)
    U = sqr(cryst.get_U())
    B = sqr(cryst.get_B())
    UnewX = Rx*U
    UnewY = Ry*U
    UnewZ = Rz*U
    
    CX = deepcopy(cryst)
    CY = deepcopy(cryst)
    CZ = deepcopy(cryst)
    CX.set_U(UnewX)
    CY.set_U(UnewY)
    CZ.set_U(UnewZ)
    CrotX.append(CX)
    CrotY.append(CY)
    CrotZ.append(CZ)
    
UnewX
Rx
U
U
U.as_numpy_array()
linalg.det(U.as_numpy_array())
linalg.det(UnewX.as_numpy_array())
UnewX
Rx
linalg.det(Rx.as_numpy_array())
CrotZ, CrotY, CrotX = [],[],[]
for deg in linspace(-0.3,0.3,25):
    Rx = x.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    Ry = y.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    Rz = z.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    U = sqr(cryst.get_U())
    B = sqr(cryst.get_B())
    UnewX = Rx*U
    UnewY = Ry*U
    UnewZ = Rz*U
    
    CX = deepcopy(cryst)
    CY = deepcopy(cryst)
    CZ = deepcopy(cryst)
    CX.set_U(UnewX)
    CY.set_U(UnewY)
    CZ.set_U(UnewZ)
    CrotX.append(CX)
    CrotY.append(CY)
    CrotZ.append(CZ)
    
params = []
for i,C in enumerate(CrotZ):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
    
CrotZ, CrotY, CrotX = [],[],[]
for deg in linspace(-0.4,0.4,25):
    Rx = x.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    Ry = y.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    Rz = z.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
    U = sqr(cryst.get_U())
    B = sqr(cryst.get_B())
    UnewX = Rx*U
    UnewY = Ry*U
    UnewZ = Rz*U
    
    CX = deepcopy(cryst)
    CY = deepcopy(cryst)
    CZ = deepcopy(cryst)
    CX.set_U(UnewX)
    CY.set_U(UnewY)
    CZ.set_U(UnewZ)
    CrotX.append(CX)
    CrotY.append(CY)
    CrotZ.append(CZ)
    
params = []
for i,C in enumerate(CrotZ):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
utils.save_flex(params, "increase_rotZ_gamm6_mos18.pkl")
params = []
for i,C in enumerate(CrotY):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
utils.save_flex(params, "increase_rotY_gamm6_mos18.pkl")
params = []
for i,C in enumerate(CrotX):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
utils.save_flex(params, "increase_rotX_gamm6_mos18.pkl")
get_ipython().magic(u'ls -lrt')
Ncells_abc = [(i,i,i) for i in arange(5,15,1)]
Ncells_abc
Ncells_abc = [(i,i,i) for i in arange(5,16,1)]
Ncells_abc
arange(1,21,2)
arange(2,21,2)
Ncells_abc = [(i,i,i) for i in arange(2,21,2)]
Ncells_abc = [(i,i,i) for i in arange(3,31,3)]
Ncells_abc = [(i,i,i) for i in arange(2,21,2)]
Ncells_abc
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
    
Cabc
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'tophat'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 500
    param['mos_spread'] = 0.125
    params.append( param)
    
    
    
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
utils.save_flex(params, "increase_Ncells_gamm6_mos18.pkl")
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
    
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = en_spec
    param['FLUX']= flux
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append( param)
    
    
    
utils.save_flex(params, "increase_abc_gamm6_mos18.pkl")
get_ipython().magic(u'ls -lrt')
a = load("spec_gamma2.npz")['en']
a
#b = 2*fpA/Yb_f0
b = load("spec_gamma2.npz")['flux']
b
#spec = linspace(8944 - dlam, 8944+dlam, 5)
dlambda = [a*s for s in scale_facs]
dlambda
scale_facs
a
(a[1] - a[0]) / 8944
a
(a[-1] - a[0]) / 8944
spec = linspace(8944 - dlam, 8944+dlam, 5)
params = []
for i,spec in enumerate(dlambda):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
    
utils.save_flex(params, "increase_lambda_gamm2_mos1.pkl")
params = []
for i,spec in enumerate(dlambda):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
params = []
for i,spec in enumerate(dlambda):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 100
    param['mos_spread'] = 0.18
    params.append(param)
    
    
    
utils.save_flex(params, "increase_lambda_gamm2_mos1.pkl")
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append(param)
    
get_ipython().magic(u'save params 300-429')
para
en_spec = a
flux =b
en_spec
flux
params = []
for i,Ncells in enumerate(Ncells_abc):
    param = {}
    param['crystal'] = cryst
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = Ncells
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append(param)
    
utils.save_flex(params, "increase_Ncells_gamm6_mos18.pkl")
utils.save_flex(params, "increase_Ncells_gamm2_mos1.pkl")
params = []
for i,C in enumerate(Cabc):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
    
    
    
utils.save_flex(params, "increase_abc_gamm2_mos1.pkl")
params = []
for i,C in enumerate(CrotZ):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
    
    
    
utils.save_flex(params, "increase_rotZ_gamm6_mos18.pkl")
params = []
for i,C in enumerate(CrotX):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
utils.save_flex(params, "increase_rotX_gamm2_mos18.pkl")
    
    
params = []
for i,C in enumerate(CrotZ):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
utils.save_flex(params, "increase_rotZ_gamm2_mos18.pkl")
    
    
params = []
for i,C in enumerate(CrotY):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
utils.save_flex(params, "increase_rotY_gamm2_mos1.pkl")
    
    
params = []
for i,C in enumerate(CrotZ):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
utils.save_flex(params, "increase_rotZ_gamm2_mos1.pkl")
    
    
params = []
for i,C in enumerate(CrotX):
    param = {}
    param['crystal'] = C
    param['ENERGIES'] = list(en_spec)
    param['FLUX']= list(flux)
    print C.get_unit_cell().parameters()
    param['shape'] = 'gauss'
    param['Ncells_abc'] = (10,10,10)
    param['beam'] = beamA
    param['order'] = i
    param['Nmos'] = 50
    param['mos_spread'] = 0.1
    params.append( param)
utils.save_flex(params, "increase_rotX_gamm2_mos1.pkl")
    
    
get_ipython().magic(u'ls -lrt')
get_ipython().magic(u'ls *gamm2_mos1.pkl')
