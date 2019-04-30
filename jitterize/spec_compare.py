
import numpy as np
from scipy.signal import correlate2d 
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline


def ellipse_structure(a,b):
    a = float(a)
    b = float(b)
    size = max(a,b)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
    return np.sqrt(i**2+ a**2 * j**2/ b**2) <= a


def get_spectrum(spec):

    # energy as a function of fast-axis pixel (0-1024)
    #x1= 136
    #x2 = 759
    #e1 = 8944
    #e2 = 9034.7
    #Efit = polyfit((x1,x2),(e1,e2), deg=1)
    Efit = np.array([1.45585875e-01, 8.92420032e+03])
    # (use with polyval) 
    
    bg_rows = 20  # number of rows in image where we expect only background
    nsig = 5  # how many 
    ev_width=5
    V = 22  # height of ellipse matcher
    H = 9  # width of ellipse matcher 
    
    # template match on the camera
    foot = ellipse_structure(V/2,H/2)  # ellipse footprint
    matcher = correlate2d( spec, foot, mode='same') / np.sum( foot)
    bg_pixels = np.hstack( (matcher[:bg_rows].ravel(), 
                            matcher[-bg_rows:].ravel() ))

    # make a mask for labeling (not critical)
    M = matcher > bg_pixels.max()   # mask of bright pixels
    labimg, nlab = ndimage.label(M)  # connected regions in the mask
    
    # we compute the center of intensity and intensity of the connected
    # regions, consider doing this for just the X largest regions
    # we will fit a line to these data
    if nlab > 1:  # are there any regions of data for the fit?
        y,x = np.array(ndimage.center_of_mass( matcher, labimg, range( 1,nlab+1))).T 
        I = ndimage.maximum(matcher, labimg, np.arange(1,nlab+1)) 

        # line fit to connected regions
        # we will interpolate along this line
        pfit = np.polyfit(x,y,1,w=I/I.max())
    else:
        pfit = array([-7.29833699e-02,  1.59887735e+02])  #  use a predetermined fit.. 

    # lines to interpolate along
    nn=V*2  # how many lines
    xdata = np.arange(spec.shape[1])
    ydatas = []
    for i in np.arange(-nn,nn+1,1):
        ydatas.append( np.polyval( pfit + np.array([0,i]), xdata ) ) 

    # interpolate
    rbs = RectBivariateSpline( np.arange(spec.shape[0]), 
                            np.arange(spec.shape[1]), matcher)
    evals4 = []  # evaluations
    for y in ydatas:
        evals4.append( rbs.ev(y, xdata))

    bg_sig = bg_pixels.std()
    bg_mean = bg_pixels.mean() 
    evals4 = np.array( evals4)
    evals4_ma = np.ma.masked_where(evals4 < bg_mean+bg_sig*nsig , evals4)
    raw_spec = np.sum( evals4_ma,axis=0)

    Edata = np.polyval( Efit,xdata)
    en_bins = np.arange( Edata[0],Edata[-1]+1, ev_width)

#   now we can bin 
    spec_hist = np.histogram( Edata , en_bins, weights=raw_spec)[0]

    return en_bins[1:]*.5  + en_bins[:-1]*.5, spec_hist , matcher, pfit


# if on psana
def get_spec_data(f):
    idx = int(f.split('_')[1])
    ev = psanaR.event( loader.times[idx])
    spec_img = spec.image(ev)
    return get_spectrum(spec_img)


import glob
from scipy.spatial import distance
import numpy as np
from scitbx.array_family import flex
from cxid9114.spots import spot_utils
from cxid9114 import utils

#run = int(sys.argv[1])

fnames = glob.glob("results/run*/*resid.pkl")


res_lims = np.array([         
                inf,  31.40920442,  15.7386434 ,  10.53009589,
         7.93687927,   6.38960755,   5.36511518,   4.63916117,
         4.0996054 ,   3.68413423,   3.35535065,   3.08945145,
         2.87056563,   2.68770553,   2.53302433,   2.40077038,
         2.28663331,   2.1873208 ,   2.10027621,   2.02348524,
         1.95534054,   1.89454478,   1.84004023,   1.79095656])


dq_min = 0.003

Nf = len(fnames)
dist_out3 = []
all_AnotB = []
all_BnotA = []
all_AandB = []
for i_f, f in enumerate(fnames):
    d = utils.open_flex(f)
    idxA = d['residA']['indexed'] 
    idxB = d['residB']['indexed']
    idxAB = np.logical_or(idxA, idxB)
    
    dQA = np.array(d['residA']['dQ']) <= dq_min
    dQB = np.array(d['residB']['dQ']) <= dq_min
    good = idxAB * dQA * dQB
  
    AnotB = np.logical_and(idxA, ~idxB)
    BnotA = np.logical_and(idxB, ~idxA)
    AandB = np.logical_and( idxA, idxB)
    
    resAnotB = d['residA']['res'][AnotB]
    resBnotA = d['residB']['res'][BnotA]
    
    resAandB = np.mean( [d['residA']['res'][AandB], 
                d['residB']['res'][AandB]], axis=0)
    
    all_AnotB.append( resAnotB)
    all_BnotA.append( resBnotA)
    all_AandB.append( resAandB)

    nA = np.logical_and( idxA, ~idxB).sum()
    nB = np.logical_and( ~idxA, idxB).sum()
    nAB = np.logical_and( idxA, idxB).sum()
    
    R = d['refls_strong']

    Nref = len(R)
    Nidx = sum( good)
    frac_idx = float(Nidx) / Nref
    
    Rpp = spot_utils.refls_by_panelname(R.select( flex.bool(good))) 
    nC = 0
    for pid in Rpp:
        r = Rpp[pid]
        x,y,_ = spot_utils.xyz_from_refl(r)
        C = distance.pdist(zip(x,y))
        nC += np.sum( (1 < C) & (C < 7))
    run_num = int(f.split("/")[1].split("run")[1])
    shot_idx = int(f.split("_")[1])
    dist_out3.append( [nC, i_f, d['rmsd_v1'], f, run_num, shot_idx, 
            frac_idx, Nref, Nidx, nA, nB, nAB] )

    print i_f, Nf




#
#
#
all_split_spots = []

for i_f, f in enumerate( fnames):
    all_h = set(map( tuple, vstack(( all_HiA[i_f], all_HiB[i_f]))) )
    
    # for each h, 
    # check that it was indexed by both colorsets
    # then check that the spot is actually not overlapping (e.g. spaced out)
    # for both color sets
    split_spots = []
    for h in all_h:
       
        #HiA and HiB are the fractional HKL for each reflection, for each color
        where_idxA = np.where(np.sum(np.abs(all_HiA[i_f] - h), axis=1)==0)[0]
        where_idxB = np.where(np.sum(np.abs(all_HiB[i_f] - h), axis=1)==0)[0]
        
        # make sure this h is possibly indexed by both colors
        if not where_idxA.size or not where_idxB.size:
            continue
        
        # in multiple spots gave the same h
        # find the spot with the smallest residual h
        frac_hA = []
        for i in where_idxA:
            frac_hA.append( all_resA[i_f][i])
        iA = np.argmin(frac_hA)
        
        frac_hB = []
        for i in where_idxB:
            frac_hB.append( all_resB[i_f][i])
        iB = np.argmin(frac_hB)

        # check that this is actually indexed by both colors (e.g. reasonable hkl residual)
        if not frac_hB[iB] < .15  and frac_hA[iA] < .15:
            #print("not indexed by both")
            continue
       
        # now lastly, make sure the spots are separated by 
        # ensuring they are not the same spot!
        if where_idxA[iA] == where_idxB[iB]:
            #print("They are same reflection")
            continue 
        # if we made it this far, likely the h is indexed
        # by both colors, yet on different places on the detector,
        # meaning its a split-spot
        split_spots.append(h)
    
    all_split_spots.append( split_spots)
        
# these spots definitely have two colors splits
def_twocol = [ fnames[i] for i,a in enumerate(all_split_spots) if len(a) >=40]




from cxid9114.sim import sim_utils
from cxid9114 import parameters

simsAB = sim_utils.sim_twocolors2(
    cryst, detector, beamA, [5000, None],
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    [1e14, 1e14], pids=None, Gauss=False, oversample=2,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=0.0, verbose=2)




################################
#
# SIMULATION of MANY vs 2 colors
#
################################
from cxid9114.sim import sim_utils
from cxid9114 import utils

spec = np.load("specs.npz")['spec16219']
en, en_intens, _ = get_spectrum( spec)

# select the color channels which have signal
spec_has_signal=  en_intens>0

en = en[spec_has_signal]
en_intens = en_intens[ spec_has_signal]

# normalize the intensity of each color channel to a flux value
en_flux = en_intens / en_intens.sum() * 1e14

# set some structure factors
StrucFacts = [None]*len(en)
StrucFacts[0] = 5000 # default value, to be applied to all others

D = utils.open_flex('ref1_det.pkl')  # load a detector
B = utils.open_flex('ref3_beam.pkl')  # load a beam
C = np.load('shot16219_pan49.npz')['crystal2'].item()  # load a crystal
dimg = np.load('shot16219_pan49.npz')['dimg']  # data image as reference
PID = 49  # detector panel ID


Sfull = sim_utils.sim_twocolors2(
    C, D, B, StrucFacts,  en, en_intens,
    pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=.05, verbose=2)

Sfull2 = sim_utils.sim_twocolors2(
    C, D, B, StrucFacts,  en, en_intens,
    pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=.05, verbose=2)

composite = np.sum([ Sfull[s][0] for s in Sfull],0)
composite2 = np.sum([ Sfull2[s][0] for s in Sfull],0)

Stwo = sim_utils.sim_twocolors2(
    C, D, B, [5000, None],[8944, 9034.7],[1e14,1e14],
    pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(20, 20, 20), mos_dom=20, mos_spread=.05, verbose=2)

Stwo2 = sim_utils.sim_twocolors2(
    C, D, B, [5000, None],[8944, 9034.7],[1e14,1e14],
    pids=[PID], Gauss=True, oversample=5,
    Ncells_abc=(10, 10, 10), mos_dom=20, mos_spread=.05, verbose=2)


np.savez("shot16219_49_manyVtwo", Sfull=Sfull, Sfull2=Sfull2,Stwo=Stwo, Stwo2=Stwo2)


########
#
# PLOT
#
########
def norm_img(img, scale=None):
    
    dimg = img.copy()
    
    dimg -= dimg.min()
    dimg /= dimg.max()
    if scale is not None:
        dimg [ dimg > scale] = 1
    return dimg

def cscale(img, contrast=0.1):
    m90 = np.percentile( img, 90)
    return np.min( [np.ones(img.shape), 
        contrast * img/m90],axis=0)


data = np.load("shot16219_49_manyVtwo.npz", encoding="bytes")
img1 = np.load('shot16219_pan49.npz', encoding="bytes")['dimg']
img2 = data['Stwo2'].item()
img3 = data['Sfull'].item()
img4 = data['Sfull2'].item()
xlim=(72.51932229236323, 186.59286106895075)
ylim= (157.2904559149981, 69.26637580300587)


data = np.load('shot924_pan50.npz', encoding="bytes")
img1 = data["dimg"]
img2 = data["simsAB11"].item()
img3 = data["simsAB112"].item()
img4 = data["simsAB1122"].item()


#xlim= (37.50588686911298, 181.4113359181758)
#ylim=(137.22284658674218, 29.282620871857546)

data = np.load('shot16219_pan49.npz', encoding="bytes")
img1 = data["dimg"]
img2 = data["simsAB11"].item()
img3 = data["simsAB112"].item()
img4 = data["simsAB1122"].item()
xlim=(77.51414652705894, 87.1350634277985)
ylim=(143.39734163512327, 123.81415058013239)


compos = lambda S: np.sum([ S[channel][0] for channel in S],0)
#compos = lambda S: S #np.sum([ S[channel][0] for channel in S],0)

fig,axs = subplots(2,2)
#axs = axs.flatten()

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[1,0]
ax4 = axs[1,1]

showDarg = dict(vmin=0,vmax=1,cmap='Greys', aspect='auto')
showarg = dict(vmin=0,vmax=1,cmap='Greys', aspect='auto')
grid_arg = dict(ls='--', alpha=.75)

sim_scale= .1

ax1.clear()
ax1.imshow( cscale(img1, contrast=0.2 ) , **showDarg)
ax1.grid(1,**grid_arg) 
ax1.xaxis.set_ticklabels([])
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
#ax1.set_xlim(xlim)

ax2.clear()
ax2.imshow( norm_img(compos(img2), scale=sim_scale) , **showarg)
ax2.grid(1,**grid_arg) 
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
#ax2.set_xlim(xlim)

ax3.clear()
ax3.imshow( norm_img(compos(img3),scale=sim_scale) , **showarg)
ax3.grid(1,**grid_arg) 
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)

ax4.clear()
ax4.imshow( norm_img(compos(img4),scale=sim_scale) , **showarg)
ax4.grid(1,**grid_arg) 
ax4.yaxis.set_ticklabels([])
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)

cax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
fig.colorbar(ax4.images[0], cax=cax)


##############3
################
###########

data = np.load('shot924_pan50.npz', encoding="bytes")
img1 = data["dimg"]
img2 = data["simsAB11"].item()
img3 = data["simsAB112"].item()
img4 = data["simsAB1122"].item()


showDarg = dict(vmin=0,vmax=1,cmap='Greys', aspect='auto')
showarg = dict(vmin=-1,vmax=1,cmap='Greys', aspect='auto')
grid_arg = dict(ls='--', alpha=.75)

sim_scale=.02
xlim= (155.89746209227656, 177.4280853099633)
ylim= (102.83494995019586, 33.394048062311185)


subplot(131)
ax1 = gca()
ax1.clear()
#ax1.set_xlim(xlim)
#ax1.set_ylim(ylim)
ax1.imshow( cscale(img1, contrast=0.2 ) , **showDarg)
ax1.grid(1,**grid_arg) 
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)


subplot(132)
ax2 = gca()
ax2.clear()

maskA = img3[0][0] > 1
maskB = img3[1][0] > 1

imgA = norm_img( img3[0][0] , scale=.02) 
imgB = norm_img( img3[1][0] , scale=.02) 


#redImg = imgA * maskA * (~maskB)
#blueImg = imgB * maskB * (~maskA)
#purpImg = (.5*imgA+.5*imgB)* maskA * maskB

#imgA[ ~redImg] = np.nan
#imgB[ ~blueImg] = np.nan
#purpImg[ purpImg==0] = np.nan


# version 1
#redImg[ redImg ==0] = np.nan
#blueImg[ blueImg ==0] = np.nan
#purpImg[ purpImg ==0] = np.nan
#purpImg[ purpImg > 0] = 1.
#
#ax2.imshow(blueImg, 
#    aspect='auto', vmin=-1,vmax=1,cmap='bwr_r')
#ax2.imshow(redImg, 
#    aspect='auto', vmin=-1,vmax=1,cmap='bwr') #, alpha=0.5)
#ax2.imshow(purpImg, 
#    aspect='auto', vmin=0,vmax=1,cmap='cool') #, alpha=0.5)
#ax2.grid(1,**grid_arg) 
#ax2.yaxis.set_ticklabels([])
#ax2.set_xlim(xlim)
#ax2.set_ylim(ylim)

# version2
imgB[~maskB] = np.nan
imgA[~maskA] = np.nan
ax2.imshow(imgB, 
    aspect='auto', vmin=-1,vmax=1,cmap='bwr_r')
ax2.imshow(imgA, 
    aspect='auto', vmin=-1,vmax=1,cmap='bwr', alpha=0.5)
ax2.grid(1,**grid_arg) 
ax2.yaxis.set_ticklabels([])
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)


#

subplot(133)
ax3 = gca()
ax3.clear()

maskA = img4[0][0] > 1
maskB = img4[1][0] > 1

imgA = norm_img( img4[0][0] , scale=.02) 
imgB = norm_img( img4[1][0] , scale=.02) 


imgB[~maskB] = np.nan
imgA[~maskA] = np.nan
ax3.imshow(imgB, 
    aspect='auto', vmin=-1,vmax=1,cmap='bwr_r')
ax3.imshow(imgA, 
    aspect='auto', vmin=-1,vmax=1,cmap='bwr', alpha=0.5)
ax3.grid(1,**grid_arg) 
ax3.yaxis.set_ticklabels([])
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)



subplot(133)
ax3 = gca()
ax3.clear()
ax3.imshow( norm_img(img4[1][0],scale=sim_scale) , **showarg)
ax3.grid(1,**grid_arg) 
ax3.yaxis.set_ticklabels([])
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)








#################





###############
# SCRATCH BELOW
################


def circular_structure(radius):
    size = radius*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
    return np.sqrt(i**2+j**2) <= radius

def ellipse_structure(a,b):
    a = float(a)
    b = float(b)
    size = max(a,b)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
    return np.sqrt(i**2+ a**2 * j**2/ b**2) <= a


def ellipse_ring_structure(a,b, w):
    a = float(a)
    b = float(b)
    a2 = a+w
    b2 = b+w
    
    size = max(a2,b2)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)

    inner =  np.sqrt(i**2+ a**2 * j**2/ b**2) <= a
    outer =  np.sqrt(i**2+ a2**2 * j**2/ b2**2) <= a2

    return np.logical_and( ~inner, outer)



def ellipse_structure2(a0,b0,a,b,w):
    size = max(a0,b0)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
   
    a2 = a+w
    b2 = b+w
    
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)

    inner =  np.sqrt(i**2+ a**2 * j**2/ b**2) <= a
    outer =  np.sqrt(i**2+ a2**2 * j**2/ b2**2) <= a2

    ring =  np.logical_and( ~inner, outer)

    disk =  np.sqrt(i**2+ a0**2 * j**2/ b0**2) <= a0

    return ring, disk



def ellipse_ring_structure(a,b, w):
    a = float(a)
    b = float(b)
    a2 = a+w
    b2 = b+w
    
    size = max(a2,b2)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)

    inner =  np.sqrt(i**2+ a**2 * j**2/ b**2) <= a
    outer =  np.sqrt(i**2+ a2**2 * j**2/ b2**2) <= a2

    return np.logical_and( ~inner, outer)

def ring_kernel(r1,r2):
    size = max(r1,r2)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
    return np.logical_and(r1 < np.sqrt(i**2+j**2), np.sqrt(i**2+j**2)  <= r2)

from scipy.signal import correlate2d 
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

V = 22
H = 9
foot = ellipse_structure(V/2,H/2)
matcher = correlate2d( spec, foot, mode='same') / np.sum( foot)
M = matcher > matcher[:10].max()  # definite background region
labimg, nlab = ndimage.label( M ) 
y,x=np.array(ndimage.center_of_mass( matcher, labimg, range( 1,nlab+1))).T 
I = ndimage.maximum(matcher, labimg, arange(1,nlab+1)) 

pfit = np.polyfit(x,y,1,w=I/I.max())
# pfit = array([-7.81150402e-02,  1.56354575e+02])

nn=V*2
xdata = np.arange(1024)
ydatas = []
for i in np.arange(-nn,nn+1,1):
    ydatas.append( polyval( pfit + np.array([0,i]), xdata ) ) 


rbs = RectBivariateSpline( np.arange(256), np.arange(1024), matcher)
evals4 = []
for y in ydatas:
    evals4.append( rbs.ev(y, xdata))

bg_sig = std(list(matcher[:20].ravel()) + list(matcher[-20:].ravel())) 
bg_mean = mean(list(matcher[:20].ravel()) + list(matcher[-20:].ravel())) 
evals4 = np.array( evals4)
evals4_ma = ma.masked_where(evals4 < bg_mean+sig*5 , evals4)
raw_spec = np.sum( evals4_ma,axis=0)

# from the plot of np.sum(evals4,axis=0)
# this bit we can fix later by 
# refining the beam energy of each shot
# I think the seeder acts as a monochromator 
# so we need just determine the two 
# energies precisely ...
Efit = polyfit((x1,x2),(e1,e2), deg=1)
Edata = polyval( Efit,xdata)
ev_width=5
en_bins = arange( Edata[0],Edata[-1]+1, ev_width)

# now we can bin 
spec_hist = histogram( Edata , en_bins, weights=raw_spec)[0]


# code for plottin the footprint
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#cmap = cm.get_cmap('Greys', 2)
#imshow( foot, cmap=cmap)
#colorbar()
#imshow( foot, cmap=cmap, vmin=0, vmax=1)
#colorbar()
#clf()
#imshow( foot, cmap=cmap, vmin=0, vmax=1)
#cbar = colorbar()
#cbar.ax.set_yticks([.25, .75])
#cbar.ax.set_yticklabels(['0', '1'])
#cbar.ax.yaxis.set_ticks?
#cbar.ax.yaxis.set_ticks( [.25,.75])
#ax = gca()
#ax.tick_params(labelsize=16)
#cbar.ax.tick_params(labelsize=16)
#cbar.ax.tick_params(labelsize=20)
#ax.tick_params(labelsize=20)
#ax.tick_params(labelsize=22)
#cbar.ax.tick_params(labelsize=22)


#imshow( foot, cmap=cmap)
#colorbar()
#imshow( foot, cmap=cmap, vmin=0, vmax=1)
#colorbar()
#clf()
#imshow( foot, cmap=cmap, vmin=0, vmax=1)

cmap = cm.get_cmap('prism', 2)
cbar = colorbar()
cbar.ax.set_yticks([.25, .75])
cbar.ax.set_yticklabels(['vacant', 'occupied'])



def add_asic_to_ax( ax, I, p,fs, ss=None,s="",s_size=12,s_color='c', **kwargs):
    """
    View along the Z-axis (usually the beam axis) at the detector

    vectors are all assumed x,y,z
    where +x is to the right when looking at detector
          +y is to down when looking at detector
          z is along cross(x,y) 
   
    Note: this assumes slow-scan is prependicular to fast-scan

    Args
    ====
    ax, matplotlib axis
    I, 2D np.array
        panels panel
    p, corner position of first pixel in memory
    fs, fast-scan direction in lab frame
    ss, slow-scan direction in lab frame, 
    s , some text
    """
    # first get the angle between fast-scan vector and +x axis
    ang = np.arccos(np.dot(fs, [1, 0, 0]) / np.linalg.norm(fs) )
    ang_deg = ang * 180 / np.pi    
    if fs[0] <= 0 and fs[1] < 0:
        ang_deg = 360 - ang_deg
    elif fs[0] >=0 and fs[1] < 0:
        ang_deg = 360-ang_deg

    im = ax.imshow(I, origin="upper",
            extent=(p[0], p[0]+I.shape[1], p[1]-I.shape[0], p[1]), 
            **kwargs)
    trans = mpl.transforms.Affine2D().rotate_deg_around( p[0], p[1], ang_deg) + ax.transData
    im.set_transform(trans)
    
    # add label to the axis
    panel_cent = .5*fs*I.shape[1] + .5*ss*I.shape[0] + p 
    _text = ax.text(panel_cent[0], panel_cent[1], s=s,size=s_size, color=s_color)


def plot_dials_cspad(det, images, 
            s_size=12, s_color='c', **kwargs):
    figure()
    ax = gca()
    pixsz = det[0].get_pixel_size()[0]
    f = array([d.get_fast_axis() for d in D]) #/ pixsz
    s = array([d.get_slow_axis() for d in D]) #/ pixsz
    p = array([d.get_origin() for d in D]) / pixsz 

    for i,(node, data) in enumerate(zip(det, images)):
        add_asic_to_ax( 
            ax=ax,
            I=data,
            p=p[i],
            ss=s[i],
            fs=f[i],
            s=str(i),
            s_size=s_size,
            s_color=s_color,
            **kwargs)

    ax.set_xlim(p[:,0].min(), p[:,0].max())
    ax.set_ylim(p[:,1].min(), p[:,1].max())
    return ax




get_res = lambda node: \
    array([node.get_resolution_at_pixel( beamA.get_s0(), (i,j)) for i in range(194) for j in range(185)]).reshape( (194, 185))

get_theta = lambda node: \
    array([node.get_two_theta_at_pixel( beamA.get_s0(), (i,j)) for i in range(194) for j in range(185)]).reshape( (194, 185))/2.

resD = np.array([get_res(d) for d in D])
thetaD = np.array([get_theta(d) for d in D])
R = np.tan( thetaD * 2 ) * 0.125
halfpx = .10992 /2.
wave_wide = 2 * resD * sin( arctan((R-halfpx)/.125) *.5)
wave_small = 2 * resD * sin( arctan((R+halfpx)/.125) *.5)
wave_width =  parameters.ENERGY_CONV/wave_wide - parameters.ENERGY_CONV/wave_small
wave_width = np.array([ W.T for W in wave_width])







