
import numpy as np
from scipy.signal import correlate2d 
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
    #M = matcher > bg_pixels.max()   # mask of bright pixels
    #labimg, nlab = ndimage.label(M)  # connected regions in the mask
    
    # we compute the center of intensity and intensity of the connected
    # regions, consider doing this for just the X largest regions
    # we will fit a line to these data
    # if nlab > 1:  # are there any regions of data for the fit?
    #    y,x = np.array(ndimage.center_of_mass( matcher, labimg, range( 1,nlab+1))).T 
    #    I = ndimage.maximum(matcher, labimg, np.arange(1,nlab+1)) 

        # line fit to connected regions
        # we will interpolate along this line
    #    pfit = np.polyfit(x,y,1,w=I/I.max())
    #else:
    pfit = np.array([-7.29833699e-02,  1.59887735e+02])  #  use a predetermined fit.. 

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
    spec_hist = np.histogram(Edata, en_bins, weights=raw_spec)[0]

    return en_bins[1:]*.5 + en_bins[:-1]*.5, spec_hist , matcher, raw_spec


# if on psana
def get_spec_data(f):
    idx = int(f.split('_')[1])
    ev = psanaR.event(loader.times[idx])
    spec_img = spec.image(ev)
    return get_spectrum(spec_img)

