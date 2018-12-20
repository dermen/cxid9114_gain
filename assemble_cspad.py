import matplotlib as mpl
import pylab as plt
import numpy as np
import sys

def add_asic_to_ax( ax, I, p,fs,ss=None,s="", **kwargs):
    """
    View along the Z-axis (usually the beam axis) at the detector

    vectors are all assumed x,y,z
    where +x is to the right when looking at detector
          +y is to the down when looking at detector
          +z is along the beam when looking at the detector
   
    Note: this assumes slow-scan is prependicular to fast-scan

    Args
    ====
    ax, matplotlib axis
    I, 2D np.array
        panels panel
    p, corner position of first pixel in memory
    fs, fast-scan direction in lab frame
    ss, slow-scan direction in lab frame, not currently used cause
        assumed to be perpendicular to fs
    """
    # first get the angle between fast-scan vector and +x axis
    ang = np.arccos( np.dot( fs , [1,0,0]) /np.linalg.norm(fs) )
    ang_deg = ang * 180 / np.pi    
    if fs[0] <= 0 and fs[1] < 0:
        ang_deg = 360 - ang_deg
    elif fs[0] >=0 and fs[1] < 0:
        ang_deg = 360-ang_deg

    im = ax.imshow(I, 
            extent=(p[0], p[0]+I.shape[1], p[1]-I.shape[0], p[1]), 
            **kwargs)
    
    trans = mpl.transforms.Affine2D().rotate_deg_around( p[0], p[1], ang_deg) + ax.transData
    _text = ax.text(p[0], p[1],s=s, color='c', transform=ax.transAxes)

    im.set_transform(trans)
    _text.set_transform(trans)

def assemble_cspad(panels, PSF, pixsize=109.92, aspect='equal', a=0,b=0,max_dim=900,
        vmin=None, vmax=None,cmap='gnuplot', show=True):
    """
    panels, list of cspad panels from the psana Detector
        >> panels = psana_detector.calib( event)
    PSF, specifically a tuple of asic corner, asic fast-scan and asic slow-scan dir
        these are defined for every psana cspad detetor object
        >> geom =psana_detector.geometry()
        >> psf = map(np.array, zip(* geom.get_psf() ) )
    pixsize, cspad pixel size in microns
    aspect, pylab plot axis aspect ratio (strings or number)
    a,b beam-intersection point on detector, the psf/psana geometry assumes its at 0,0 so small integer offsets
        could apply
    extent of cspad from 0,0 in either -x,-y,+x,+y, ok to leave as 900 for 32 panel cspad
    vmin,vmax,cmap matplotlib imshow arguments. colorscale-min, colorscale-max and colormap-scheme, respectively

    """ 
    P_,S_,F_ = PSF
    P = P_/pixsize
    S = S_/pixsize
    F = F_/pixsize

    P[:,0] -= a
    P[:,1] -= b

    plt.figure()
    ax = plt.gca()
    ax.set_aspect(aspect)
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim( -max_dim, max_dim)

    if vmin is None and vmax is None:
        m = panels.mean()
        s = panels.std()
        vmin = m-s
        vmax = m+5*s

    imshow_arg = {"vmin":vmin, "vmax":vmax, "interpolation":'none', "cmap":cmap}

    for i in range(32): # 32 panels
        # we split the panels up along the fast-scan
        panel = panels[i] #*mask[i]
        asicA = panel[:,:194]
        asicB = panel[:,194:]
        cornerA = P[i]
        add_asic_to_ax( ax=ax, I=asicA, 
                        p=cornerA, fs=F[i], **imshow_arg)
        
        shift = 194. + (274.8-109.92)*2./pixsize
        unit_f = F[i]/np.linalg.norm(F[i])
        cornerB = cornerA + unit_f*shift
        add_asic_to_ax( ax=ax, I=asicB, 
                        p=cornerB, fs=F[i],s=str(i), **imshow_arg)

    try:
        ax.set_facecolor('dimgray')
    except AttributeError:
        try:
            ax.set_axis_bgcolor("dimgray")
        except AttributeError:
            pass
        pass
    if show:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.1) # necessary pause to update canvas

if __name__ =="__main__":
    panels = np.load(sys.argv[1])
    psf = np.load(sys.argv[2])
    if len(sys.argv) > 3:
        vmin = float( sys.argv[3])
        vmax = float(sys.argv[4])
    else:
        vmin=vmax=None
    assemble_cspad(panels=panels, PSF=psf, vmin=vmin, vmax=vmax, show=True)
