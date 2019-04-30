import numpy as np
import matplotlib as mpl
import pylab as plt
import numpy as np


def cspad_geom_splitter(psf, returned_units='mm'):
    """
    Splits the psana geometry from 32 panels to 64 panels
    because of the non-uniform pixel gap
    This creates a list of 64 origins, 64 slow-scan directions and 64 fast-scan directions
    though slow and fast scan directions are assumed parallel within
    manufacturing error
    :param psf: tuple or (3 x 32 x 3) array of (origin, fast-scan, slow-scan) for each panel, 
        origin is vector from interaction region to panel corner. 
        These are in psgeom convention (TJL)
        see get_psf for any cspads geometryAccess instance <PSCalib.GeometryAccess.GeometryAccess>
    :param returned_units: string of 'mm', 'um', or 'pixels'
    :return: PSF vectors, 64 long
    """
    #geom = cspad.geometry(event)
    origin_64 = np.zeros((64, 3))
    FS_64 = np.zeros_like(origin_64)
    SS_64 = np.zeros_like(origin_64)

    origin_32, SS_32, FS_32 = psf
    for i in range(32):
        # create the origins of each sub asic
        origin_A = origin_32[i]
        shift = 194. * 109.92 + (274.8 - 109.92) * 2.
        unit_f = FS_32[i] / np.linalg.norm(FS_32[i])
        origin_B = origin_A + unit_f * shift

        # save two sub-asics per each of the 32 actual asics
        idx_A = 2 * i
        idx_B = 2 * i + 1
        origin_64[idx_A] = origin_A
        origin_64[idx_B] = origin_B
        FS_64[idx_A] = FS_64[idx_B] = FS_32[i]
        SS_64[idx_A] = SS_64[idx_B] = SS_32[i]

    if returned_units == "mm":  # dials convention
        return origin_64 / 1000., SS_64 / 1000., FS_64 / 1000.,
    elif returned_units == "um":  # psgeom convention
        return origin_64, SS_64, FS_64
    elif returned_units == "m":  # bornagain
        return origin_64/1e6, SS_64/1e6, FS_64/1e6
    elif returned_units == "pixels":  # crystfel convention
        return origin_64 / 109.92, SS_64 / 109.92, FS_64 / 109.92


def  cspad_data_splitter(data):
    """
    splits 185 x 388 asics into 184 x 194 panels

    :param data:  32 x 185 x 388 cspad data
    :return: 64 x 185 x 194 cspad data
    """
    asics64 = []
    for split_asic in [(asic[:, :194], asic[:, 194:]) for asic in data]:
        for sub_asic in split_asic:  # 185x194 arrays
            asics64.append(sub_asic)
    return asics64


def add_asic_to_ax( ax, I, p,fs, ss=None,s="",patches=[], **kwargs):
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
    
    panel_cent = .5*fs*I.shape[1] + .5*ss*I.shape[0] + p 
   
    offset = mpl.transforms.Affine2D().translate(panel_cent[0], panel_cent[1])
    for patch in patches:
        patch.set_transform(trans + offset)
        ax.add_patch(patch)

    # add label to the axis
    _text = ax.text(panel_cent[0], panel_cent[1], s=s, color='c')




def random_rotation(deflection=1.0, randnums=None):
    r"""
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """

    # from
    # http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    vec = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    rot = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    mat = (np.outer(vec, vec) - np.eye(3)).dot(rot)
    return mat.reshape(3, 3)

