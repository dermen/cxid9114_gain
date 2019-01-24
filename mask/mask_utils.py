import numpy as np
from scipy.ndimage import morphology
import os
import h5py
import pylab as plt

MASK_FILE \
    =  os.path.join(os.path.dirname(__file__), 'masks.hdf5')

from cxid9114 import utils
from dials.array_family import flex


def compress_masks(mask_pickles, outname):
    """
    masks compress rather nicely
    :param mask_pickles:  mask pickle files containing dials masks
    :param outname: hdf5 name
    """
    with h5py.File(outname, "w") as h5:
        for mask_f in mask_pickles:
            mask = utils.open_flex(mask_f)  # these are tuples in dials land
            data = [m.as_numpy_array() for m in mask]
            h5.create_dataset(mask_f, data=data,
                              compression='gzip', compression_opts=9, shuffle=True)


def unload_compress_masks(fname):
    """
    unpacks the fname and stores the masks in dials format
    :param fname:  a compress_mask files created using
    compress_masks
    """
    with h5py.File(fname, "r") as h5:
        for name in h5.keys():
            if os.path.exists(name):
                print("Path %s exists!, skipping" % name)
                continue
            data = h5[name][()]
            if data.dtype != bool:
                continue
            data = tuple([flex.bool(m) for m in data])
            utils.save_flex(data, name)


def load_mask(key):
    with h5py.File(MASK_FILE) as h5:
        if key not in list(h5.keys()):
            raise Exception("Only accepts: %s"%", ".join(h5.keys()))
        mask = h5[key].value
    return mask


def mask_small_regions(gain_data, Ncutoff=1000):
    """ masks certain high-low gain regions
    that have limited number of pixels"""
    mask = np.ones_like( gain_data)
    for i_pan in range(32):
        is_low = gain_data[i_pan]
        is_high = ~is_low
        Nhigh = is_high.sum()
        Nlow = is_low.sum()
        if Nhigh < Ncutoff:
            mask[i_pan][is_high] = False
        if Nlow < Ncutoff:
            mask[i_pan][is_low] = False
    return mask


def details_mask(raw, border=3, plot=False):
    mask = np.ones_like(raw).astype(np.bool)
    mask[:,:,191:196] = False
    mask[:,-border:, : ] = False
    mask[:,:border, : ] = False
    mask[:,:, :border ] = False
    mask[:,:, -border: ] = False

    # plt.hist(raw[ mask==1].ravel(), bins=400)
    # plt.show()
    thresh = float(raw_input("Below which value should I mask pixels? "))

    pixmask = raw < thresh
    struc = morphology.generate_binary_structure(2,2)
    pixmask = np.array([morphology.binary_dilation(pan, struc, iterations=1)
        for pan in pixmask] )

    M = ~pixmask * mask

    if plot:
        fig, ax = plt.subplots( 1,1)
        for i, panel in enumerate(raw): #panel in
            ax.cla()
            ax.imshow( panel*M[i])
            plt.draw()
            plt.pause(.2)
    return M


if __name__=="__main__":
    import psana
    ds = psana.DataSource("exp=cxid9114:run=62")
    events = ds.events()

    det = psana.Detector('CxiDs2.0:Cspad.0')
    dark = det.pedestals(62)
    gain_map = det.gain_mask(62) == 1
    # geom = det.geometry()
    # PSF = map( np.array, zip(*geom.get_psf()))
    raw_sum = None
    counts = 0
    for ev in events:
        if ev is None:
            continue
        raw = det.raw(ev)
        if raw is None:
            continue
        raw = raw.astype(np.float32) -  dark
        raw[gain_map]*=6.85
        if raw_sum is None:
            raw_sum= raw
        else:
            raw_sum += raw
        counts += 1

    mask = details_mask( raw_sum/counts, border=3, plot=True)
    np.save("details_mask", mask.astype(np.bool))
