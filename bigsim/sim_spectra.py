import h5py
from cxid9114.sim import scattering_factors
from cxid9114.parameters import ENERGY_CONV
from cxid9114 import utils
from cctbx import crystal, sgtbx,miller
from cctbx.array_family import flex
import numpy as np

def sfall_channels(en_chans, output_name):
    
    Fout = []
    for i_en,en in enumerate(en_chans):
        print i_en, len(en_chans)
        wave = ENERGY_CONV/en
        F = scattering_factors.get_scattF(wave, 
            '003_s0_mark0_001.pdb', 
            algo='fft', 
            dmin=1.5, 
            ano_flag=True, load_lines=False)
        Fout.append(F.data().as_numpy_array())
    hkl = F.indices()  # should be same hkl list for all channels
    hkl = np.vstack([hkl[i] for i in range(len(hkl))])
    with h5py.File(output_name, "w") as h5_out:

        h5_out.create_dataset("indices", data=hkl, dtype=np.int, compression="lzf")
        h5_out.create_dataset("data", data=np.vstack(Fout), compression="lzf")
    
def load_spectra(fname):
    f = h5py.File(fname, "r")
    data = f["data"][()]
    indices = f["indices"][()]
    sg = sgtbx.space_group(" P 4nw 2abw")
    Symm = crystal.symmetry( unit_cell=(79,79,38,90,90,90), space_group=sg)
    indices_flex = tuple(map( tuple,indices))
    mil_idx = flex.miller_index(indices_flex)
    mil_set = miller.set( crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)
   
    mil_ar = {} 
    for i_chan, data_chan in enumerate(data):
        print i_chan
        data_flex = flex.complex_double(np.ascontiguousarray(data_chan))
        mil_ar[i_chan] = miller.array(mil_set, data=data_flex)
    return mil_ar

if __name__=="__main__":
    import h5py
    f = h5py.File("test_data.h5", "r")
    en_chans = f["energy_bins"][()]
    sfall_channels(en_chans, "test_sfall.h5")
    print load_spectra("test_sfall.h5")[0].data()[0]
