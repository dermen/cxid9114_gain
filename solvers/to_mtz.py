# coding: utf-8
from cxid9114.sim import scattering_factors
from cxid9114.parameters import ENERGY_CONV

def from_sim():
    waveA = ENERGY_CONV/8944.
    FA = scattering_factors.get_scattF(waveA, 
        '003_s0_mark0_001.pdb', 
        algo='fft', dmin=1.5, ano_flag=True)
    IA = FA.as_intensity_array()
    out = IA.as_mtz_dataset(column_root_label="Iobs", title="4bs7", wavelength=waveA)
    out.add_miller_array(miller_array=IA.average_bijvoet_mates(), column_root_label="IMEAN")
    obj = out.mtz_object()
    obj.write("4bs7.mtz")

def from_data():
    from cxid9114 import utils
    waveA = ENERGY_CONV/8944.
    IA = utils.open_flex("test_refined.pkl") # this is refined miller data
    out = IA.as_mtz_dataset(column_root_label="Iobs", title="test_refined", wavelength=waveA)
    out.add_miller_array(miller_array=IA.average_bijvoet_mates(), column_root_label="IMEAN")
    obj = out.mtz_object()
    obj.write("test_refined.mtz")

def from_data_init():
    from cxid9114 import utils
    waveA = ENERGY_CONV/8944.
    IA = utils.open_flex("test_refined_init.pkl") # this is refined miller data
    out = IA.as_mtz_dataset(column_root_label="Iobs", title="test_refined_init", wavelength=waveA)
    out.add_miller_array(miller_array=IA.average_bijvoet_mates(), column_root_label="IMEAN")
    obj = out.mtz_object()
    obj.write("test_refined_init.mtz")


def from_named_data(pkl_name):
    from cxid9114 import utils
    waveA = ENERGY_CONV/8944.
    IA = utils.open_flex(pkl_name) # this is refined miller data
    out = IA.as_mtz_dataset(column_root_label="Iobs", title="B", wavelength=waveA)
    out.add_miller_array(miller_array=IA.average_bijvoet_mates(), column_root_label="IMEAN")
    obj = out.mtz_object()
    obj.write(pkl_name.replace(".pkl",".mtz"))

if __name__=="__main__":
    import sys
    from_named_data(sys.argv[1])
