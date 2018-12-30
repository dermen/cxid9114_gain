from dials.array_family import flex
from scipy import constants
import pylab as plt
import numpy as np
import cPickle
from simtbx import nanoBragg
from scipy.interpolate import interp1d
from iotbx import pdb
from cctbx.eltbx import henke
from simtbx.nanoBragg import shapetype

def interp_spectrum( spectrum, energies,
            new_energies):
    """
    spectrum, np/ndarray
        lineout of the FEEspec
    energies, np.ndarray
        known energies corresponding to spec axis
    new_energies, np.ndarray
        evaluate spectrum at these energies
    """
    I = interp1d( energies,
                spectrum, bounds_error=False )
    new_spec =  I( new_energies)
    if np.any( np.isnan( new_spec)):
        print("Asking for energies off the spectrogram")
    return new_spec



crystal = cPickle.load(open("test_cryst.pkl","r"))
beam = cPickle.load(open("test_beam.pkl", "r"))
detector = cPickle.load(open("test_det.pkl", "r"))

SIM = nanoBragg.nanoBragg(detector, beam)
SIM.Amatrix = crystal.get_A()

spectrum = np.load("spec_trace_mean.npy")
energies = np.load("energy_cal_r62.npy") # energy that matches the trace

spectrum[ spectrum < 0 ] =0 # because of dark cal, sometimes get negatives

# we want to simulate into these intensities
# can do full range, but simplified first to save on time..
interp_energies = np.hstack((
            np.linspace( 8940, 8952, 25),
            np.linspace(9030, 9038, 25)
            ))

from IPython import embed
embed()

scatt_param = {'d_min':1.5, # found these in an example
        'anomalous_flag':True,
        'algorithm':'direct'}

pdb_name = "4bs7.pdb" # Ytt derivitive
Ncells_abc = (10,10,10) # needs to be bigger
flux = 1e12
verbose=8
beamsize_mm = 0.004 # my best guess..

# ===========

wavelens_A = 1e10 * constants.h * constants.c/ \
    (constants.electron_volt* interp_energies)

# downsample the spectrum?
new_spec = interp_spectrum( spectrum, energies, interp_energies)
flux_per_en = new_spec / np.sum( new_spec) * flux # assume all flux into this spectrum

plt.plot( wavelens_A, flux_per_en, '.')
plt.xlabel("Angstrom")
plt.ylabel("flux of photons")
plt.title("Inputs to simtbx")
plt.show()

# load the PDB as text file to pass to iotbx later
lines = open(pdb_name,"r").readlines()
SIM.oversample=1 # oversamples the pixel ?
SIM.polarization=1 # polarization fraction ?
SIM.F000=200 # should be number of electrons ?
SIM.default_F=0
SIM.xtal_shape=shapetype.Tophat
SIM.progress_meter=False
SIM.flux=flux
SIM.beamsize_mm=beamsize_mm

def get_scattF(wavelen_A, lines, **kwargs):
    """
    mostly borrowed from tst_nanoBragg_basic.py

    I think I dont want the primitive setting
    """
    pdb_in = pdb.input(source_info=None, lines=lines)
    xray_structure = pdb_in.xray_structure_simple()

    scatts = xray_structure.scatterers()

    for sc in scatts:
        expected_henke = henke.table(
            sc.element_symbol()).at_angstrom(wavelen_A)

        sc.fp = expected_henke.fp()
        sc.fdp = expected_henke.fdp()

    fcalc = xray_structure.structure_factors(**kwargs).f_calc()
    return fcalc

# this part is slow, but done once
fcalc_at_wavelen = {}
for i_wave, waveA in enumerate(wavelens_A):
    print("Computing scattering at wavelen %.4f ( %d/ %d )" \
          % (waveA, i_wave + 1, len(wavelens_A)))
    fcalc_at_wavelen[i_wave] = \
        get_scattF(waveA, lines, **scatt_param)

def PatternFactory(wavelens_A=wavelens_A,
                   flux_per_en=flux_per_en,
                   SIM=SIM,
                   fcalc_at_wavelen=fcalc_at_wavelen):
    # SIM.randomize_orientation()
    # print SIM.missets_deg

    Nwavelen = len(wavelens_A)
    pattern_at_wave = {}
    for i_wave in range(Nwavelen):
        print ("sim spots %d / %d" % (i_wave + 1, Nwavelen))
        SIM.wavelength_A = wavelens_A[i_wave]
        SIM.flux = flux_per_en[i_wave]
        SIM.Fhkl = fcalc_at_wavelen[i_wave].amplitudes()
        SIM.add_nanoBragg_spots()
        pattern_at_wave[i_wave] = SIM.raw_pixels.as_numpy_array()
        SIM.raw_pixels *= 0

    full_pattern = np.sum([pattern_at_wave[i_wave]
                           for i in range(Nwavelen)],
                          axis=0)
    return full_pattern