import numpy as np
import cPickle

from cxid9114.sim import sim_utils
from cxid9114.sim import scattering_factors

from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype
from simtbx_nanoBragg_ext import convention
from dials.array_family import flex


crystal = cPickle.load(open("c1.pkl","r"))
beam = cPickle.load(open("test_beam.pkl", "r"))
detector = cPickle.load(open("test_det.pkl", "r"))
spectrum = np.load("spec_trace_mean.npy")

# =============================================
# Define the simulator here
SIM2 = nanoBragg(detector, beam, verbose=7)
SIM2.beamcenter_convention = convention.DIALS
SIM2.oversample = 1  # oversamples the pixel ?
SIM2.polarization = 1  # polarization fraction ?
SIM2.F000 = 200  # should be number of electrons ?
SIM2.default_F = 0
SIM2.Amatrix = sim_utils.Amatrix_dials2nanoBragg(crystal)
SIM2.xtal_shape = shapetype.Tophat
SIM2.progress_meter = False
SIM2.flux = 1e14
SIM2.beamsize_mm = 0.004
SIM2.Ncells_abc = (10, 10, 10,)
# SIM2.xtal_size_mm = (5e-5, 5e-5, 5e-5)
SIM2.interpolate = 0
SIM2.mosaic_domains = 25  # from LS49
SIM2.mosaic_spread_deg = 0.05  # from LS49
SIM2.progress_meter = False
SIM2.set_mosaic_blocks(sim_utils.mosaic_blocks(SIM2.mosaic_spread_deg,
                                               SIM2.mosaic_domains))

# ===========================================

# downsample the provided spectrum
new_spec = sim_utils.interp_spectrum(spectrum,
                                     sim_utils.ENERGY_CAL,
                                     scattering_factors.interp_energies)
# assume all flux into this spectrum
flux_per_en = new_spec / np.sum(new_spec) * SIM2.flux

from IPython import embed
embed()
#full_pattern = sim_utils.simSIM(SIM2)



