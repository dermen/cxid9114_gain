import pylab as plt
import numpy as np
import cPickle
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype
from simtbx_nanoBragg_ext import convention
from dials.array_family import flex


crystal = cPickle.load(open("test_cryst.pkl","r"))
beam = cPickle.load(open("test_beam.pkl", "r"))
detector = cPickle.load(open("test_det.pkl", "r"))

# initial A-matrix
Amatrix = tuple(np.array(crystal.get_A()).reshape((3, 3)).T.ravel())

SIM2 = nanoBragg(detector, beam, verbose=10)
SIM2.beamcenter_convention = convention.DIALS
SIM2.oversample = 1  # oversamples the pixel ?
SIM2.polarization = 1  # polarization fraction ?
SIM2.F000 = 200  # should be number of electrons ?
SIM2.default_F = 0
SIM2.xtal_shape = shapetype.Tophat
SIM2.progress_meter = False
SIM2.flux = 1e14
SIM2.beamsize_mm = 0.004
SIM2.Amatrix = Amatrix
SIM2.Ncells_abc = (10, 10, 10,)
SIM2.xtal_size_mm = (5e-5, 5e-5, 5e-5)
SIM2.interpolate = 0
SIM2.mosaic_domains = 25  # from LS49
SIM2.mosaic_spread_deg = 0.05  # from LS49
SIM2.verbose = 9
SIM2.progress_meter=False


