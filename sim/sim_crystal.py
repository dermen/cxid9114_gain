import os, inspect
import numpy as np
import pylab as plt
cwd = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))

from scitbx.matrix import sqr
from cxid9114.sim import sim_utils
from cxid9114.sim import scattering_factors
from cxid9114 import utils
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype
from simtbx_nanoBragg_ext import convention

cryst_f = os.path.join(cwd,"c1.pkl")
det_f = os.path.join( cwd, "test_det.pkl")
beam_f = os.path.join( cwd, "test_beam.pkl")


class PatternFactory:

    def __init__(self, crystal=None, detector=None, beam=None):
        """
        :param crystal:  dials crystal model
        :param detector:  dials detector model
        :param beam: dials beam model
        """
        self.beam = beam
        self.detector = detector
        if crystal is None:
            crystal = utils.open_flex(cryst_f)
        if self.detector is None:
            self.detector = utils.open_flex(det_f)
        if self.beam is None:
            self.beam = utils.open_flex(beam_f)

        self.SIM2 = nanoBragg( self.detector, self.beam, verbose=10)
        self.SIM2.beamcenter_convention = convention.DIALS
        self.SIM2.oversample = 2  # oversamples the pixel ?
        self.SIM2.polarization = 1  # polarization fraction ?
        self.SIM2.F000 = 10  # should be number of electrons ?
        self.SIM2.default_F = 0
        self.SIM2.Amatrix = sim_utils.Amatrix_dials2nanoBragg(crystal)  # sets the unit cell
        self.SIM2.xtal_shape = shapetype.Tophat
        #self.SIM2.xtal_shape = shapetype.Gauss
        self.SIM2.progress_meter = False
        self.SIM2.flux = 1e14
        self.SIM2.beamsize_mm = 0.004
        self.SIM2.Ncells_abc = (10, 10, 10)
        self.SIM2.interpolate = 0
        self.SIM2.progress_meter = False
        self.SIM2.verbose = 0

    def make_pattern(self, crystal, spectrum, show_spectrum=False,
                     mosaic_domains=5,
                     mosaic_spread=0.1):
        """
        :param crystal:  cctbx crystal
        :param spectrum: np.array of shape 1024
        :return: simulated pattern
        """
        if spectrum.shape[0] != 1024:
            raise ValueError("Spectrum needs to have length 1024 in current version")

        # downsample the provided spectrum
        new_spec = sim_utils.interp_spectrum(spectrum,
                                             sim_utils.ENERGY_CAL,
                                             scattering_factors.interp_energies)

        if show_spectrum:
            plt.plot(scattering_factors.interp_energies, new_spec, 'o')
            plt.show()

        # assume all flux passes into this spectrum
        flux_per_en = new_spec / np.sum(new_spec) * self.SIM2.flux

        # set mosaicity
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.set_mosaic_blocks(sim_utils.mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                            self.SIM2.mosaic_domains))
        pattern = sim_utils.simSIM(self.SIM2,
            ener_eV = scattering_factors.interp_energies,
            flux_per_en = flux_per_en,
            fcalcs = scattering_factors.fcalc_at_wavelen,
            Amatrix = sim_utils.Amatrix_dials2nanoBragg(crystal))
        return pattern

    def adjust_mosaicity(self, mosaic_domains=None, mosaic_spread=None):
        if mosaic_domains is None:
            mosaic_domains = 2  # default
        if mosaic_spread is None:
            mosaic_spread = 0.1
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.set_mosaic_blocks(sim_utils.mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                            self.SIM2.mosaic_domains))

    def make_pattern2(self, crystal, flux_per_en, energies_eV, fcalcs_at_energies,
                      mosaic_domains=None, mosaic_spread=None,ret_sum=True, Op=None):
        """
        :param crystal:
        :param flux_per_en:
        :param energies_eV:
        :param fcalcs_at_energies:
        :param mosaic_domains:
        :param mosaic_spread:
        :param ret_sum:
        :param Op:
        :return:
        """
        # set mosaicity
        if mosaic_domains is not None or mosaic_spread is not None:
            self.adjust_mosaicity(mosaic_domains, mosaic_spread)
        if Op is not None:
            print("Rots!!")
            p_init = crystal.get_unit_cell().parameters()
            Arot = Op * sqr(crystal.get_U()) * sqr(crystal.get_B())
            crystal.set_A(Arot)
            p_final = crystal.get_unit_cell().parameters()
            if not np.allclose( p_init, p_final):
                print "Trying to use matrix Op:"
                print Op
                raise ValueError("Matrix Op is not proper rotation!")

        pattern = sim_utils.simSIM(self.SIM2,
                                   ener_eV = energies_eV,
                                   flux_per_en = flux_per_en,
                                   fcalcs = fcalcs_at_energies,
                                   Amatrix = sim_utils.Amatrix_dials2nanoBragg(crystal),
                                   ret_sum=ret_sum)
        return pattern
