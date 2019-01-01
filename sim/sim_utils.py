from scipy.interpolate import interp1d
import numpy as np
import inspect

from simtbx.nanoBragg import nanoBragg
from dials.array_family import flex
import scitbx
from scitbx.matrix import col

ENERGY_CAL = np.load("energy_cal_r62.npy")


def energy_cal():
    """
    loads the energy calibration (energy per pixel in the spectrometer)
    :return:  energy per line-readout in the spectrometer (1024 pixels)
    """
    energy_cal = np.load("energy_cal_r62.npy")
    return energy_cal

def mosaic_blocks(mos_spread_deg, mos_domains,
                  twister_seed=0, random_seed=1234):
    """
    :param mos_spread_deg: spread in degrees
    :param mos_domains: number of mosaic domains
    :param twister_seed: default from ls49 code
    :param random_seed: default from ls49 code
    :return:
    """
    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=twister_seed)
    scitbx.random.set_random_seed(random_seed)
    rand_norm = scitbx.random.normal_distribution(mean=0,
                                                  sigma=mos_spread_deg*np.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mos_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(m, deg=False) )
    return UMAT_nm


def interp_spectrum(spectrum, energies, new_energies):
    """
    spectrum, np/ndarray
        lineout of the FEEspec
    energies, np.ndarray
        known energies corresponding to spec axis
    new_energies, np.ndarray
        evaluate spectrum at these energies
    """
    I = interp1d(energies, spectrum, bounds_error=False)
    new_spec = I(new_energies)
    if np.any(np.isnan( new_spec)):
        print("There are Nan in your spectrum...")
    return new_spec


def print_parameters(SIM):
    """
    prints nanobragg params
    :param SIM: nanobragg instance
    :return:
    """
    isprop = lambda x: isinstance(x, property)
    bad_params = ('Fbg_vs_stol', 'Fhkl_tuple', 'amplitudes', 'indices', 'raw_pixels',
                  'xray_source_XYZ', 'xray_source_intensity_fraction', 'xray_beams',
                  'xray_source_wavelengths_A', 'unit_cell_tuple', 'progress_pixel')
    params = [name
              for (name, value) in inspect.getmembers(nanoBragg, isprop)
              if name not in bad_params]

    print "Did not try to get these parameters:"
    print bad_params

    failed = []
    for p in params:
        try:
            param_value = getattr(SIM, p)
            print p, param_value
            print
        except ValueError:
            failed.append(p)

    print "Failed to get these parameters:"
    print failed


def Amatrix_dials2nanoBragg(crystal):
    """
    returns the A matrix from a cctbx crystal object
    in nanoBragg frormat
    :param crystal: cctbx crystal
    :return: Amatrix as a tuple
    """
    Amatrix = tuple(np.array(crystal.get_A()).reshape((3, 3)).T.ravel())
    return Amatrix


def simSIM(SIM, ener_eV, flux_per_en, fcalcs, Amatrix):
    """
    :param SIM:  instance of nanoBragg
    :param ener_eV:  the spectrum energies in eV
    :param flux_per_en:  the flux per wavelength channel
    :param fcalcs:  the structure factors computed per energy channel
    :param Amatrix: A matrix for cctbx
    :return:
    """
    n_ener = len(ener_eV)
    patterns = []
    for i_ener in range(n_ener):
        print ("sim spots %d / %d" % (i_ener+1, n_ener))
        SIM.energy_eV = ener_eV[i_ener]
        SIM.flux = flux_per_en[i_ener]
        SIM.Fhkl = fcalcs[i_ener].amplitudes()
        SIM.Amatrix = Amatrix
        SIM.add_nanoBragg_spots()
        patterns.append(SIM.raw_pixels.as_numpy_array())
        SIM.raw_pixels *= 0

    full_pattern = np.sum( patterns,axis=0)
    return full_pattern

