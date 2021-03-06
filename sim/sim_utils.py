
import os, sys
import io
import scipy.interpolate
interp1d = scipy.interpolate.interp1d
import numpy as np
import inspect
import dxtbx
import cctbx
import cxid9114.utils as utils
import dials.array_family.flex as flex
import scitbx
import scitbx.matrix
sqr = scitbx.matrix.sqr
col = scitbx.matrix.col
import cPickle
import copy
deepcopy = copy.deepcopy
import cxid9114.sim.scattering_factors as scattering_factors
import pylab as plt
from cxid9114 import parameters
import h5py

import simtbx.nanoBragg
nanoBragg = simtbx.nanoBragg.nanoBragg
shapetype = simtbx.nanoBragg.shapetype
convention = simtbx.nanoBragg.convention

try:
    import joblib
    effective_n_jobs = joblib.effective_n_jobs
    Parallel = joblib.delayed
    Parallel = joblib.Parallel
    NO_JOBLIB = False
except ImportError:
    NO_JOBLIB = True

cwd = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))
energy_file = os.path.join(cwd, "energy_cal_r62.npy")
ENERGY_CAL = np.load(energy_file)
cryst_f = os.path.join(cwd, "c1.pkl")
det_f = os.path.join(cwd, "test_det.pkl")
beam_f = os.path.join(cwd, "test_beam.pkl")

def energy_cal():
    """
    loads the energy calibration (energy per pixel in the spectrometer)
    :return:  energy per line-readout in the spectrometer (1024 pixels)
    """
    energy_cal = np.load(energy_file)
    return energy_cal

def load_fcalc_file(fcalc_file):
    fcalcs_data = utils.open_flex(fcalc_file)
    fcalcs_at_en = fcalcs_data["fcalc"]
    energies = fcalcs_data["energy"]
    return energies, fcalcs_at_en

def save_fcalc_file(energies, fcalcs_at_en, filename):
    fcalc_data = {}
    fcalc_data["energy"] = energies
    fcalc_data["fcalc"] = fcalcs_at_en
    with open(filename, "w") as ff:
        cPickle.dump( fcalc_data, ff)

def mosaic_blocks(mos_spread_deg, mos_domains,
                  twister_seed=0, random_seed=1234):
    """
    Code from LS49 for adjusting mosaicity of simulation
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
    new_spec[ new_spec < 0] = 0
    return new_spec

def compare_sims(SIM1, SIM2):
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
            param_value1 = getattr(SIM1, p)
            param_value2 = getattr(SIM2, p)
            if isinstance(param_value1, tuple):
                params_are_equal = np.allclose( param_value1, param_value2)
            else:
                params_are_equal = param_value1 == param_value2
            if not params_are_equal:
                print p, param_value1
                print p, param_value2
                print
        except ValueError:
            failed.append(p)

    print "Failed to get these parameters:"
    print failed

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


def simSIM(SIM=None, ener_eV=None, flux_per_en=None,
           fcalcs=None, Amatrix=None, silence=True, ret_sum=True):
    """
    :param SIM:  instance of nanoBragg
    :param ener_eV:  the spectrum energies in eV
    :param flux_per_en:  the flux per wavelength channel
    :param fcalcs:  the structure factors computed per energy channel
    :param Amatrix: A matrix for cctbx
    :return: pattern simulated
    """
    n_ener = len(ener_eV)
    patterns = []
    v = SIM.verbose
    if silence:
        SIM.verbose = 0
    for i_ener in range(n_ener):
        print "\rsim spots %d / %d" % (i_ener+1, n_ener),
        sys.stdout.flush()
        SIM.wavelength_A = parameters.ENERGY_CONV / ener_eV[i_ener]
        SIM.flux = flux_per_en[i_ener]
        if SIM.flux > 0:
            print "%0.4f, %0.4f" % (SIM.energy_eV, SIM.wavelength_A)
            SIM.Fhkl = fcalcs[i_ener].amplitudes()
            SIM.Amatrix = Amatrix
            SIM.add_nanoBragg_spots()
        patterns.append(SIM.raw_pixels.as_numpy_array())
        SIM.raw_pixels *= 0
    SIM.verbose = v
    print
    if ret_sum:
        full_pattern = np.sum( patterns,axis=0)
        return full_pattern
    else:
        return patterns


def simulate_xyscan_result(scan_data_file, prefix=None):
    """
    scan data is the output of yhe xyscan refinement script

    :param scan_data_file: string path
    :param prefix: output path prefix where results will be written
    :return:
    """
    scan_data = np.load(scan_data_file)
    idx = int(scan_data['hit_idx'])
    if prefix is None:
        prefix = "refined_sim%d" % idx

    # crystal = scan_data['crystal']
    refl = scan_data['refl']

    # xR = scan_data['optX']
    # yR = scan_data['optY']
    optCrystal = scan_data['optCrystal']
    fracA = int(scan_data["fracA"])
    fracB = int(scan_data["fracB"])
    flux = [fracA*1e14, fracB*1e14]
    energy, fcalc_f = load_fcalc_file(scan_data['fcalc_f'])

    P = PatternFactory()
    P.adjust_mosaicity(2, 0.05)

    sim_A, sim_B = P.make_pattern2(crystal=deepcopy(optCrystal),
                                   flux_per_en=flux,
                                   energies_eV=energy,
                                   fcalcs_at_energies=fcalc_f,
                                   mosaic_spread=None,
                                   mosaic_domains=None,
                                   ret_sum=False,
                                   Op=None)  # xR * yR)

    if 'optX_fine' in scan_data.keys():
        #xR_fine = scan_data['optX_fine']
        #yR_fine = scan_data['optY_fine']
        optCryst_fine = scan_data["optCrystal_fine"]
        sim_A_fine, sim_B_fine = P.make_pattern2(crystal=deepcopy(optCryst_fine),
                                       flux_per_en=flux,
                                       energies_eV=energy,
                                       fcalcs_at_energies=fcalc_f,
                                       mosaic_spread=None,
                                       mosaic_domains=None,
                                       ret_sum=False,
                                       Op=None)  # xR_fine * yR_fine * xR * yR)
        fine_scan=True
    else:
        fine_scan = False
    # work this into the data stream, perhaps use experiment lists??
    img_file = "/Users/dermen/cxid9114/run62_hits_wtime.h5"
    loader = dxtbx.load(img_file)
    raw_img = loader.get_raw_data(idx).as_numpy_array()

    patts = [sim_A, sim_B, sim_A+sim_B, raw_img]
    refls = [refl]*4
    utils.images_and_refls_to_simview(prefix, patts, refls)

    if fine_scan:
        patts = [sim_A_fine, sim_B_fine, sim_A_fine+sim_B_fine, raw_img]
        refls = [refl]*4
        utils.images_and_refls_to_simview(prefix+"_fine", patts, refls)


class PatternFactory:
    def __init__(self, crystal=None, detector=None, beam=None,
                 Ncells_abc=(10,10,10), Gauss=False, oversample=0, panel_id=0,
                 recenter=True, verbose=10, profile=None, device_Id=None,
                 beamsize_mm=None, exposure_s=None, flux=None):
        """
        :param crystal:  dials crystal model
        :param detector:  dials detector model
        :param beam: dials beam model
        """
        self.beam = beam
        self.detector = detector
        self.panel_id = panel_id
        if crystal is None:
            crystal = utils.open_flex(cryst_f)
        if self.detector is None:
            self.detector = utils.open_flex(det_f)
        if self.beam is None:
            self.beam = utils.open_flex(beam_f)

        self.SIM2 = nanoBragg(self.detector, self.beam, verbose=verbose, panel_id=panel_id)
        if oversample > 0:
            self.SIM2.oversample = oversample
        self.SIM2.polarization = 1  # polarization fraction ?
        self.SIM2.Ncells_abc = Ncells_abc  # important to set this First!
        self.SIM2.F000 = 0  # should be number of electrons ?
        self.SIM2.default_F = 0
        self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)  # sets the unit cell
        if Gauss:
            self.SIM2.xtal_shape = shapetype.Gauss
        else:
            self.SIM2.xtal_shape = shapetype.Tophat
        
        if profile is not None:  # override above
            if profile == "gauss":
                self.SIM2.xtal_shape = shapetype.Gauss
            elif profile == "tophat":
                self.SIM2.xtal_shape = shapetype.Tophat
            elif profile == "round":
                self.SIM2.xtal_shape = shapetype.Round
            elif profile == "square":
                self.SIM2.xtal_shape = shapetype.Square
       
        if device_Id is not None: 
            self.SIM2.device_Id=device_Id
            self.SIM2.timelog=False
        self.SIM2.progress_meter = False
        if flux is not None:
            self.SIM2.flux = flux
        else:
            self.SIM2.flux = 1e14
        if beamsize_mm is not None:
            self.SIM2.beamsize_mm = beamsize_mm
        else:
            self.SIM2.beamsize_mm = 0.004

        self.SIM2.interpolate = 0
        self.SIM2.progress_meter = False
        self.SIM2.verbose = verbose
        self.SIM2.seed = 9012
        self.default_fcalc = None
        self.default_interp_en = scattering_factors.interp_energies
        self.FULL_ROI = self.SIM2.region_of_interest

        if recenter:  # FIXME: I am not sure why this seems to be necessary to preserve geom
            #a = self.SIM2.beam_center_mm
            #print "Beam center was:",a
            self.SIM2.beam_center_mm = self.detector[panel_id].get_beam_centre(self.beam.get_s0())
            #print "Now, beam center is ", self.SIM2.beam_center_mm, "Why is this happening???"

    def make_pattern_default(self, crystal, spectrum, show_spectrum=False,
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
        new_spec = interp_spectrum(spectrum,
                                     ENERGY_CAL,
                                     self.default_interp_en)

        if self.default_fcalc is None:
            print("Initializing Fcalcs default values, done just once!")
            self.default_fcalc = scattering_factors.load_fcalc_for_default_spectrum()

        if show_spectrum:
            plt.plot(self.default_interp_en, new_spec, 'o')
            plt.show()

        # assume all flux passes into this spectrum
        flux_per_en = new_spec / np.sum(new_spec) * self.SIM2.flux

        # set mosaicity
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.set_mosaic_blocks(mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                self.SIM2.mosaic_domains))

        pattern = simSIM(self.SIM2,
                           ener_eV=self.default_interp_en,
                           flux_per_en=flux_per_en,
                           fcalcs=self.default_fcalc,
                           Amatrix=Amatrix_dials2nanoBragg(crystal))
        return pattern

    def adjust_mosaicity(self, mosaic_domains=None, mosaic_spread=None):
        if mosaic_domains is None:
            mosaic_domains = 2  # default
        if mosaic_spread is None:
            mosaic_spread = 0.1
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.SIM2.set_mosaic_blocks(mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                    self.SIM2.mosaic_domains))

    def adjust_divergence(self, div_tuple=(0,0)):
        self.SIM2.divergence_hv_mrad = div_tuple

    def adjust_dispersion(self, pct=0.):
        self.SIM2.dispersion_pct = pct

    def make_pattern2(self, crystal, flux_per_en, energies_eV, fcalcs_at_energies,
                      mosaic_domains=None, mosaic_spread=None, ret_sum=True, Op=None):
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
            if not np.allclose(p_init, p_final):
                print "Trying to use matrix Op:"
                print Op
                raise ValueError("Matrix Op is not proper rotation!")

        pattern = simSIM(self.SIM2,
                           ener_eV=energies_eV,
                           flux_per_en=flux_per_en,
                           fcalcs=fcalcs_at_energies,
                           Amatrix=Amatrix_dials2nanoBragg(crystal),
                           ret_sum=ret_sum)
        return pattern

    def primer(self, crystal, energy, flux, F=None):
        self.SIM2.wavelength_A = parameters.ENERGY_CONV / energy
        self.SIM2.flux = flux
        if isinstance(F, cctbx.miller.array):
            self.SIM2.Fhkl = F.amplitudes()
        elif F is not None:
            self.SIM2.default_F = F
        if crystal is not None:
            self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)
        self.SIM2.raw_pixels *= 0
        self.SIM2.region_of_interest = self.FULL_ROI

    def sim_rois(self, rois, reset=True, cuda=False, omp=False, add_water=False,
                 add_noise=False, water_par=None, noise_par=None, boost=1, add_spots=True):
        if add_spots:
            for roi in rois:
                self.SIM2.region_of_interest = roi
                if cuda:
                    self.SIM2.add_nanoBragg_spots_cuda()
                elif omp:
                    from boost.python import streambuf  # will deposit printout into dummy StringIO as side effect
                    from six.moves import StringIO
                    self.SIM2.add_nanoBragg_spots_nks(streambuf(StringIO()))
                else:
                    self.SIM2.add_nanoBragg_spots()

            self.SIM2.raw_pixels = self.SIM2.raw_pixels*boost
        
        if add_water:
            water_scatter = flex.vec2_double(
                [(0, 2.57), (0.0365, 2.58), (0.07, 2.8), (0.12, 5), (0.162, 8), (0.2, 6.75), (0.18, 7.32),
                 (0.216, 6.75), (0.236, 6.5), (0.28, 4.5), (0.3, 4.3), (0.345, 4.36), (0.436, 3.77), (0.5, 3.17)])
            self.SIM2.Fbg_vs_stol = water_scatter
            self.SIM2.amorphous_sample_thick_mm = 0.004  # typical GDVN
            self.SIM2.amorphous_density_gcm3 = 1
            self.SIM2.amorphous_molecular_weight_Da = 18
            self.SIM2.exposure_s = 1.0  # multiplies flux x exposure
            self.SIM2.region_of_interest = self.FULL_ROI
            self.SIM2.add_background()

        if add_noise:
            self.SIM2.add_noise()

        img = self.SIM2.raw_pixels.as_numpy_array()

        if reset:
            self.SIM2.raw_pixels *= 0
            self.SIM2.region_of_interest = self.FULL_ROI

        return img


def sim_twocolors(crystal, detector=None, panel_id=0, Gauss=False, oversample=0,
             Ncells_abc=(5,5,5), mos_dom=20, fcalc_f="fcalc_slim.pkl",
             mos_spread=0.15, fracA=0.5, fracB=0.5, tot_flux=1e14):
    """
    The idea, given a crystal and a list of reflections, simulate patterns
    in the two color channels,

    :param crystal: dxtbx crystal model
    :param detector: dxtbx detector model
    :param panel_id: panel id for the detector
    :param Gauss: simtbx param, use a gaussian crystal model or not (if not, use tophat)
    :param oversample: simtbx param, over-sample each pixel in simulation to get better intensity meas
    :param Ncells_ab: simtbx param, how many cells
    :param mos_dom: simtbx param, number of mosaic domains
    :param fcalc_f: simtbx param, path to fcalc file, in this case the two color fcalc file should be specified
                an fcalc file is a pickled dictionary with two keys, unlocking 1) photon energies (floats) and
                2) cctbx structure factor objects for that energy, see `cxid9114/sim/fcalc_slim.pkl` which
                is default for two color, made using the code in `sim/scattering_factors.py`
    :param mos_spread:  azimuthal spread parameter in degrees
    :param fracA: fraction of color A from the 2-color spectrum
    :param fracB:  `` colorB ``
    :param tot_flux: total flux for the simtbx simulation, will be divided into two color channels
    :return: the output dictionary described above, has a lot of useful information! Just explore below.
    """
    Patts = PatternFactory(detector=detector,
                           Ncells_abc=Ncells_abc,
                           Gauss=Gauss,
                           oversample=oversample,
                           panel_id=panel_id)

    en, fcalc = load_fcalc_file(fcalc_f)
    flux = [fracA * tot_flux, fracB * tot_flux]
    sim_patt = Patts.make_pattern2(crystal, flux, en, fcalc, mos_dom, mos_spread, False)
    imgA, imgB = sim_patt

    # OUTPUT DICTIONARY, many objects stored for bookkeeping purposes.
    dump = {'imgA': imgA,
            'imgB': imgB,
            'sim_param': {'mos_dom': mos_dom,
                          'mos_spread': mos_spread,
                          'Gauss': Gauss,
                          'Ncells_abc': Ncells_abc,
                          'tot_flux': tot_flux,
                          'fracA': fracA,
                          'fracB': fracB,
                          'fcalc_f': fcalc_f,
                          'crystal': crystal
                          }
            }

    return dump

def sim_twocolors2(crystal, detector, beam, fcalcs, energies, fluxes, pids=None,
                   Gauss=False, oversample=0, Ncells_abc=(5,5,5),verbose=0,
                   div_tup=(0.,0.), disp_pct=0., mos_dom=2, mos_spread=0.15, profile=None,
                   roi_pp=None, counts_pp=None, cuda=False, omp=False, gimmie_Patt=False,
                   add_water=False, add_noise=False, boost=1, device_Id=0,
                   beamsize_mm=None, exposure_s=None, accumulate=False, only_water=False, add_spots=True):

    Npan = len(detector)
    Nchan = len(energies)

    if only_water:
        add_spots = False
        add_water = True
    else:
        add_spots = True
    # initialize output form
    panel_imgs = {}
    for i_en in range(Nchan):
        if accumulate:
            panel_imgs = [None]*Npan
        else:
            panel_imgs[i_en] = []

    if pids is None:
        pids = range(Npan)

    for ii, i_pan in enumerate(pids):
        PattF = PatternFactory(detector=detector,
                               crystal=crystal,
                               beam=beam,
                               panel_id=i_pan,
                               recenter=True,
                               Gauss=Gauss,
                               verbose=verbose,
                               Ncells_abc=Ncells_abc,
                               oversample=oversample, 
                               profile=profile, 
                               beamsize_mm=beamsize_mm,
                               exposure_s=exposure_s,
                               flux=np.sum(fluxes),
                               device_Id=device_Id)
        
        if not only_water:
            PattF.adjust_mosaicity(mos_dom, mos_spread)
        else:
            PattF.adjust_mosaicity(1,0)
       
        PattF.adjust_dispersion(disp_pct)
        PattF.adjust_divergence(div_tup)
        
        for i_en in range(Nchan):
            if fluxes[i_en] == 0:
                continue
            
            PattF.primer(crystal=crystal,
                         energy=energies[i_en],
                         F=fcalcs[i_en],
                         flux=fluxes[i_en])

            if roi_pp is None:
                color_img = PattF.sim_rois(rois=[PattF.FULL_ROI], 
                            reset=True, cuda=cuda, omp=omp,
                            add_water=add_water, add_noise=False, add_spots=add_spots,
                            boost=boost)
            else:
                color_img = PattF.sim_rois(rois=roi_pp[ii], reset=True, 
                            cuda=cuda, omp=omp,
                            add_water=add_water, 
                            add_spots=add_spots,
                            add_noise=False,boost=boost)
                where_finite = counts_pp[ii] > 0
                if np.any(where_finite):
                    color_img[where_finite] /= counts_pp[ii][where_finite]

            if accumulate:
                if panel_imgs[i_pan] is None:
                    panel_imgs[i_pan] = color_img
                else:
                    panel_imgs[i_pan] += color_img
            else:
                panel_imgs[i_en].append(color_img)

        PattF.SIM2.free_all()
    if gimmie_Patt:
        return panel_imgs, PattF
    else:
        #PattF.SIM2.free_all()
        #del PattF
        return panel_imgs


def save_twocolor(simsAB, iset, fname, force=False):
    """
    save a 4-image dataset
    First image is color channel A
    Second is color channel B
    3rd is the composited A+B color images
    4th is (should be) the data to compare with

    The raw data of iset should be 64 panels, and the sims should
    be simulated onto the same geometry / detector
    simsAB is generateed in sim_twocolor2 (panel_imgs) if two colors
    are provided as its arguments./
    :param simsAB:  output of sim_twocolor2 for two color channels
        so its a dictionary with two keys, 0 and 1, each pointint
        to a list of 64 panels of size 185 x 194
    :param iset: an imageset containing raw data which is a
        tuple of length 64 and each element is a
        flex array when converted to numpy is a 185x194 array
    :param fname: output fname
    :param force: whether to force an override of fname
    :return:
    """
    data_panels = np.array( map(lambda x: x.as_numpy_array(), iset.get_raw_data(0) ))
    data = np.array( [ simsAB[0], simsAB[1],
                      np.array(simsAB[0]) + np.array(simsAB[1]),
                      data_panels])
    if os.path.exists(fname) and not force:
        print("%s exists!" % fname)
        return

    with h5py.File(fname,"w") as h5:
        h5.create_dataset("sim64_d9114_images", data=data)


#def sim_channel(crystal, channel_en, Fcalc, detector=None,
#                panel_id=0, Gauss=False, oversample=2,
#                Ncells_abc=(5,5,5), mos_dom=20,
#                mos_spread=0.15, tot_flux=1e14):
#
#    Patts = PatternFactory(detector=detector,
#                           Ncells_abc=Ncells_abc,
#                           Gauss=Gauss,
#                           oversample=oversample,
#                           panel_id=panel_id)
#
#    en, fcalc = load_fcalc_file(fcalc_f)
#    flux = [fracA * tot_flux, fracB * tot_flux]
#    sim_patt = Patts.make_pattern2(crystal, flux, en, fcalc, mos_dom, mos_spread, False)
#    imgA, imgB = sim_patt
#
#    # OUTPUT DICTIONARY, many objects stored for bookkeeping purposes.
#    dump = {'imgA': imgA,
#            'imgB': imgB,
#            'sim_param': {'mos_dom': mos_dom,
#                          'mos_spread': mos_spread,
#                          'Gauss': Gauss,
#                          'Ncells_abc': Ncells_abc,
#                          'tot_flux': tot_flux,
#                          'fracA': fracA,
#                          'fracB': fracB,
#                          'fcalc_f': fcalc_f,
#                          'crystal': crystal
#                          }
#            }
#
#    return dump
