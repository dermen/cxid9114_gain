
import sys
from cxid9114 import utils
from dxtbx.model.experiment_list import  ExperimentListFactory
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import numpy as np
import scipy.ndimage
from cxid9114.refine import metrics

exp_name = sys.argv[1]
data_name = sys.argv[2]  
tag = sys.argv[3]
hkl_tol = .15

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
                   # unless you want energy-dependent structure factors
                   # you need only provide one number -or- one structure factor flex miller table
                   # and the computer will know to preserve that for all beam colors

# load sf for the data, contains wavelength dependence!
FFdat = [utils.open_flex("SA.pkl"), utils.open_flex("SB.pkl")]


FLUX = [1e12, 1e12]  # fluxes of the beams

flux_frac = np.random.uniform(.2,.8)
chanA_flux = flux_frac*1e12
chanB_flux = (1.-flux_frac)*1e12
FLUXdat = [chanA_flux, chanB_flux]
GAIN = np.random.uniform(0.5,3)

waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
exp_lst = ExperimentListFactory.from_json_file(exp_name) #, check_format=False)
iset = exp_lst.imagesets()[0]
detector = iset.get_detector(0)
data = utils.open_flex( data_name)
beamA = deepcopy(iset.get_beam())
beamB = deepcopy(iset.get_beam())
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

crystalAB = data["crystalAB"]


simsAB = sim_utils.sim_twocolors2(
    crystalAB, detector, iset.get_beam(0), FF,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=1, mos_spread=0.0)

simsData = sim_utils.sim_twocolors2(
    crystalAB, detector, iset.get_beam(0), FFdat,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUXdat, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=1, mos_spread=0.)

simsDataTruth = sim_utils.sim_twocolors2(
    crystalAB, detector, iset.get_beam(0), FFdat,
    [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
    FLUX, pids=None, Gauss=False, oversample=2,
    Ncells_abc=(7, 7, 7), mos_dom=1, mos_spread=0.)

simsDataSum = GAIN * ( np.array(simsData[0]) + np.array(simsData[1]))
simsDataTruthSum = np.array(simsDataTruth[0]) + np.array(simsDataTruth[1])

refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA) 
refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB) 

refl_truthA = spot_utils.refls_from_sims(simsDataTruth[0], detector, beamA) 
refl_truthB = spot_utils.refls_from_sims(simsDataTruth[1], detector, beamB) 

# This only uses the beam to instatiate an imageset / datablock
# but otherwise the return value (refl_data) is indepent of the 
# beam object passed
refl_data = spot_utils.refls_from_sims(simsDataSum, detector, beamA) 
refl_dataNoScale = spot_utils.refls_from_sims(simsDataTruthSum, detector, beamA) 

residA = metrics.check_indexable2(
    refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
residB = metrics.check_indexable2(
    refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)

# use the form factors
resid_truthA = metrics.check_indexable2(
    refl_dataNoScale, refl_simA, detector, beamA, crystalAB, hkl_tol)
resid_truthB = metrics.check_indexable2(
    refl_dataNoScale, refl_simB, detector, beamB, crystalAB, hkl_tol)

dump = {"crystalAB": crystalAB,
        "residA": residA,
        "residB": residB,
        "resid_truthA": resid_truthA,
        "resid_truthB": resid_truthB,
        "beamA": beamA,
        "beamB": beamB,
        "detector": detector,
        "refls_simA": refl_simA,
        "refls_simB": refl_simB,
        "flux_data": FLUXdat,
        "gain": GAIN,
        "refls_truthA": refl_truthA,
        "refls_truthB": refl_truthB,
        "refls_dataNoScale": refl_dataNoScale,
        "refls_data": refl_data}

dump_name = data_name.replace(".pkl", "_%s.pkl" % tag)
utils.save_flex(dump, dump_name)
print "Wrote %s" % dump_name
