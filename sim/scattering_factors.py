from joblib import Parallel, delayed, effective_n_jobs
import os
import numpy as np
import cPickle
from scipy import constants
import iotbx
from cctbx.eltbx import henke
from dials.array_family import flex  # needed for pickle

fcalc_file = "fcalc_at_wave.pkl"
pdb_name = "4bs7.pdb"

# load the PDB as text file to pass to iotbx later
lines = open(pdb_name,"r").readlines()

interp_energies = np.hstack((
            np.linspace(8940, 8952, 25),
            np.linspace(9030, 9038, 25)
            ))
wavelens_A = 1e10 * constants.h * constants.c/ \
             (constants.electron_volt* interp_energies)

scatt_param = {
    'd_min': 1.5,  # found these in an example
    'anomalous_flag': True,
    'algorithm': 'direct'}

def get_scattF(wavelen_A, **kwargs):
    """
    mostly borrowed from tst_nanoBragg_basic.py
    """
    pdb_in = iotbx.pdb.input(source_info=None, lines=lines)
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
def main(wavelens_A, wavelen_idx, jid):
    fcalc_at_wavelen = {}
    for i_wave in wavelen_idx:
        waveA = wavelens_A[i_wave]
        print("Job %d;  Computing scattering at wavelen %.4f ( %d/ %d )" \
              % (jid, waveA, i_wave + 1, len(wavelens_A)))
        fcalc_at_wavelen[i_wave] = \
            get_scattF(waveA, lines, **scatt_param)
    return fcalc_at_wavelen

if fcalc_file is None or not os.path.exists(fcalc_file):
    n_jobs = effective_n_jobs()
    wavelens_idx = np.array_split(range(len(wavelens_A)), n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(main)(wavelens_A, wavelens_idx[jid], jid)
                                      for jid in range(n_jobs))
    fcalc_at_wavelen = {}
    for result_dict in results:
        for k, v in result_dict.iteritems():
            fcalc_at_wavelen.setdefault(k, []).append(v)
            fcalc_at_wavelen[k] = fcalc_at_wavelen[k][0]
else:
    fcalc_at_wavelen = cPickle.load(open(fcalc_file,"r"))
