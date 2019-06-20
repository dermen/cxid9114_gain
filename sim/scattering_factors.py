from joblib import Parallel, delayed, effective_n_jobs
import os
import numpy as np
from scipy import constants

from cctbx.eltbx import henke, sasaki

from cctbx.eltbx import henke
import inspect
from cxid9114 import utils
from cxid9114 import refdata


cwd = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))

# defaults
fcalc_file = os.path.join(cwd, "fcalc_at_wave.pkl")
interp_energies = np.hstack((
            np.linspace(8940, 8952, 25),
            np.linspace(9030, 9038, 25)
            ))
wavelens_A = 1e10 * constants.h * constants.c/ \
             (constants.electron_volt* interp_energies)


def elem_fp_fdp_at_eV(elem_symbol, energy, how="henke"):
    if how == 'sasaki':
        tbl = sasaki.table(elem_symbol)
    else:
        tbl = henke.table(elem_symbol)
    keV = energy * 1e-3
    factor = tbl.at_kev(keV)
    return factor.fp(), factor.fdp()

def Yb_fp_fdp_at_eV(energy, how="henke"):
    if how=='sasaki':
        tbl = sasaki.table("Yb")
    elif how=='henke':
        tbl = henke.table("Yb")

    keV = energy * 1e-3
    factor = tbl.at_kev(keV)
    return factor.fp(), factor.fdp()


def Yb_f0_at_reso(reso):
    cman_parm = refdata.get_cromermann_parameters(70)

    if isinstance(reso, int) or isinstance(reso, float):
        reso =np.array([reso])
    Qmag = 2*np.pi/reso  # NOTE: cmann code uses 2PI convention
    cman_data = refdata.get_cmann_form_factors(cman_parm, Qmag)
    return cman_data.values()[0]

def get_scattF(wavelen_A, pdb_name, algo, dmin, ano_flag, line_filter=False):
    """
    mostly borrowed from tst_nanoBragg_basic.py
    """

    #pdblines = open(pdb_name, "r").readlines()
    #if line_filter:
    #    pdblines = [l for l in pdblines if l.startswith('ATOM')]
    from iotbx import pdb
    pdb_in = pdb.input( pdb_name) #source_info=None, lines=pdblines)
    xray_structure = pdb_in.xray_structure_simple()
    scatts = xray_structure.scatterers()

    for sc in scatts:
        expected_henke = henke.table(
            sc.element_symbol()).at_angstrom(wavelen_A)
        sc.fp = expected_henke.fp()
        sc.fdp = expected_henke.fdp()

    fcalc = xray_structure.structure_factors(
        d_min=dmin,
        algorithm=algo,
        anomalous_flag=ano_flag)
    return fcalc.f_calc()

def main(wavelens_A, wavelen_idx, jid, dmin=1.5, algo='direct', ano_flag=True, pdb_name="4bs7.pdb"):

    fcalc_at_wavelen = {}

    for i_wave in wavelen_idx:
        waveA = wavelens_A[i_wave]
        print("Job %d;  Computing scattering at wavelen %.4f ( %d/ %d )" \
              % (jid, waveA, i_wave + 1, len(wavelens_A)))
        fcalc_at_wavelen[i_wave] = \
            get_scattF(waveA,
                       pdb_name=pdb_name,
                       dmin=dmin,
                       ano_flag=ano_flag,
                       algo=algo)
    return fcalc_at_wavelen


def load_fcalc_for_default_spectrum():

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
        fcalc_at_wavelen = utils.open_flex(fcalc_file)

    return fcalc_at_wavelen



