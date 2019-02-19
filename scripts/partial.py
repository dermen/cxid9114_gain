# coding: utf-8
import pandas
from scitbx.matrix import sqr
from cxid9114 import utils
from cxid9114.spots import spot_utils
from copy import deepcopy
from cxid9114 import parameters
import numpy as np
import sys

waveA= parameters.ENERGY_CONV / 8944.
waveB = parameters.ENERGY_CONV / 9034.7

# from looking at the simulations
dom_eff = np.power(79*22 *79*22 *38*22, 1/3.)

# mosaic angle
mos_ang = 0.03

run = int(sys.argv[1])
data_f = "/reg/d/psdm/cxi/cxid9114/res/dermen/run%d/run%d_data.pd.hdf5" % (run,run)
det = utils.open_flex("/reg/d/psdm/cxi/cxid9114/scratch/dermen/idx/mad/ref1_det.pkl")
beam = utils.open_flex("/reg/d/psdm/cxi/cxid9114/scratch/dermen/idx/mad/ref3_beam.pkl")

beamA = deepcopy(beam)
beamB = deepcopy(beam)
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

df_refls = pandas.read_hdf(data_f, "reflections")
df_cryst = pandas.read_hdf(data_f, "overview")


for c in ["QA_xobs", "QA_yobs", "QA_zobs"]\
    +["QA_xewald", "QA_yewald", "QA_zewald"]\
    + ["QB_xobs", "QB_yobs", "QB_zobs"]\
    +["QB_xewald", "QB_yewald", "QB_zewald"]:
    df_refls[c] = np.nan

u_idx = df_refls.shot_idx.unique()
for i_idx, idx in enumerate(u_idx):
    
    refls = df_refls.query("shot_idx==%d" % idx)
    cryst = df_cryst.query("shot_idx==%d" % idx)
    Amat = sqr(cryst.Amatrix.values[0]).as_numpy_array()

    hklA = refls[['hA', 'kA','lA']]
    hklB = refls[['hB', 'kB','lB']]

    QA_ewald = np.dot( Amat, hklA.values.T).T
    QB_ewald = np.dot( Amat, hklB.values.T).T

    fs,ss,pids = zip(*refls[['x','y', 'pid']].values)
    pids = map(int, pids)
    QA_obs = spot_utils.fs_ss_to_q(fs,ss,pids,det,beamA)
    QB_obs = spot_utils.fs_ss_to_q(fs,ss,pids,det,beamB)

    resoA = refls['resA']
    resoB = refls['resB']
    rS_chanA = 1./dom_eff + .5 * mos_ang /  resoA.values
    rS_chanB = 1./dom_eff + .5 * mos_ang /  resoB.values

    rH_chanA = np.sqrt(np.sum((QA_obs - QA_ewald)**2,1))
    rH_chanB = np.sqrt(np.sum((QB_obs - QB_ewald)**2,1))
    ParA = (rS_chanA**2- rH_chanA**2) / (dom_eff * rS_chanA**3)
    ParB = (rS_chanB**2- rH_chanB**2) / (dom_eff * rS_chanB**3)
    
    df_refls.loc[refls.index, "rS_chanA"] = rS_chanA
    df_refls.loc[refls.index, "rS_chanB"] = rS_chanB
    df_refls.loc[refls.index, "rH_chanA"] = rH_chanA
    df_refls.loc[refls.index, "rH_chanB"] = rH_chanB
    df_refls.loc[refls.index, "HS_ratioA"] =  np.abs(rH_chanA/rS_chanA)
    df_refls.loc[refls.index, "HS_ratioB"] =  np.abs(rH_chanB/rS_chanB)
    
    df_refls.loc[refls.index, "partialityA"] = ParA
    df_refls.loc[refls.index, "partialityB"] = ParB

    df_refls.loc[refls.index, ["QA_xobs", "QA_yobs", "QA_zobs"]] = QA_obs
    df_refls.loc[refls.index, ["QB_xobs", "QB_yobs", "QB_zobs"]] = QB_obs
    
    df_refls.loc[refls.index, ["QA_xewald", "QA_yewald", "QA_zewald"]] = QA_ewald
    df_refls.loc[refls.index, ["QB_xewald", "QB_yewald", "QB_zewald"]] = QB_ewald
   
    print i_idx, len( u_idx)

AandB = df_refls.query("AandB")
AnotB = df_refls.query("AnotB")
BnotA = df_refls.query("BnotA")
df_refls.loc[AandB.index, "partiality"] = AandB.partialityA*.5 + AandB.partialityB*.5
df_refls.loc[AnotB.index, "partiality"] = AnotB.partialityA
df_refls.loc[BnotA.index, "partiality"] = BnotA.partialityB

df_refls.loc[AandB.index, "HS_ratio"] = AandB.HS_ratioA*.5 + AandB.HS_ratioB*.5
df_refls.loc[AnotB.index, "HS_ratio"] = AnotB.HS_ratioA
df_refls.loc[BnotA.index, "HS_ratio"] = BnotA.HS_ratioB

df_refls.to_hdf(data_f, "reflections")

