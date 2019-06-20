
import pandas
from cxid9114.sim import scattering_factors
from cxid9114.parameters import  ENERGY_CONV

df = pandas.read_pickle('rocketships/all_the_goodies_wReso_and_anom_corrected.pkl')
idx = df.duplicated()
df = df.loc[~idx]

waveA = ENERGY_CONV/8944.
waveB = ENERGY_CONV/9034.7

FA = scattering_factors.get_scattF(waveA,
    'rocketships/003_s0_mark0_001.pdb', algo='fft',
    dmin=1.5, ano_flag=True)
FAmap = {FA.indices()[i]:FA.data()[i] for i in range(len(FA.indices()))}

FB = scattering_factors.get_scattF(waveB,
    'rocketships/003_s0_mark0_001.pdb', algo='fft',
    dmin=1.5, ano_flag=True)
FBmap = {FB.indices()[i]:FB.data()[i] for i in range(len(FB.indices()))}

gb = df.groupby(['hAnom', 'kAnom', 'lAnom'])
H = gb.groups.keys()

Hidx = 107
IA = abs(FAmap[H[Hidx]])**2
IB = abs(FBmap[H[Hidx]])**2


d = gb.get_group(H[Hidx])
dA = d.query("PA > 0 and PB == 0")
dB = d.query("PB > 0 and PA == 0")
dAB = d.query("PA  > 0 and PB > 0")

from IPython import embed
embed()

import numpy as np
def f(x, DF, Npan):# , Nscale):
    param_gain = x[:Npan]
    param_GA = x[Npan:] #Npan+Nscale]
    #param_GB = x[Npan+Nscale:]
    gain = param_gain[DF.pan_id_id]
    scale = param_GA[DF.shot_loc_id]
    #GB = param_GB[DF.pan_id_id]
    Yobs = DF.D / gain
    Ymodel = scale*( DF.LA*DF.PA/1e20 *DF.IA + DF.LB*DF.PB/1e20*DF.IB)
    resid = (Yobs - Ymodel)**2
    return np.sum(resid)


