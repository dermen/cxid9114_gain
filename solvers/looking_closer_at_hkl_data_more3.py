# coding: utf-8

from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import pandas
import numpy as np
from cxid9114.sim import scattering_factors
from cxid9114.parameters import ENERGY_CONV
from IPython import embed

# constraints    
def MinRatio(x):
    Nhkl = 45
    IA = x[2:2+Nhkl]
    IB = x[2+Nhkl:2+Nhkl*2]
    result = min(IA/IB)
    return result

def MaxRatio(x):
    Nhkl = 45
    IA = x[2:2+Nhkl]
    IB = x[2+Nhkl:2+Nhkl*2]
    result = max(IA/IB)
    return result


def fhkl2(x, DF, Nl, Nhkl, Nscale, gain):
    param_l = np.power(10, x[:Nl] )
    param_hklA = np.power(10, x[Nl:Nl+Nhkl])
    param_hklB = np.power(10, x[Nl+Nhkl:Nl+Nhkl+Nhkl])
    param_scale = np.power(10,x[Nl+Nhkl+Nhkl:])
    scale = param_scale[DF.shot_loc_id]
    IA = param_hklA[DF.hkl_loc_id]
    IB = param_hklB[DF.hkl_loc_id]
    a,b = param_l
    Yobs = DF.D / gain
    Ymodel = scale*( (a*DF.LA )*DF.PA/1e20*IA + (b*DF.LB)*DF.PB/1e20*IB)
    resid = (Yobs - Ymodel)**2
    return np.sum(resid)

df = pandas.read_pickle('rocketships/all_the_goodies_wReso_and_anom_corrected.pkl')
idx = df.duplicated()
df = df.loc[~idx]
gb = df.groupby(['hAnom','kAnom','lAnom'])
H = gb.groups.keys()
Hidx = 107
d = gb.get_group(H[Hidx])
pan_ids = d.pid.unique()
df_pan = df.loc[df.pid.isin( pan_ids)]
gb_pan_shots = df_pan.groupby(['shot_idx', 'run'])

gbd = d.groupby(['shot_idx', 'run'])
shot_id = gbd.groups.keys()

others = []
for sid in shot_id:
    d_sh = gb_pan_shots.get_group(sid)
    #d_sh = d_sh.loc[ d_sh.pid.isin(pan_ids)]
    print len( d_sh)
    others.append( d_sh)
df_others = pandas.concat(others)
df_others_res = df_others.query("reso > 3.05 and reso < 3.11")
H_others_res = df_others_res.groupby(['hAnom','kAnom','lAnom']).groups.keys()

embed()

DF = pandas.concat( (d,df_others_res))
DF.reset_index(inplace=True)

waveA = ENERGY_CONV/8944.
waveB = ENERGY_CONV/9037.4
FA = scattering_factors.get_scattF(waveA, 'rocketships/003_s0_mark0_001.pdb', algo='fft', dmin=1.5, ano_flag=True)
FB = scattering_factors.get_scattF(waveB, 'rocketships/003_s0_mark0_001.pdb', algo='fft', dmin=1.5, ano_flag=True)

FAmap = {FA.indices()[i]:FA.data()[i] for i in range(len(FA.indices()))}
FBmap = {FB.indices()[i]:FB.data()[i] for i in range(len(FB.indices()))}

DF_hkl = DF[['hAnom','kAnom', 'lAnom']].values
IA = [ abs(FAmap[tuple(h)])**2 for h in DF_hkl]
IB = [ abs(FBmap[tuple(h)])**2 for h in DF_hkl]
DF["IB"] = IB
DF["IA"] = IA
DF['hkl_loc'] = ["h=%d;k=%d;l=%d" % (h,k,l) for h,k,l in DF[['hAnom','kAnom','lAnom']].values]
hkl_loc_map = {h:i for i,h in enumerate(DF.hkl_loc.unique())}
DF['hkl_loc_id'] = [hkl_loc_map[s] for s in DF.hkl_loc]

DF['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in DF[['shot_idx','run']].values]
shot_loc_map = {pid:i for i,pid in enumerate(DF.shot_loc.unique())}
DF['shot_loc_id'] = [shot_loc_map[s] for s in DF.shot_loc]

embed()

MinC = NonlinearConstraint(MinRatio, lb=0.98,ub=0.99) 
MaxC = NonlinearConstraint(MaxRatio, lb=1.01,ub=1.02) 

GB = DF.groupby(['hAnom','kAnom','lAnom'])
HH = GB.groups.keys()
Nscale = len(DF.shot_loc_id.unique())
Nhkl = len(HH)
param_array2 = np.zeros( 2 + Nhkl*2 + Nscale)
param_array2[:2] = 1  

GB_hkl_loc = DF.groupby("hkl_loc_id")
IA_init = [GB_hkl_loc.get_group(i).IA.values[0] for i in range(Nhkl)]
IB_init = [GB_hkl_loc.get_group(i).IB.values[0] for i in range(Nhkl)]
param_array2[2+Nhkl:2+2*Nhkl] = IB_init
param_array2[2:2+Nhkl] = IA_init
param_array2[2+Nhkl*2:] = 3e9

bounds2 = [(0.2,12)]*2 + [(1,18)]*Nhkl*2 +[(7,14)]*Nscale

out_hkl2_C = minimize(fhkl2, 
    x0=np.log10(param_array2), 
    args=(DF,2,Nhkl, Nscale,28),
    bounds=bounds2, 
    constraints=[MinC, MaxC], 
    method="trust-constr")

IA_final2C = out_hkl2_C.x[2:2+Nhkl]
IB_final2C = out_hkl2_C.x[2+Nhkl: 2+2*Nhkl]

#figure()
#plot( np.log10(IA_init),np.log10(IB_init), 's')
#plot( IA_final,IB_final, 'o')
#plot( IA_final2C,IB_final2C, 'd')
#figure()
#plot( log10(IA_init), IA_final2C)
#clf()
#plot( log10(IA_init), IA_final2C,'.')
#get_ipython().magic(u'save looking_closer_at_hkl_data_more3.py 867-927')
