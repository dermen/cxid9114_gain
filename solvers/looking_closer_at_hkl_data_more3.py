# coding: utf-8

from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import pandas
import numpy as np
from cxid9114.sim import scattering_factors
from cxid9114.parameters import ENERGY_CONV
import pylab as plt

import sys

Hidx = int(sys.argv[1]) #107
plot = int(sys.argv[2])

waveA = ENERGY_CONV/8944.
waveB = ENERGY_CONV/9037.4

FA = scattering_factors.get_scattF(waveA, 'rocketships/003_s0_mark0_001.pdb', algo='fft', dmin=1.5, ano_flag=True)
FB = scattering_factors.get_scattF(waveB, 'rocketships/003_s0_mark0_001.pdb', algo='fft', dmin=1.5, ano_flag=True)
FAmap = {FA.indices()[i]:FA.data()[i] for i in range(len(FA.indices()))}
FBmap = {FB.indices()[i]:FB.data()[i] for i in range(len(FB.indices()))}

# constraints



def fhkl2(x, DF, Nl, Nhkl, Nscale, gain, weights=False):
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
    if weights:
        return np.sum(resid / DF.Dnoise)
    else:
        return np.sum(resid)

df = pandas.read_pickle('rocketships/all_the_goodies_wReso_and_anom_corrected.pkl')
idx = df.duplicated()
df = df.loc[~idx]
gb = df.groupby(['hAnom','kAnom','lAnom'])
H = gb.groups.keys()
d = gb.get_group(H[Hidx])
pan_ids = d.pid.unique()
df_pan = df.loc[df.pid.isin( pan_ids)]
gb_pan_shots = df_pan.groupby(['shot_idx', 'run'])

gbd = d.groupby(['shot_idx', 'run'])
shot_id = gbd.groups.keys()

print ("Finding others")
others = []
for sid in shot_id:
    d_sh = gb_pan_shots.get_group(sid)
    others.append( d_sh)
df_others = pandas.concat(others)
print "Found %d others" % len(df_others)

d_res = d.reso.values[0]

Npar = 1
Nmeas = 0
expand = 0
max_try = 10
ntry = 0
while Npar > Nmeas:
    if ntry == max_try:
        print("Cannot determine subset problem, infinite solutions could exist")
        break
    resmin = d_res-(0.03 + expand)
    resmax = d_res+(0.03 + expand)
    df_others_res = df_others.query("reso > %f and reso < %f" % (resmin, resmax))
    H_others_res = df_others_res.groupby(['hAnom','kAnom','lAnom']).groups.keys()

    DF = pandas.concat( (d,df_others_res))
    DF.reset_index(inplace=True)


    DF_hkl = DF[['hAnom','kAnom', 'lAnom']].values
    IA = [ abs(FAmap[tuple(h)])**2 for h in DF_hkl]
    IB = [ abs(FBmap[tuple(h)])**2 for h in DF_hkl]
    DF["IB"] = IB
    DF["IA"] = IA
    DF['hkl_loc'] = ["h=%d;k=%d;l=%d" % (h,k,l) for h,k,l in DF[['hAnom','kAnom','lAnom']].values]
    hkl_loc_map = {h:i for i,h in enumerate(DF.hkl_loc.unique())}
    DF['hkl_loc_id'] = [hkl_loc_map[s] for s in DF.hkl_loc]

    DF['shot_loc'] = ["shot=%d;run=%d" % (s, r) for s, r in DF[['shot_idx','run']].values]
    shot_loc_map = {pid:i for i, pid in enumerate(DF.shot_loc.unique())}
    DF['shot_loc_id'] = [shot_loc_map[s] for s in DF.shot_loc]

    GB = DF.groupby(['hAnom','kAnom','lAnom'])
    HH = GB.groups.keys()
    Nscale = len(DF.shot_loc_id.unique())
    Nhkl = len(HH)
    Npar = 2 + Nhkl*2 + Nscale
    Nmeas = len(DF)
    print("Number of unknowns=%d; Number of equations=%d"% (Npar, Nmeas))
    print
    expand += 0.005
    ntry += 1

param_array2 = np.zeros(Npar )
param_array2[:2] = 1
print "<><><><><><><><><>"
print " Parameter space "
print "<><><><><><><><><>"
print "Multiplicity of HKL measurements in subset:"
mult_data = GB.count().reset_index()[["hAnom","kAnom","lAnom","D"]]
print mult_data.rename(columns={"D": "multiplicity"})
print

print "Resolution range of data subset:"
print "resmin: %.3f, resmax: %.3f  (Angstroms)" \
    % (DF.reso.min(), DF.reso.max())
print

print "Panels seen:"
print DF.pid.unique()
print

GB_hkl_loc = DF.groupby("hkl_loc_id")
IA_init = [GB_hkl_loc.get_group(i).IA.values[0] for i in range(Nhkl)]
IB_init = [GB_hkl_loc.get_group(i).IB.values[0] for i in range(Nhkl)]
param_array2[2+Nhkl:2+2*Nhkl] = IB_init
param_array2[2:2+Nhkl] = IA_init
param_array2[2+Nhkl*2:] = 3e9  # seems like good guess from looking at data

# lots of wiggle room for the parameters..
# these are log10:
#         channel flux        I_hkl           pattern scale
bounds2 = [(-2,12)]*2 + [(-1,18)]*Nhkl*2 +[(-1,17)]*Nscale


# make the constraints
def MinRatio(x, Nhkl=Nhkl):
    IA = x[2:2+Nhkl]
    IB = x[2+Nhkl:2+Nhkl*2]
    result = min(IA/IB)
    return result

def MaxRatio(x, Nhkl=Nhkl):
    IA = x[2:2+Nhkl]
    IB = x[2+Nhkl:2+Nhkl*2]
    result = max(IA/IB)
    return result

MinC = NonlinearConstraint(MinRatio, lb=0.98,ub=0.99)
MaxC = NonlinearConstraint(MaxRatio, lb=1.01,ub=1.02)

print "Minimizing without weights"
out_hkl2_C = minimize(fhkl2, 
    x0=np.log10(param_array2), 
    args=(DF,2,Nhkl, Nscale,28, False),
    bounds=bounds2, 
    constraints=[MinC, MaxC], 
    method="trust-constr")

IA_final2C = out_hkl2_C.x[2:2+Nhkl]
IB_final2C = out_hkl2_C.x[2+Nhkl: 2+2*Nhkl]

print "Minimizing with weights"
out_hkl2_CW = minimize(fhkl2, 
    x0=np.log10(param_array2), 
    args=(DF,2,Nhkl, Nscale,28, True),
    bounds=bounds2, 
    constraints=[MinC, MaxC], 
    method="trust-constr")

IA_final2CW = out_hkl2_CW.x[2:2+Nhkl]
IB_final2CW = out_hkl2_CW.x[2+Nhkl: 2+2*Nhkl]

r_init = fhkl2( np.log10(param_array2), DF, 2, Nhkl, Nscale, 28, False)
r_fin = fhkl2( out_hkl2_C.x, DF, 2, Nhkl, Nscale, 28, False)
r_finW = fhkl2( out_hkl2_CW.x, DF, 2, Nhkl, Nscale, 28, False)
print "Initial residual: %f" % r_init
print "Final Unweighted residual using Unweighted refinement: %f" % r_fin
print "Final Unweighted residual using Weighted refinement: %f" % r_finW

print "FluxA scale factor (Unweighted): %.4f" % np.power(10, out_hkl2_C.x[0])
print "FluxB scale factor (Unweighted): %.4f" % np.power(10, out_hkl2_C.x[1])

print "FluxA scale factor (Weighted): %.4f" % np.power(10, out_hkl2_CW.x[0])
print "FluxB scale factor (Weighted): %.4f" % np.power(10, out_hkl2_CW.x[1])

np.savez("_results_%d" % Hidx,
         outC=out_hkl2_C, outCW=out_hkl2_CW, DF=DF, columns=list(DF))
DF.to_pickle("_DF_%d.pkl" % Hidx)

if plot:

    plt.figure()
    plt.plot(np.log10(IA_init), IA_final2C, 'o')
    plt.plot(np.log10(IA_init), IA_final2CW, 's')
    plt.xlabel("Initial estimates (from reference PDB)  $\log_{10} (F^2)$", fontsize=14)
    plt.ylabel("Final estimates   $\log_{10} (F^2)$", fontsize=14)
    plt.legend(("Unweighted", "Weighted"))

    plt.figure()
    scale = out_hkl2_C.x[2+Nhkl*2:]
    scaleW = out_hkl2_CW.x[2+Nhkl*2:]
    plt.bar(np.arange(Nscale) + 0., scale, width=0.4)
    plt.bar(np.arange(Nscale)+0.4, scaleW, width=0.4)
    plt.xlabel("Pattern index", fontsize=14)
    plt.ylabel(r"$\log_{10}$ (Scale)",fontsize=14)
    plt.legend(("Unweighted", "Weighted"))

    plt.show()


