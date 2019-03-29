import h5py
import glob
from scipy.spatial import distance
import numpy as np
from scitbx.array_family import flex
from cxid9114.refine import metrics
from cxid9114.spots import spot_utils, integrate
from cxid9114 import utils, gain_utils, fit_utils
from cxid9114.geom import geom_utils
import sys
import pylab as plt
import pandas
import os
import dxtbx
import time
import psana


# ------------------
run = 62 #int(sys.argv[1])
odir = "/reg/d/psdm/cxi/cxid9114/res/dermen/simulation"
hkl_tol = 0.33
dq_min = 0.005
rank = int(sys.argv[1])
n_jobs = int(sys.argv[2])
# ------------------

fnames = glob.glob("results/run%d/*simtest.pkl" % run)

Nf = len(fnames)

dist_out3 = []
all_resAnotB = []
all_resBnotA = []
all_resAandB = []
all_resAorB = []

all_dAnotB = []
all_dBnotA = []
all_dAandB = []
all_dAorB = []

all_qAnotB = []
all_qBnotA = []
all_qAandB = []
all_qAorB = []

all_AnotB = []
all_BnotA = []
all_AandB = []
all_AorB = []

all_x = []
all_y = []
all_run = []
all_resA = []
all_resB = []
all_qA = []
all_qB = []
all_dA = []
all_dB = []
all_shotidx = []


all_refA_idx = []
all_refB_idx = []
all_refA_I =[]
all_refB_I = []
all_refA_pid = []
all_refB_pid = []

all_data_pid = []
all_hklA = []
all_hklA_res = []
all_hklB = []
all_hklB_res = []
all_dvecAi = []
all_dvecBi = []
all_dvecAj = []
all_dvecBj = []

all_dvecAqx = []
all_dvecAqy = []
all_dvecAqz = []

all_dvecBqx = []
all_dvecBqy = []
all_dvecBqz = []

all_data_intens = []  
all_data_intens2 = []
all_data_intens3 = []
all_data_intens4 = []
all_data_intens5 = []
all_data_intens6 = []
all_g = []
all_bgg = []
all_lowGres = []
all_highGres = []
all_phot3_g = []
all_phot4_g = []
all_bkgrnd2 = []
all_bkgrnd3 = []
all_bkgrnd4 = []
all_bkgrnd5 = []
all_bkgrnd6 = []

all_trans = []
all_thick = []
is_low = []
all_gas11 = []
all_gas12 = []
all_gas21 = []
all_gas22 = []
all_pix_per = []

all_t_num = []
all_sec =[]
all_nsec = []
all_fid = []

all_refl_idx = []
all_ua = []
all_ub = []
all_uc = []
all_channA_intens = []
all_channB_intens = []
all_gain = []
all_Amat = []

timer_start = time.time()
for i_f, f in enumerate(fnames):
    if i_f % n_jobs != rank:
        continue
    d = utils.open_flex(f)
    refls_data = d['refls_data']
    Nref = len(refls_data)
    
    shot_idx = int(os.path.basename(f).split("_")[1])
    
    channA_intens, channB_intens = d['flux_data']
    all_channA_intens.append( [channA_intens]*Nref)
    all_channB_intens.append( [channB_intens]*Nref)

    refls_simA = d["refls_simA"]
    refls_simB = d["refls_simB"]

    beamA = d["beamA"]
    beamB = d["beamB"]
    detector = d["detector"]
    cryst = d["crystalAB"]
    
    ua,ub,uc,_,_,_ = cryst.get_unit_cell().parameters()
    all_ua.append([ua]*Nref)
    all_ub.append([ub]*Nref)
    all_uc.append([uc]*Nref)

    all_gain.append([d['gain']]*Nref)

    all_Amat.append( list(cryst.get_A()))
    residA = metrics.check_indexable2( refls_data, refls_simA, detector, beamA, cryst, hkl_tol)
    residB = metrics.check_indexable2( refls_data, refls_simB, detector, beamB, cryst, hkl_tol)
    
    all_refA_idx.append( residA['sim_refl_idx'])
    all_refA_I.append( residA['sim_intens'])
    all_refA_pid.append( residA['sim_pid'])
    
    all_refB_idx.append( residB['sim_refl_idx'])
    all_refB_I.append( residB['sim_intens'])
    all_refB_pid.append( residB['sim_pid'])

    all_hklA.append( residA['hkl'])  
    all_hklA_res.append( residA['hkl_res'])  
    
    all_hklB.append( residB['hkl'])  
    all_hklB_res.append( residB['hkl_res'])  
    
    all_data_pid.append( refls_data['panel'].as_numpy_array() )  
    
    dvecAi, dvecAj = zip(*[[i[0], i[1]]  if not np.any( np.isnan( i)) 
                        else [np.nan, np.nan] for i in residA['dvecij']])
    all_dvecAi.append( dvecAi )
    all_dvecAj.append( dvecAj )
    
    dvecBi, dvecBj = zip(*[[i[0], i[1]]  if not np.any( np.isnan( i)) 
                        else [np.nan, np.nan] for i in residB['dvecij']])
    all_dvecBi.append( dvecBi )
    all_dvecBj.append( dvecBj )
    
    dvecAqx, dvecAqy, dvecAqz= zip(*[[i[0], i[1], i[2]]  if not np.any( np.isnan( i)) 
                        else [np.nan, np.nan, np.nan] for i in residA['dvecQ']])
    all_dvecAqx.append( dvecAqx )
    all_dvecAqy.append( dvecAqy )
    all_dvecAqz.append( dvecAqz )
    
    dvecBqx, dvecBqy, dvecBqz= zip(*[[i[0], i[1], i[2]]  if not np.any( np.isnan( i)) 
                        else [np.nan, np.nan, np.nan] for i in residB['dvecQ']])
    all_dvecBqx.append( dvecBqx )
    all_dvecBqy.append( dvecBqy )
    all_dvecBqz.append( dvecBqz )

    all_refl_idx.append( range(Nref))
    all_data_intens.append(refls_data["intensity.sum.value"].as_numpy_array())

    idxA = residA['indexed'] 
    idxB = residB['indexed']
   
    dQA = np.array(residA['dQ']) <= dq_min
    dQB = np.array(residB['dQ']) <= dq_min
   
    idxA = np.logical_and( dQA, idxA)
    idxB = np.logical_and( dQB, idxB)
    
    AorB = np.logical_or(idxA, idxB)
    AnotB = np.logical_and(idxA, ~idxB)
    BnotA = np.logical_and(idxB, ~idxA)
    AandB = np.logical_and( idxA, idxB)
    
    all_AorB.append( AorB)
    all_AandB.append( AandB)
    all_AnotB.append( AnotB)
    all_BnotA.append( BnotA)

    all_resA.append( residA['res'])
    all_resB.append( residB['res'])
    all_dA.append( residA['dij'])
    all_dB.append( residB['dij'])
    all_qA.append( residA['dQ'])
    all_qB.append( residB['dQ'])

    resAnotB = np.array(residA['res'])[AnotB]
    resBnotA = np.array(residB['res'])[BnotA]
    resAandB = np.mean( [ np.array(residA['res'])[AandB], 
                np.array(residB['res'])[AandB]], axis=0)
    resAorB = np.mean( [ np.array(residA['res'])[AorB], 
                np.array(residB['res'])[AorB]], axis=0)

    dAnotB = np.array(residA['dij'])[AnotB]
    dBnotA = np.array(residB['dij'])[BnotA]
    dAandB = np.mean( [ np.array(residA['dij'])[AandB], 
                np.array(residB['dij'])[AandB]], axis=0)
    dAorB = np.mean( [ np.array(residA['dij'])[AorB], 
                np.array(residB['dij'])[AorB]], axis=0)
    
    qAnotB = np.array(residA['dQ'])[AnotB]
    qBnotA = np.array(residB['dQ'])[BnotA]
    qAandB = np.mean( [ np.array(residA['dQ'])[AandB], 
                np.array(residB['dQ'])[AandB]], axis=0)
    qAorB = np.mean( [ np.array(residA['dQ'])[AorB], 
                np.array(residB['dQ'])[AorB]], axis=0)
    
    all_resAnotB.append( resAnotB)
    all_resBnotA.append( resBnotA)
    all_resAandB.append( resAandB)
    all_resAorB.append( resAorB)
    
    all_dAnotB.append( dAnotB)
    all_dBnotA.append( dBnotA)
    all_dAandB.append( dAandB)
    all_dAorB.append( dAorB)
    
    all_qAnotB.append( qAnotB)
    all_qBnotA.append( qBnotA)
    all_qAandB.append( qAandB)
    all_qAorB.append( qAorB)

    nA = AnotB.sum()
    nB = BnotA.sum()
    nAandB = AandB.sum()
    nAorB = AorB.sum()
    
    Nidx = sum( AorB)
    frac_idx = float(Nidx) / Nref
    
    Rpp = spot_utils.refls_by_panelname(refls_data.select( flex.bool(AorB))) 
    nC = 0
    for pid in Rpp:
        r = Rpp[pid]
        x,y,_ = spot_utils.xyz_from_refl(r)
        C = distance.pdist(zip(x,y))
        nC += np.sum( (1 < C) & (C < 7))
    dist_out3.append( [nC, i_f, np.nan, f, run, shot_idx, 
            frac_idx, Nref, Nidx, nA, nB, nAandB, nAorB] )

    x,y,_ = spot_utils.xyz_from_refl(refls_data)
    
    # flag whether a reflection was in the low gain region
    #is_low.append([ gain64[pid][ int(slow-.5), int(fast-.5)] for fast,slow,pid in 
    #    zip( x,y,refls_data['panel'].as_numpy_array() ) ] )
     
    all_x.append(x)
    all_y.append(y)
    all_run.append( [run]*len(x))
    all_shotidx.append( [shot_idx]*len(x))
   
    # integrate with tilt plane subtraction 
    #pan_data = [ pan_d.as_numpy_array() for pan_d in ISET.get_raw_data( shot_idx)]
    #data_intens2, bkgrnd2, noise2, pix_per  = integrate.integrate2( 
    #    refls_data, mask, pan_data , gain=nom_gain)  # nominal gain!
    #all_data_intens2.append( data_intens2)
    #all_bkgrnd2.append( bkgrnd2)
    pix_per = spot_utils.npix_per_spot(refls_data) 
    all_pix_per.append( pix_per)

    print i_f, Nf
#

timer_stop = time.time()

all_hklA= np.vstack( all_hklA)
all_hklB= np.vstack( all_hklB)
data = {
    "AnotB": np.hstack( all_AnotB),
    "AorB": np.hstack( all_AorB),
    "BnotA": np.hstack( all_BnotA),
    "AandB": np.hstack( all_AandB),
    "AorB": np.hstack( all_AorB),
    "run": np.hstack(all_run),
    "shot_idx": np.hstack(all_shotidx),
    "resA": np.hstack(all_resA),
    "resB": np.hstack(all_resB),
    "dijA": np.hstack(all_dA),
    "dijB": np.hstack(all_dB),
    "dqA": np.hstack(all_qA),
    "dqB": np.hstack(all_qB),
    "x": np.hstack(all_x),
    "y": np.hstack(all_y),
    "ucell_a": np.hstack(all_ua), 
    "ucell_b": np.hstack(all_ub), 
    "ucell_c": np.hstack(all_uc), 
    "gain": np.hstack(all_gain), 

    "ref_strong_idx": np.hstack(all_refl_idx),
    "refA_idx" :np.hstack(all_refA_idx),
    "refB_idx" :np.hstack(all_refB_idx),
    "intensA" :np.hstack(all_refA_I) ,
    "intensB" :np.hstack(all_refB_I) ,
    "pidA" :np.hstack(all_refA_pid), 
    "pidB" :np.hstack(all_refB_pid) ,
    "pid" :np.hstack(all_data_pid) ,
    "hA" : all_hklA[:,0],
    "kA" : all_hklA[:,1],
    "lA" : all_hklA[:,2],
    "hB" : all_hklB[:,0],
    "kB" : all_hklB[:,1],
    "lB" : all_hklB[:,2],
    "hklA_res" :np.hstack(all_hklA_res), 
    "hklB_res" :np.hstack(all_hklB_res), 
    "dvecAi" :np.hstack(all_dvecAi) ,
    "dvecBi" :np.hstack(all_dvecBi), 
    "dvecAj" :np.hstack(all_dvecAj), 
    "dvecBj" :np.hstack(all_dvecBj), 
    
    "dvecAqx" :np.hstack(all_dvecAqx), 
    "dvecAqy" :np.hstack(all_dvecAqy), 
    "dvecAqz" :np.hstack(all_dvecAqz), 

    "dvecBqx" :np.hstack(all_dvecBqx), 
    "dvecBqy" :np.hstack(all_dvecBqy), 
    "dvecBqz" :np.hstack(all_dvecBqz), 
    
    "intens" :np.hstack(all_data_intens),
    
    "pix_per_spot": np.hstack(all_pix_per),
    
    "channA_intens": np.hstack(all_channA_intens),
    "channB_intens": np.hstack(all_channB_intens),
    }


odir = "%s/run%d" % ( odir,run)
if not os.path.exists(odir):
    os.makedirs(odir)
outpath = os.path.join( odir, "run%d_data_rank%d.pd.hdf5" % (run,rank))

df_data = pandas.DataFrame(data)
df_data.to_hdf( outpath, "reflections")

dist_out3 = np.array( dist_out3)
cols = ["numclose", "rmsd_v1", "fname", "run_num", "shot_idx", 
    "frac_indexed", "Nref", "Nindexed", "NAnotB", "NBnotA", "NAandB", "NAorB"]
dtypes = [np.int32, np.float32, str, np.int32, np.int32, np.float32, np.int32, 
    np.int32, np.int32, np.int32,np.int32, np.int32]
df = pandas.DataFrame( dist_out3[:,[0] + range(2,13)], columns=cols)
for i,col in enumerate(cols):
    df[col] = df[col].astype(dtypes[i])
df["Amatrix"] = all_Amat
df.to_hdf(outpath, "overview")

print "TIMINGZ: %f per shot!" % ((timer_stop - timer_start) / (i_f+1))

