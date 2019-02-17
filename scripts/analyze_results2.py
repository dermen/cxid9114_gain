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
run = int(sys.argv[1])
fix_gain = False
plot = False
odir = "/reg/d/psdm/cxi/cxid9114/res/dermen/minor_fix"
hkl_tol = 0.33
dq_min = 0.005
nom_gain = 28  # nominal photon gain for corrected CSPAD
# ------------------

fnames = glob.glob("results/run%d/*resid.pkl" % run)
mask = [m.as_numpy_array() for m in utils.open_flex("dials_mask_64panels_2.pkl")]

spec_df = pandas.read_pickle('ana_result/run%d/run%d_overview_wspec.pdpkl' % (run,run))

all_spec_hist = []
all_raw_spec = []

loader = dxtbx.load("image_files/_autogen_run%d.loc" % run )
PSANA_ENV = loader.run_mapping[run][2].env()  # important to use this env!

# these values in the dataframe  represent nominal values, but sometimes
# attenuators were pulled or inserted mid-run, hence
# these are not trustworthy, and we resort to per-event attenuation checks
atten = pandas.read_pickle("atten_cxid9114.pdpkl")
thick, trans = atten.loc[ atten.run==run, ['thickness', 'transmission']].iloc[0].values
det_ids = range(2,12)  # need reference to the motors themselves
atten_dets = { det_id:psana.Detector("XRT:DIA:MMS:%02d.RBV" % det_id, PSANA_ENV) 
            for det_id in det_ids}
# each motor represents a piece of Silicon foil, of varying thickness
atten_vals = {det_id: 20*2**i  for i, det_id in enumerate(det_ids)}  # thicknesses
def get_trans( event):
    tot_thick = 0  # total Silicon thickness is sum of thicknesses of each inserted piece
    for det_id in atten_dets.keys():
        motor_reading = atten_dets[det_id](event)
        if motor_reading is None:
            return np.nan, np.nan
        if abs(motor_reading) < 3:   # motor values hang out around 0 when foil is in
            tot_thick += atten_vals[det_id]
    trans = np.exp(- tot_thick * 84.77*1e-4 )   # attenuation coefficient for Silicon yields transmission
    return tot_thick, trans
# --

# gain map on 64 panels
gain64 = geom_utils.psana_data_to_aaron64_data(loader.gain, as_flex=False)

ISET = loader.get_imageset( loader.get_image_file())

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
# gas detector proportional to intensity
gas = psana.Detector("FEEGasDetEnergy", PSANA_ENV)

all_Amat = []

timer_start = time.time()
for i_f, f in enumerate(fnames):
    d = utils.open_flex(f)
    refls_data = d['refls_strong']
    Nref = len(refls_data)
    
    shot_idx = int(os.path.basename(f).split("_")[1])
    
    spec_df.query("shot_idx==%d" % shot_idx)['has_spec']
    has_spec = spec_df.query("shot_idx==%d" % shot_idx)['has_spec']
    if has_spec.values[0]:
        raw_spec = spec_df.query("shot_idx==%d" % shot_idx)['raw_spec'].values[0]
        spec_hist = spec_df.query("shot_idx==%d" % shot_idx)['spec_hist'].values[0]
        channA_intens = sum(spec_hist[2:5])
        channB_intens = sum(spec_hist[21:24])
    else:
        raw_spec = np.nan
        spec_hist = np.nan
        channA_intens = np.nan
        channB_intens = np.nan
    all_raw_spec.append( raw_spec)
    all_spec_hist.append(spec_hist)
    all_channA_intens.append( [channA_intens]*Nref)
    all_channB_intens.append( [channB_intens]*Nref)

    psana_ev = loader._get_event(shot_idx)

    ev_t = loader.times[shot_idx]  # event time
    sec, nsec, fid = ev_t.seconds(), ev_t.nanoseconds(), ev_t.fiducial()
    t_num, _ = utils.make_event_time( sec, nsec, fid)

    all_t_num.append( [t_num]*Nref)
    all_sec.append( [sec]*Nref)
    all_nsec.append( [nsec]*Nref)
    all_fid.append( [fid]*Nref)
    
    # get Si thickness and transmisson
    thick, trans = get_trans(psana_ev)
    all_thick.append( [thick]*Nref)
    all_trans.append( [trans]*Nref)
    
    gas_en = gas.get(psana_ev)
    if gas_en is not None: 
        all_gas11.append( [gas_en.f_11_ENRC()]*Nref)
        all_gas12.append( [gas_en.f_12_ENRC()]*Nref)
        all_gas21.append( [gas_en.f_21_ENRC()]*Nref)
        all_gas22.append( [gas_en.f_22_ENRC()]*Nref)
    else:
        all_gas11.append([np.nan]*Nref) 
        all_gas12.append([np.nan]*Nref) 
        all_gas21.append([np.nan]*Nref) 
        all_gas22.append([np.nan]*Nref) 

    refls_simA = d["refls_simA"]
    refls_simB = d["refls_simB"]

    beamA = d["beamA"]
    beamB = d["beamB"]
    detector = d["detector"]
    cryst = d["optCrystal"]
    
    ua,ub,uc,_,_,_ = cryst.get_unit_cell().parameters()
    all_ua.append([ua]*Nref)
    all_ub.append([ub]*Nref)
    all_uc.append([uc]*Nref)

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
    all_data_intens.append(refls_data["intensity.sum.value"].as_numpy_array() / nom_gain)

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
    dist_out3.append( [nC, i_f, d['rmsd_v1'], f, run, shot_idx, 
            frac_idx, Nref, Nidx, nA, nB, nAandB, nAorB] )

    x,y,_ = spot_utils.xyz_from_refl(refls_data)
    
    # flag whether a reflection was in the low gain region
    is_low.append([ gain64[pid][ int(slow-.5), int(fast-.5)] for fast,slow,pid in 
        zip( x,y,refls_data['panel'].as_numpy_array() ) ] )
     
    all_x.append(x)
    all_y.append(y)
    all_run.append( [run]*len(x))
    all_shotidx.append( [shot_idx]*len(x))
   
    # integrate with tilt plane subtraction 
    pan_data = [ pan_d.as_numpy_array() for pan_d in ISET.get_raw_data( shot_idx)]
    data_intens2, bkgrnd2, noise2, pix_per  = integrate.integrate2( 
        refls_data, mask, pan_data , gain=nom_gain)  # nominal gain!
    all_data_intens2.append( data_intens2)
    all_bkgrnd2.append( bkgrnd2)
  
    all_pix_per.append( pix_per)

    if fix_gain: 
        data32 = np.array([np.hstack([pan_data[i*2], pan_data[i*2+1]]) for i in range(32)])
        # undo the nominal low-to-high gain correction 
        data32[loader.gain] /= loader.nominal_gain_val  # this is default 6.85

        outg = gain_utils.get_gain_dists(data32, loader.gain, loader.cspad_mask,
            plot=False, norm=True) 
        
        fit = fit_utils.fit_low_gain_dist(outg[0], outg[1], plot=plot)
        fitH = fit_utils.fit_high_gain_dist(outg[2], outg[3], plot=plot)
        bgg = fitH[2].params['wid0'].value / fit[2].params['wid0'].value
        low2highG = fitH[2].params['mu1'].value / fit[2].params['mu1'].value

        lowGres = np.sum( fit[2].residual**2)
        highGres = np.sum( fitH[2].residual**2)
       
        all_lowGres.append( [lowGres]*Nref)
        all_highGres.append( [highGres]*Nref)

        all_g.append( [low2highG]*Nref)
        all_bgg.append( [bgg]*Nref)
        
        data32[loader.gain] = data32[loader.gain]* low2highG
        
        phot3_g = fitH[2].params['mu1'].value
        all_phot3_g.append( [phot3_g]*Nref)
        data64 = geom_utils.psana_data_to_aaron64_data(data32, as_flex=False)
         
        data_intens3, bkgrnd3, noise3, _ = integrate.integrate2( refls_data, mask, data64, 
                                            gain=phot3_g) 
        all_data_intens3.append( data_intens3)
        all_bkgrnd3.append( bkgrnd3)
    
        phot4_g = outg[2][ utils.smooth(outg[3], 11, 40)[100:180].argmax()+100]
        #phot4_g = 28
        all_phot4_g.append( [phot4_g]*Nref)
     
        data_intens4, bkgrnd4, noise4, _ = integrate.integrate2( refls_data, mask, data64 , 
                                            gain=phot4_g) 
        
        all_data_intens4.append( data_intens4)
        all_bkgrnd4.append( bkgrnd4)
        
        # use phot3_g with nominal low to high gain setting
        data32[loader.gain] = data32[loader.gain] * (loader.nominal_gain_val/ low2highG)
        data64 = geom_utils.psana_data_to_aaron64_data(data32, as_flex=False)
        data_intens5, bkgrnd5, noise5, _ = integrate.integrate2( refls_data, mask, data64 , 
                                            gain=phot3_g) 
        all_data_intens5.append( data_intens5)
        all_bkgrnd5.append( bkgrnd5)
        
        # use phot4_g with nominal low to high gain setting
        data_intens6, bkgrnd6, noise6, _ = integrate.integrate2( refls_data, mask, data64 , 
                                            gain=phot4_g) 
        all_data_intens6.append( data_intens6)
        all_bkgrnd6.append( bkgrnd6)

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
    "transmission" : np.hstack(all_trans),
    "Si_thicknes" : np.hstack(all_thick),
    "is_lowGain" : np.hstack(is_low), 
    
    "gas11": np.hstack(all_gas11),
    "gas12": np.hstack(all_gas12),
    "gas21": np.hstack(all_gas21),
    "gas22": np.hstack(all_gas22),
    "pix_per_spot": np.hstack(all_pix_per),
    
    "event_time": np.hstack(all_t_num),
    "event_second": np.hstack(all_sec),
    "event_nanosecond": np.hstack(all_nsec),
    "event_fiducial": np.hstack(all_fid),
    "channA_intens": np.hstack(all_channA_intens),
    "channB_intens": np.hstack(all_channB_intens),
    }

data["intens2"]= np.hstack( all_data_intens2)
data["bkgrnd2"]= np.hstack( all_bkgrnd2)
if fix_gain:
    data["intens3"]= np.hstack( all_data_intens3)
    data["intens4"]= np.hstack( all_data_intens4)
    data["intens5"]= np.hstack( all_data_intens5)
    data["intens6"]= np.hstack( all_data_intens6)
    data["bkgrnd3"]= np.hstack( all_bkgrnd3)
    data["bkgrnd4"]= np.hstack( all_bkgrnd4)
    data["bkgrnd5"]= np.hstack( all_bkgrnd5)
    data["bkgrnd6"]= np.hstack( all_bkgrnd6)
    data["lowG_fitresidual"]= np.hstack(all_lowGres)
    data["highG_fitresidual"]= np.hstack( all_highGres)
    data["low_to_high_G"]= np.hstack( all_g)
    data["phot3_g"]= np.hstack( all_phot3_g)
    data["phot4_g"]= np.hstack( all_phot4_g)
    data["bgg"]= np.hstack( all_bgg) 

odir = "%s/run%d" % ( odir,run)
if not os.path.exists(odir):
    os.makedirs(odir)
df_data = pandas.DataFrame(data)
df_data["nominal_gain"] = nom_gain
df_data["phot2_gain"] = nom_gain
df_data.to_hdf( odir + "/" + "run%d_data.pd.hdf5" % run, "reflections")
#df_data.to_pickle( odir + "/" + "run%d_refl_details.pdpkl" % run)

dist_out3 = np.array( dist_out3)
cols = ["numclose", "rmsd_v1", "fname", "run_num", "shot_idx", 
    "frac_indexed", "Nref", "Nindexed", "NAnotB", "NBnotA", "NAandB", "NAorB"]
dtypes = [np.int32, np.float32, str, np.int32, np.int32, np.float32, np.int32, 
    np.int32, np.int32, np.int32,np.int32, np.int32]
df = pandas.DataFrame( dist_out3[:,[0] + range(2,13)], columns=cols)
for i,col in enumerate(cols):
    df[col] = df[col].astype(dtypes[i])
df["Amatrix"] = all_Amat
df["spec_hist"] = all_spec_hist
df["raw_spec"] = all_raw_spec
df.to_hdf( odir + "/" + "run%d_data.pd.hdf5" % run, "overview")
#df.to_pickle( odir + "/" + "run%d_overview.pdpkl" % run )

print "TIMINGZ: %f per shot!" % ((timer_stop - timer_start) / (i_f+1))

