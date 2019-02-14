import glob
from scipy.spatial import distance
import numpy as np
from scitbx.array_family import flex
from cxid9114.spots import spot_utils
from cxid9114 import utils
import sys
import pandas
import os
run = int(sys.argv[1])

fnames = glob.glob("results/run%d/*resid.pkl" % run)


res_lims = np.array([         
             np.inf,  31.40920442,  15.7386434 ,  10.53009589,
         7.93687927,   6.38960755,   5.36511518,   4.63916117,
         4.0996054 ,   3.68413423,   3.35535065,   3.08945145,
         2.87056563,   2.68770553,   2.53302433,   2.40077038,
         2.28663331,   2.1873208 ,   2.10027621,   2.02348524,
         1.95534054,   1.89454478,   1.84004023,   1.79095656])


dq_min = 0.003

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

for i_f, f in enumerate(fnames):
    d = utils.open_flex(f)
    idxA = d['residA']['indexed'] 
    idxB = d['residB']['indexed']
    
    dQA = np.array(d['residA']['dQ']) <= dq_min
    dQB = np.array(d['residB']['dQ']) <= dq_min
    
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

    all_resA.append( d['residA']['res'])
    all_resB.append( d['residB']['res'])
    all_dA.append( d['residA']['dij'])
    all_dB.append( d['residB']['dij'])
    all_qA.append( d['residA']['dQ'])
    all_qB.append( d['residB']['dQ'])

    resAnotB = np.array(d['residA']['res'])[AnotB]
    resBnotA = np.array(d['residB']['res'])[BnotA]
    resAandB = np.mean( [ np.array(d['residA']['res'])[AandB], 
                np.array(d['residB']['res'])[AandB]], axis=0)
    resAorB = np.mean( [ np.array(d['residA']['res'])[AorB], 
                np.array(d['residB']['res'])[AorB]], axis=0)

    dAnotB = np.array(d['residA']['dij'])[AnotB]
    dBnotA = np.array(d['residB']['dij'])[BnotA]
    dAandB = np.mean( [ np.array(d['residA']['dij'])[AandB], 
                np.array(d['residB']['dij'])[AandB]], axis=0)
    dAorB = np.mean( [ np.array(d['residA']['dij'])[AorB], 
                np.array(d['residB']['dij'])[AorB]], axis=0)
    
    qAnotB = np.array(d['residA']['dQ'])[AnotB]
    qBnotA = np.array(d['residB']['dQ'])[BnotA]
    qAandB = np.mean( [ np.array(d['residA']['dQ'])[AandB], 
                np.array(d['residB']['dQ'])[AandB]], axis=0)
    qAorB = np.mean( [ np.array(d['residA']['dQ'])[AorB], 
                np.array(d['residB']['dQ'])[AorB]], axis=0)
    
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
    
    R = d['refls_strong']

    Nref = len(R)
    Nidx = sum( AorB)
    frac_idx = float(Nidx) / Nref
    
    Rpp = spot_utils.refls_by_panelname(R.select( flex.bool(AorB))) 
    nC = 0
    for pid in Rpp:
        r = Rpp[pid]
        x,y,_ = spot_utils.xyz_from_refl(r)
        C = distance.pdist(zip(x,y))
        nC += np.sum( (1 < C) & (C < 7))
    run_num = int(f.split("/")[1].split("run")[1])
    shot_idx = int(f.split("_")[1])
    dist_out3.append( [nC, i_f, d['rmsd_v1'], f, run_num, shot_idx, 
            frac_idx, Nref, Nidx, nA, nB, nAandB, nAorB] )

    x,y,_ = spot_utils.xyz_from_refl(R)
    
    all_x.append(x)
    all_y.append(y)
    all_run.append( [run]*len(x))
    all_shotidx.append( [shot_idx]*len(x))
    print i_f, Nf

    #if i_f == 50:
    #    break

#
data = {
    "AnotB": np.hstack( all_AnotB),
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
    "y": np.hstack(all_y)}


odir = "ana_result/run%d" % run
if not os.path.exists(odir):
    os.makedirs(odir)
df_data = pandas.DataFrame(data)
df_data.to_pickle( odir + "/" + "run%d_refl_details.pdpkl" % run)

data2 = {
    "dQ_AnotB": all_qAnotB,
    "dQ_BnotA": all_qBnotA,
    "dQ_AandB": all_qAandB,
    "dQ_AorB": all_qAorB,

    "dij_AnotB": all_dAnotB,
    "dij_BnotA": all_dBnotA,
    "dij_AandB": all_dAandB,
    "dij_AorB": all_dAorB,

    "res_AnotB": all_resAnotB,
    "res_BnotA": all_resBnotA,
    "res_AandB": all_resAandB,
    "res_AorB": all_resAorB,
    }


utils.save_flex(data2,  odir + "/" +  "run%d_details.pkl" % run)

dist_out3 = np.array( dist_out3)
cols = ["numclose", "rmsd_v1", "fname", "run_num", "shot_idx", 
    "frac_indexed", "Nref", "Nindexed", "AnotB", "BnotA", "AandB", "AorB"]
dtypes = [np.int32, np.float32, str, np.int32, np.int32, np.float32, np.int32, 
    np.int32, np.int32, np.int32,np.int32, np.int32]
df = pandas.DataFrame( dist_out3[:,[0] + range(2,13)], columns=cols)
for i,col in enumerate(cols):
        df[col] = df[col].astype(dtypes[i])
df.to_pickle( odir + "/" + "run%d_overview.pdpkl" % run )



