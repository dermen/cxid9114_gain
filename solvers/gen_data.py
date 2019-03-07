import pandas
import numpy as np


def gen_data(noise_lvl=0):

    #K = 10000**2 * 1e12
    #df = pandas.read_hdf("r62_simdata2_fixed_oversamp_labeled.hdf5","data")
    #ydata = df.D.values
    #adata = df.adata.values
    #gdata = df.gdata.values
    #LAdata = df.LA.values
    #LBdata = df.LB.values
    #PAdata = df.PA.values / 1e20
    #PBdata = df.PB.values / 1e20
    #gains = df.gain.values
    #FAdat = df.FA.abs().values
    #FBdat = df.FB.abs().values
    #np.savez("data", ydata=ydata, gains=gains, FAdat=FAdat, FVdata=FBdat, PAdata=PAdata,
    #         PBdata=PBdata,LAdata=LAdata, LBdata=LBdata, adata=adata, gdata=gdata )



    data = np.load("data.npz")
    ydata = data["ydata"]
    LAdata = data["LAdata"]
    LBdata = data["LBdata"]
    PAdata = data["PAdata"]
    PBdata = data["PBdata"]
    FAdat = data["FAdat"]
    FBdat = data["FVdata"]  # NOTE: type in stored data table
    gdata = data["gdata"]
    adata = data["adata"]
    gains = data["gains"]

    # remap adata and gdata
    amp_remap = {a:i_a for i_a, a in enumerate(set(adata))}
    adata = np.array([amp_remap[a] for a in adata])
    gain_remap = {g:i_g for i_g,g in enumerate(set(gdata))}
    gdata = np.array([gain_remap[g] for g in gdata])

    Nmeas = len( ydata)
    Namp = np.unique(adata).shape[0]
    Ngain = np.unique(gdata).shape[0]
    print "N-unknowns: 2xNhkl + Ngain = %d unknowns," % (2*Namp + Ngain)
    print "N-measurements: %d" % Nmeas

    ydata = np.random.normal( ydata, noise_lvl)

    return {"Yobs": ydata, "LA":LAdata, "LB":LBdata, "IA": FAdat**2,
            "IB":FBdat**2, "G": gains, "Aidx": adata, "Gidx": gdata,
            "PA": PAdata, "PB": PBdata}


def guess_data(data, perturbate=True):

    np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

    Namp = np.unique(data["Aidx"]).shape[0]
    Ngain = np.unique(data["Gidx"]).shape[0]

    IA = data['IA']
    IB = data['IB']
    G = data['G']

    UAvals = zip(data["Aidx"], IA)
    UBvals = zip(data["Aidx"], IB)
    UGvals = zip(data["Gidx"], G)

    UAvals = sorted(set(UAvals), key=lambda x: x[0])
    UBvals = sorted(set(UBvals), key=lambda x: x[0])
    UGvals = sorted(set(UGvals), key=lambda x: x[0])

    Avals = np.array(UAvals)[:, 1]
    Bvals = np.array(UBvals)[:, 1]
    Gvals = np.array(UGvals)[:, 1]

    if perturbate:
        AmpA_guess = np.exp( np.random.uniform( np.log(Avals)-1, np.log(Avals)+1, Namp) )
        AmpB_guess = np.exp( np.random.uniform( np.log(Bvals)-1, np.log(Bvals)+1, Namp) )
        Gain_guess = np.random.uniform(data['G'].min(), data["G"].max(), Ngain)  # going in blind here on the gain
    else:
        AmpA_guess = Avals
        AmpB_guess = Bvals
        Gain_guess = Gvals

    G = Gain_guess[data["Gidx"]]
    IA = AmpA_guess[data["Aidx"]]
    IB = AmpB_guess[data["Aidx"]]

    #return {"IA": AmpA_guess, "IB": AmpB_guess, "G": Gain_guess, }
    return {"IA":IA, "IB":IB, "G":G, "Gprm": Gain_guess, "IAprm": AmpA_guess,
            "IBprm": AmpB_guess}

