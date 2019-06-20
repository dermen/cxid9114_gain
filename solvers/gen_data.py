import numpy as np

def gen_real_data_and_guess(gain=1):
    """data from CXID9114"""
    data = np.load("real_dataA.npz")
    gdata = data["gdata"]
    adata = data["adata"]
    
    ydata = data["ydata"] / gain
    LAdata = data["LA"]
    LBdata = data["LB"]
    PAdata = data["PA"]
    PBdata = data["PB"]
    ynoise = data["ynoise"]

    IA_guess = data["Iprm"] / gain
    IB_guess = data["Iprm"] / gain
    # TODO: consider making this better
    Ngain = len(set(gdata))  # guess["Gprm"].shape[0]
    GainA_guess = np.random.uniform(LAdata.min(), LAdata.max(), Ngain)
    GainB_guess = np.random.uniform(LBdata.min(), LBdata.max(), Ngain)

    GAdata = GainA_guess[gdata]
    GBdata = GainB_guess[gdata]

    DATA = {"Yobs": ydata, "LA":LAdata, "LB":LBdata,
            "Aidx": adata, "Gidx": gdata, "GAdata": GAdata, "GBdata": GBdata,
            "PA": PAdata, "PB": PBdata, "Ysig": ynoise}

    GUESS = {"GAprm": GainA_guess, "GBprm": GainB_guess, "IAprm": IA_guess,
            "IBprm": IB_guess}

    return {"data": DATA, "guess": GUESS}


def gen_truth_for_data():
    data = np.load("real_dataA.npz")
    #data = np.load("rocketships/truth_TR_synch.npz")
    IAprm = data['IAprm']
    IBprm = data['IBprm']
    from IPython import embed
    embed()
    return {"IAprm": IAprm, "IBprm": IBprm}


def gen_data(noise_lvl=0, Nshot_max = None, load_hkl=False):

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

    gdata = data["gdata"]
    if Nshot_max is not None:
        sel = gdata < Nshot_max
    else:
        sel = np.ones(gdata.shape[0], bool)
    gdata = data["gdata"][sel]
    ydata = data["ydata"][sel]

    gains = data["gains"][sel]
    LAdata = data["LAdata"][sel]
    LBdata = data["LBdata"][sel]

    GAdata = gains*LAdata
    GBdata = gains*LBdata

    PAdata = data["PAdata"][sel]
    PBdata = data["PBdata"][sel]
    FAdat = data["FAdat"][sel]
    FBdat = data["FVdata"][sel]  # NOTE: type in stored data table

    # remap adata and gdata
    if Nshot_max is not None:
        amp_remap = {a:i_a for i_a, a in enumerate(set(adata))}
        adata = np.array([amp_remap[a] for a in adata])
        gain_remap = {g:i_g for i_g,g in enumerate(set(gdata))}
        gdata = np.array([gain_remap[g] for g in gdata])

    Nmeas = len(ydata)
    Namp = np.unique(adata).shape[0]
    Ngain = np.unique(gdata).shape[0]
    print "N-unknowns: 2xNhkl + 2xNgain = %d unknowns," % (2*Namp + 2*Ngain)
    print "N-measurements: %d" % Nmeas

    ydata = np.random.normal(ydata, noise_lvl)

    if load_hkl:
        data_hkl = np.load("data_hkl")
        h = data_hkl["h"]
        k = data_hkl["k"]
        l = data_hkl["l"]
    else:
        h = k = l = None

    return {"Yobs": ydata, "LA":LAdata, "LB":LBdata, "IA": FAdat**2,
            "IB":FBdat**2, "G": gains, "Aidx": adata, "Gidx": gdata,
            "PA": PAdata, "PB": PBdata, "h": h, "k": k, "l": l,
            "GA": GAdata, "GB": GBdata}


def guess_data(data, perturbate=True, perturbate_factor=.1):

    np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

    Namp = np.unique(data["Aidx"]).shape[0]
    Ngain = np.unique(data["Gidx"]).shape[0]

    IA = data['IA']
    IB = data['IB']
    GA = data['GA']
    GB = data['GB']

    UAvals = zip(data["Aidx"], IA)
    UBvals = zip(data["Aidx"], IB)
    UGAvals = zip(data["Gidx"], GA)
    UGBvals = zip(data["Gidx"], GB)

    UAvals = sorted(set(UAvals), key=lambda x: x[0])
    UBvals = sorted(set(UBvals), key=lambda x: x[0])
    UGAvals = sorted(set(UGAvals), key=lambda x: x[0])
    UGBvals = sorted(set(UGBvals), key=lambda x: x[0])

    Avals = np.array(UAvals)[:, 1]
    Bvals = np.array(UBvals)[:, 1]
    GAvals = np.array(UGAvals)[:, 1]
    GBvals = np.array(UGBvals)[:, 1]

    if perturbate:
        _p = perturbate_factor
        AmpA_guess = np.exp(np.random.uniform( np.log(Avals)-_p, np.log(Avals)+_p, Namp) )
        AmpB_guess = np.exp(np.random.uniform( np.log(Bvals)-_p, np.log(Bvals)+_p, Namp) )
        GainA_guess = np.random.uniform(data['GA'].min(), data["GA"].max(), Ngain)  # going in blind here on the gain
        GainB_guess = np.random.uniform(data['GB'].min(), data["GB"].max(), Ngain)  # going in blind here on the gain
    else:
        AmpA_guess = Avals
        AmpB_guess = Bvals
        GainA_guess = GAvals
        GainB_guess = GBvals

    GA = GainA_guess[data["Gidx"]]
    GB = GainB_guess[data["Gidx"]]
    IA = AmpA_guess[data["Aidx"]]
    IB = AmpB_guess[data["Aidx"]]

    return {"IA":IA, "IB":IB, "GA":GA, "GB": GB, "GAprm": GainA_guess,
            "GBprm": GainB_guess, "IAprm": AmpA_guess,
            "IBprm": AmpB_guess}

