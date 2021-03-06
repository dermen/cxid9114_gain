import numpy as np


def gen_data(input_file, noise_lvl=0, load_hkl=False, Nshot_max = None):

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

    data = np.load(input_file)
    #data = np.load("C2_all.npz")
    #data = np.load("CS2_2.npz")
    #data = np.load("perf2_G.npz")
    #data = np.load("perf22.npz")
    #data = np.load("perf.npz")
    #data = np.load("data.npz")

    gdata = data["gdata"]
    if Nshot_max is not None:
        sel = gdata < Nshot_max
    else:
        sel = np.ones(gdata.shape[0], bool)
    gdata = data["gdata"][sel]
    ydata = data["ydata"][sel]
    LAdata = data["LAdata"][sel]
    LBdata = data["LBdata"][sel]
    PAdata = data["PAdata"][sel]
    PBdata = data["PBdata"][sel]
    FAdat = data["FAdata"][sel]  # NOTE: type in stored data table
    FBdat = data["FBdata"][sel]  # NOTE: type in stored data table

    #FAdat = data["FAdat"][sel]
    #FBdat = data["FVdata"][sel]  # NOTE: type in stored data table
    adata = data["adata"][sel]
    gains = data["gains"][sel]

    # remap adata and gdata
    if Nshot_max is not None:
        amp_remap = {a: i_a for i_a, a in enumerate(set(adata))}
        adata = np.array([amp_remap[a] for a in adata])
        gain_remap = {g: i_g for i_g, g in enumerate(set(gdata))}
        gdata = np.array([gain_remap[g] for g in gdata])

    Nmeas = len(ydata)
    Namp = np.unique(adata).shape[0]
    Ngain = np.unique(gdata).shape[0]
    print "N-unknowns: 2xNhkl + Ngain = %d unknowns," % (2*Namp + Ngain)
    print "N-measurements: %d" % Nmeas
    print "Ngain", Ngain

    ydata = np.random.normal(ydata, noise_lvl)

    if load_hkl:
        data_hkl = np.load("data_hkl")
        h = data_hkl["h"]
        k = data_hkl["k"]
        l = data_hkl["l"]
    else:
        h = k = l = None

    try:
        W = np.nan_to_num(data["Weights"])
    except:
        W = np.ones_like(ydata)

    return {"Yobs": ydata, "LA":LAdata, "LB":LBdata, "IA": FAdat**2,
            "IB": FBdat**2, "G": gains, "Aidx": adata, "Gidx": gdata,
            "PA": PAdata, "PB": PBdata, "Iprm": data["Iprm"], "weights": W}


def guess_data(data, perturbate=True, use_Iguess=False, perturbate_factor=.1, hmap=None, reso_bins=None):

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
        _p = perturbate_factor
        AmpA_guess = np.exp( np.random.uniform(np.log(Avals)-_p, np.log(Avals)+_p, Namp))
        AmpB_guess = np.exp( np.random.uniform(np.log(Bvals)-_p, np.log(Bvals)+_p, Namp))
        if reso_bins is not None and hmap is not None:
            a = b = 79
            c = 38
            reso = np.zeros(Namp)
            hmap2 = {v: k for k, v in hmap.items()}
            for i in range(Namp):
                h, k, l = hmap2[i]
                reso[i] = 1/np.sqrt(h*h/a/a+k*k/b/b+l*l/c/c)

            reso_idx = np.digitize(reso, reso_bins)-1
            unique_bin_idx = np.unique(reso_idx)
            for i_bin in unique_bin_idx:
                if i_bin == -1 or i_bin == len(unique_bin_idx)-1:
                    continue

                F_idx = reso_idx == i_bin
                resoFAvals = AmpA_guess[F_idx]
                resoFBvals = AmpB_guess[F_idx]

                all_resoVals = np.hstack((resoFAvals, resoFBvals))

                numF = F_idx.sum()
                AmpA_guess[F_idx] = np.random.choice(all_resoVals,
                                                size=numF, replace=True)
                AmpB_guess[F_idx] = np.random.choice(all_resoVals,
                                                size=numF, replace=True)

        Gain_guess = np.random.uniform(Gvals*.5, Gvals*2)
        #Gain_guess = np.random.uniform(1, 10, Ngain)
    else:
        AmpA_guess = Avals
        AmpB_guess = Bvals
        Gain_guess = Gvals

    if use_Iguess:
        AmpA_guess = data["Iprm"]
        AmpB_guess = data["Iprm"]

    G = Gain_guess[data["Gidx"]]
    IA = AmpA_guess[data["Aidx"]]
    IB = AmpB_guess[data["Aidx"]]

    return {"IA":IA, "IB":IB, "G":G, "Gprm": Gain_guess, "IAprm": AmpA_guess,
            "IBprm": AmpB_guess}

