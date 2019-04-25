"""
Here are some functions for comparing
how well two crystals represent the
two color data
"""

from cxid9114.spots import spot_utils
from cxid9114 import utils
import numpy as np
from scipy.spatial import cKDTree

def hkl_residue_twocolor(refls, cryst, det, beamA, beamB ):
    """
    Given two beams find the HKL residues for each
    reflection for each beam
    :param refls: reflection table
    :param cryst: dxtbx crystal odel
    :param det: detector model
    :param beamA: beam model colorA
    :param beamB: beam model colorB
    :return: the resA and resB , or the HKL residues for each reflection for each color
        tuple of two numpy arrays, (Nrefl x 3) , (Nrefl x3)
        with beamA residues coming first
    """

    Hi_A, H_A = spot_utils.multipanel_refls_to_hkl(
        refls, det,
        beamA, cryst)
    Hi_B, H_B = spot_utils.multipanel_refls_to_hkl(
        refls, det,
        beamB, cryst)

    resA = np.abs(Hi_A - H_A)
    resB = np.abs(Hi_B - H_B)
    return resA, resB

def likeliest_color_and_res(refls, cryst, det, beamA, beamB, hkl_tol=0.1):
    """
    given beams of two colors, find the residual HKL
    for each reflection for each of the two beams (colors),
    and then check , for each  reflection, which beam, if any
    provides a smaller residual HKL.
    :param refls:
    :param cryst:
    :param det:
    :param beamA:
    :param beamB:
    :return: for each hkl, the likeliest residual HKL and the likeliest color
        indicated by 'A','B' corresponding to beamA,beamB in the arguments list
    """

    resHKLA, resHKLB = hkl_residue_twocolor(refls, cryst, det, beamA, beamB)

    likeliest_resid = []
    likeliest_color = []
    for rA,rB in zip(resHKLA, resHKLB):
        res = []
        colors =[]
        if np.all( rA < hkl_tol):
            res.append( rA )
            colors.append('A')
        if np.all( rB < hkl_tol):
            res.append( rB)
            colors.append('B')
        if len(res) == 2 :
            idx_res = np.argmin([ res[0].sum(), res[1].sum()])
            likeliest_resid.append(res[idx_res])
            likeliest_color.append(colors[idx_res])
        elif len(res) == 1:
            likeliest_resid.append(res[0])
            likeliest_color.append(colors[0])
        else:
            likeliest_resid.append(None)
            likeliest_color.append(None)

    return likeliest_resid, likeliest_color


def q_vecs_from_spotData(spotData):
    """

    :param spotData:
    :return:
    """
    all_q_vecs = []
    for pid in spotData:
        if spotData[pid] is None:
            continue
        q_vecs = spotData[pid]['q_vecs']
        all_q_vecs.append( q_vecs)
    return np.vstack(all_q_vecs)

def sim_to_data_Qdeviation(spotDataA, spotDataB, refls, detector, beam):
    """
    :param spotDataA:
    :param spotDataB:
    :param refls:
    :param detector:
    :param beam:
    :return:
    """
    reflsPP = spot_utils.refls_by_panelname(refls)
    data_qvecs = []
    for pid in reflsPP:
        R = reflsPP[pid]
        x,y,z = spot_utils.xyz_from_refl(R)
        qvecs = spot_utils.xy_to_q(x, y, detector[pid], beam)
        data_qvecs.append(qvecs)
    data_qvecs = np.vstack(data_qvecs)

    simA_qvecs = q_vecs_from_spotData(spotDataA)
    simB_qvecs = q_vecs_from_spotData(spotDataB)

    treeA = cKDTree(simA_qvecs)
    treeB = cKDTree(simB_qvecs)

    distA, idxA = treeA.query(data_qvecs , k=1)
    distB, idxB = treeB.query(data_qvecs , k=1)

    nnA = treeA.data[idxA]
    nnB = treeB.data[idxB]

    return distA, nnA, distB, nnB



def indexing_residuals_twocolor(spotA, spotB, refls, detector):
    """

    :param spotA:
    :param spotB:
    :param refls:
    :param beamA:
    :param beamB:
    :param detector:
    :return:
    """
    distA, distA_vecs = spot_utils.msisrp_overlap(spotA, refls, detector)
    distB, distB_vecs = spot_utils.msisrp_overlap(spotB, refls, detector)

    distsAB = zip(distA, distB)
    closer_vec = np.argmin(distsAB, axis=1)
    closer_dist = np.min(distsAB, axis=1)

    Nref = len(refls)
    dv = np.zeros( (Nref, 3))
    for i_ref in range(Nref):
        if closer_vec[i_ref] == 0:
            dv[i_ref] = (distA_vecs[i_ref])
        else:
            dv[i_ref] = ( distB_vecs[i_ref])
    return closer_dist, dv, closer_vec

#for f in fnames:
#    data = utils.open_flex(f)
#    spotA= data["spot_dataA"]
#    spotB = data["spot_dataB"]
#    refls = data["refls_strong"]
#    beamA = data["beamA"]
#    beamB = data["beamB"]
#    detector = data["detector"]
#    distA, distA_vecs = metrics.msisrp_overlap(spotA, refls, detector)
#    distB, distB_vecs = metrics.msisrp_overlap(spotB, refls, detector)
#    all_distA.append(distA)
#    all_distB.append(distB)
#    all_distA_vecs.append(distA_vecs)
#    all_distB_vecs.append( distB_vecs)
#all_distA = np.hstack(all_distA)
#all_distB = np.hstack(all_distB)
#all_distA_vecs = np.vstack( all_distA_vecs)
#all_distB_vecs = np.vstack( all_distB_vecs)

def plot_img(refls, spot_dataA, spot_dataB, detector, beamA,beamB, crystal, iset, name, bad=None):
    from cxid9114  import utils
    import numpy as np
    import pylab as plt
    d, dvecs, best = indexing_residuals_twocolor(spot_dataA, spot_dataB, refls, detector)
    HA, HiA = spot_utils.refls_to_hkl(refls, detector, beamA, crystal)
    HB, HiB = spot_utils.refls_to_hkl(refls, detector, beamB, crystal)

    Q = spot_utils.refls_to_q(refls, detector, beamA)
    Qmag = np.linalg.norm(Q, axis=1)
    res = 1. / Qmag

    HAres = np.round((HA - HiA),2)
    HBres = np.round((HB-HiB),2)
    plot_dvecs(d, dvecs, HAres, HBres)

    if bad is None:
        bad = np.where(utils.is_outlier(d, 4))[0]
    for i_b,b in enumerate(bad):
        r = refls[b]
        reso = res[b]
        panel = r['panel']
        HAres = np.round((HA - HiA)[b], 2)
        HBres = np.round((HB - HiB)[b], 2)
        # continue
        yA, xA = zip(*spot_dataA[panel]['comIpos'])
        yB, xB = zip(*spot_dataB[panel]['comIpos'])
        xp, yp, _ = r['xyzobs.px.value']
        img = iset.get_raw_data(0)[panel].as_numpy_array()
        plt.figure()
        plt.imshow(img, vmax=150)
        plt.plot(xA, yA, 'o', mfc='none', color='lightblue', ms=10, mew=2)
        plt.plot(xB, yB, 's', mfc='none', color='C3', ms=10, mew=2)

        plt.plot(xp, yp, 'o', color='C2', mfc='none', ms=10, mew=2)
        HAres = np.sqrt(np.sum(HAres**2))
        HBres = np.sqrt(np.sum(HBres**2))

        title_s = "fhklA: %.2f,    fhklB: %.2f" % (HAres, HBres)
        title_s += "\nresolution of spot: ~ %.2f Angstrom" % reso
        plt.gca().set_title(title_s)
        plt.draw()
        plt.pause(5)
        plt.savefig("%s_%d.png" % (name, i_b))
        plt.close()


def plot_dvecs(d, dvecs ,HA, HB):
    import pylab as plt
    import numpy as np
    from cxid9114 import utils
    bad = np.where(utils.is_outlier(d, 4))[0]
    plt.figure()
    plt.plot( dvecs[:,0]/.10992, dvecs[:,1]/.10992, 'o',color='C0', ms=3)
    ax = plt.gca()
    ax.add_patch(plt.Circle(xy=(0,0), radius=2, ec='C1', fc='none',ls='dashed'))
    for b in bad:
        hA = np.sqrt(np.sum(HA[b]**2))
        hB = np.sqrt(np.sum(HB[b]**2))
        h= min([hA,hB])
        s = "frac_h: %.2f" % h
        i,j,_ = dvecs[b]/.10992
        t=ax.text(i,j+2, s=s)
        t.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='w'))
        plt.plot( i,j, 'D', mec='C2', mfc='none',mew=2, ms=10 )
        #ax.add_patch(plt.Circle(xy=(i,j), radius=.5, ec='C2', fc='none',lw=2))

    plt.xlabel("$\Delta_X$ (pixels)", fontsize=18)
    plt.ylabel("$\Delta_Y$ (pixels)", fontsize=18)
    ax.tick_params(labelsize=15)



def gauss2d(x, y, A, xo, yo, sigma_x, sigma_y):
    Gxy = A*np.exp( - .5 / sigma_x**2 *(x-xo)**2  - .5 / sigma_y**2 * (y-yo)**2 )
    return Gxy

def gauss2d_resid(params, x,y, data):
    # A,x0,y0,sigma_x,sigma_y = params
    Gxy = gauss2d(x,y,*params)
    return np.sum( (data - Gxy)**2)


def check_indexable(refls_data, refls_sim, detector, beam, crystal, hkl_tol=.15 ):
    """
    checks whether the reflections in the data are indexed by the
    crystal model, and further whether  the corresponding miller index
    is present in the simulated reflections

    :param refls_data: reflection table from data
    :param refls_sim:  reflection table from simulation
    :param detector: dxtbx detector
    :param beam: dxtbx beam
    :param crystal: dxtbx crystal
    :param hkl_tol: fractional hkl hypotenuse to determine if a spot was indexed
    :return: various distance metrics and an array specifying indexability
    """

    from scipy.spatial import distance

    XYZ_dat = spot_utils.refls_xyz_in_lab(refls_data, detector)
    IJmm_dat = spot_utils.refls_to_pixelmm(refls_data, detector)
    H_dat, Hi_dat, Q_dat = spot_utils.refls_to_hkl(
        refls_data, detector, beam,
        crystal=crystal, returnQ=True )
    Hres_dat = np.sqrt(np.sum((H_dat - Hi_dat)**2, 1))  # residual hkl hypotenuse

    XYZ_sim = spot_utils.refls_xyz_in_lab(refls_sim, detector)
    IJmm_sim = spot_utils.refls_to_pixelmm(refls_sim, detector)
    H_sim, Hi_sim, Q_sim = spot_utils.refls_to_hkl(
        refls_sim, detector, beam,
        crystal=crystal, returnQ=True )

    # make tree for quick lookups
    HKLsim_tree = cKDTree(Hi_sim)

    all_d, all_dij, all_dQ, all_dvec, all_dijvec,all_dQvec, all_res, all_pid = \
        [],[],[],[],[],[],[],[]

    indexed = np.zeros( len(refls_data), bool)
    for i_r,r in enumerate(refls_data):
        indexable = True
        
        mil_idx = Hi_dat[i_r]
        if not Hres_dat[i_r] < hkl_tol:
            indexable = False

        # check the data miller is in the simulated millers
        miller_dist, i_r_sim = HKLsim_tree.query(mil_idx)
        if miller_dist > 0:  # the miller index of the data spot was not simulated
            indexable = False

        if not indexable:
            all_d.append(np.nan)
            all_dij.append(np.nan)
            all_dQ.append (np.nan)
            all_dvec.append( np.nan)
            all_dijvec.append( np.nan)
            all_dQvec.append( np.nan)
            all_res.append( np.nan)
            all_pid.append( np.nan)
            indexed[i_r] = False
            continue 

        dxyz = distance.euclidean(XYZ_dat[i_r], XYZ_sim[i_r_sim])
        dxyz_vec = XYZ_dat[i_r] - XYZ_sim[i_r_sim]

        dij = distance.euclidean(IJmm_dat[i_r], IJmm_sim[i_r_sim])
        dij_vec = IJmm_dat[i_r] - IJmm_sim[i_r_sim]


        dQ = distance.euclidean(Q_dat[i_r] , Q_sim[i_r_sim])
        dQ_vec = Q_dat[i_r] - Q_sim[i_r_sim]
        res = 1./ np.linalg.norm(Q_dat[i_r])

        all_d.append(dxyz)
        all_dij.append(dij)
        all_dQ.append (dQ)
        all_dvec.append( dxyz_vec)
        all_dijvec.append( dij_vec)
        all_dQvec.append( dQ_vec)
        all_res.append( res)
        all_pid.append( r['panel'])
        indexed[i_r] = True

    return {'d':all_d, 'dij': all_dij, 'dQ':all_dQ,
            'dvec':all_dvec, 'dvecij': all_dijvec, 'dvecQ':all_dQvec,
            'res':all_res, 'pid':all_pid, 'indexed': indexed}


def plot(X,Y,Z, cmap, xlim=None, ylim=None):
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    linewidth=1, antialiased=False,
                    cmap=cmap)

    # Adjust the limits, ticks and view angle
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(Z.min(), Z.max())
    ax.view_init(27, -21)
    return ax


def check_indexable2(refls_data, refls_sim, detector, beam, crystal, hkl_tol=.15 ):
    """
    checks whether the reflections in the data are indexed by the
    crystal model, and further whether  the corresponding miller index
    is present in the simulated reflections

    :param refls_data: reflection table from data
    :param refls_sim:  reflection table from simulation
    :param detector: dxtbx detector
    :param beam: dxtbx beam
    :param crystal: dxtbx crystal
    :param hkl_tol: fractional hkl hypotenuse to determine if a spot was indexed
    :return: various distance metrics and an array specifying indexability
    """
    from scipy.spatial import distance

    XYZ_dat = spot_utils.refls_xyz_in_lab(refls_data, detector)
    IJmm_dat = spot_utils.refls_to_pixelmm(refls_data, detector)
    H_dat, Hi_dat, Q_dat = spot_utils.refls_to_hkl(
        refls_data, detector, beam,
        crystal=crystal, returnQ=True )
    Hres_dat = np.sqrt(np.sum((H_dat - Hi_dat)**2, 1))  # residual hkl hypotenuse
    XYZ_sim = spot_utils.refls_xyz_in_lab(refls_sim, detector)
    IJmm_sim = spot_utils.refls_to_pixelmm(refls_sim, detector)
    H_sim, Hi_sim, Q_sim = spot_utils.refls_to_hkl(
        refls_sim, detector, beam,
        crystal=crystal, returnQ=True )

    # make tree for quick lookups
    HKLsim_tree = cKDTree(Hi_sim)

    all_d, all_dij, all_dQ, all_dvec, all_dijvec,all_dQvec, all_res, all_pid = \
        [],[],[],[],[],[],[],[]

    all_intens_sim = []
    all_sim_idx = []
    sim_pid = []
    indexed = np.zeros( len(refls_data), bool)
    
    for i_r, r in enumerate(refls_data):
        indexable = True
        
        mil_idx = Hi_dat[i_r]
        if not Hres_dat[i_r] < hkl_tol:
            indexable = False

        # check the data miller is in the simulated millers
        miller_dist, i_r_sim = HKLsim_tree.query(mil_idx)
        if miller_dist > 0:  # the miller index of the data spot was not simulated
            indexable = False

        res = 1. / np.linalg.norm(Q_dat[i_r])
        pid = r['panel']
        sim_intens = refls_sim['intensity.sum.value'][i_r_sim]
        if not indexable:
            all_d.append(np.nan)
            all_dij.append(np.nan)
            all_dQ.append (np.nan)
            all_dvec.append( np.nan)
            all_dijvec.append( np.nan)
            all_dQvec.append( np.nan)
            all_res.append(res)
            all_pid.append(pid)
            indexed[i_r] = False
            all_intens_sim.append(np.nan)  # NOTE: check me
            all_sim_idx.append( i_r_sim )
            sim_pid.append( np.nan)
            continue 

        dxyz = distance.euclidean(XYZ_dat[i_r], XYZ_sim[i_r_sim])
        dxyz_vec = XYZ_dat[i_r] - XYZ_sim[i_r_sim]

        dij = distance.euclidean(IJmm_dat[i_r], IJmm_sim[i_r_sim])
        dij_vec = IJmm_dat[i_r] - IJmm_sim[i_r_sim]

        dQ = distance.euclidean(Q_dat[i_r] , Q_sim[i_r_sim])
        dQ_vec = Q_dat[i_r] - Q_sim[i_r_sim]
        #res = 1./ np.linalg.norm(Q_dat[i_r])

        all_d.append(dxyz)
        all_dij.append(dij)
        all_dQ.append (dQ)
        all_dvec.append( dxyz_vec)
        all_dijvec.append( dij_vec)
        all_dQvec.append( dQ_vec)
        all_res.append( res)
        all_pid.append( pid)
        sim_pid.append( refls_sim['panel'][i_r_sim] )  # better be the same as the data!
        indexed[i_r] = True

        all_intens_sim.append( sim_intens)
        all_sim_idx.append( i_r_sim)

    return {'d':all_d, 'dij': all_dij, 'dQ':all_dQ,
            'dvec':all_dvec, 'dvecij': all_dijvec, 'dvecQ':all_dQvec,
            'res':all_res, 'pid':all_pid, 'indexed': indexed, 
            'hkl': Hi_dat, 'hkl_res' :  Hres_dat, 
            "sim_intens": all_intens_sim, "sim_refl_idx": all_sim_idx, 'sim_pid': sim_pid}
    
def twocolor_indexable(refls_data, refls_simA, refls_simB, detector, beamA, beamB, crystal, hkl_tol=.30 ):
    """
    checks whether the reflections in the data are indexed by the
    crystal model, and further whether  the corresponding miller index
    is present in the simulated reflections

    :param refls_data: reflection table from data
    :param refls_sim:  reflection table from simulation
    :param detector: dxtbx detector
    :param beam: dxtbx beam
    :param crystal: dxtbx crystal
    :param hkl_tol: fractional hkl hypotenuse to determine if a spot was indexed
    :return: various distance metrics and an array specifying indexability
    """

    from scipy.spatial import distance

    XYZ_dat = spot_utils.refls_xyz_in_lab(refls_data, detector)
    IJmm_dat = spot_utils.refls_to_pixelmm(refls_data, detector)
    H_dat, Hi_dat, Q_dat = spot_utils.refls_to_hkl(
        refls_data, detector, beam,
        crystal=crystal, returnQ=True )
    Hres_dat = np.sqrt(np.sum((H_dat - Hi_dat)**2, 1))  # residual hkl hypotenuse
    # metrics for simulation A
    XYZ_simA = spot_utils.refls_xyz_in_lab(refls_simA, detector)
    IJmm_simA = spot_utils.refls_to_pixelmm(refls_simA, detector)
    H_simA, Hi_simA, Q_simA = spot_utils.refls_to_hkl(
        refls_simA, detector, beamA,
        crystal=crystal, returnQ=True )
    
    # metrics for simulation B
    XYZ_simB = spot_utils.refls_xyz_in_lab(refls_simB, detector)
    IJmm_simB = spot_utils.refls_to_pixelmm(refls_simB, detector)
    H_simB, Hi_simB, Q_simB = spot_utils.refls_to_hkl(
        refls_simB, detector, beamB,
        crystal=crystal, returnQ=True )


    # now we should group simA and simB on HKL index
    treeA = cKDTree(Hi_simA)
    treeB = cKDTree( Hi_simB)

    treeA.query_ball_tree(treeB, r=0.1)

    from IPython import embed
    embed()

    Qsim_tree = cKDTree(Q_sim)
    
    all_d, all_dij, all_dQ, all_dvec, all_dijvec,all_dQvec, all_res, all_pid = \
        [],[],[],[],[],[],[],[]

    all_intens_sim = []
    all_sim_idx = []
    all_sim_pid = []
    indexed = np.zeros( len(refls_data), bool)
    
    #from IPython import embed
    #embed()

    for i_r,r in enumerate(refls_data):
        indexable = True
        
        mil_idx = Hi_dat[i_r]
        #if not Hres_dat[i_r] < hkl_tol:
        #    indexable = False

        # check the data miller is in the simulated millers
        _, i_r_sim = Qsim_tree.query(Q_dat[i_r])
        
        sim_mil_idx = Hi_sim[i_r_sim]

        if not all([i1==i2 for i1,i2 in zip( mil_idx, sim_mil_idx)]):
            indexable = False
        
        res = 1./ np.linalg.norm(Q_dat[i_r])
        pid = r['panel']
        sim_intens = refls_sim['intensity.sum.value'][i_r_sim]
        sim_pid =  refls_sim['panel'][i_r_sim] 

        dxyz = distance.euclidean(XYZ_dat[i_r], XYZ_sim[i_r_sim])
        dxyz_vec = XYZ_dat[i_r] - XYZ_sim[i_r_sim]

        if sim_pid == pid:
            dij = distance.euclidean(IJmm_dat[i_r], IJmm_sim[i_r_sim])
            dij_vec = IJmm_dat[i_r] - IJmm_sim[i_r_sim]
        else:
            dij = np.nan
            dij_vec = np.array([np.nan, np.nan])

        dQ = distance.euclidean(Q_dat[i_r] , Q_sim[i_r_sim])
        dQ_vec = Q_dat[i_r] - Q_sim[i_r_sim]

        all_d.append(dxyz)
        all_dij.append(dij)
        all_dQ.append (dQ)
        all_dvec.append( dxyz_vec)
        all_dijvec.append( dij_vec)
        all_dQvec.append( dQ_vec)
        all_res.append( res)
        all_pid.append( pid)
        all_sim_pid.append(sim_pid) 
        indexed[i_r] = indexable
        all_intens_sim.append( sim_intens)
        all_sim_idx.append( i_r_sim)

    return {'d':all_d, 'dij': all_dij, 'dQ':all_dQ,
            'dvec':all_dvec, 'dvecij': all_dijvec, 'dvecQ':all_dQvec,
            'res':all_res, 'pid':all_pid, 'indexed': indexed, 
            'hkl': Hi_dat, 'hkl_res' :  Hres_dat, 
            "sim_intens": all_intens_sim, "sim_refl_idx": all_sim_idx, 'sim_pid': sim_pid}

