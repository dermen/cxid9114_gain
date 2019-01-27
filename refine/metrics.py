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
