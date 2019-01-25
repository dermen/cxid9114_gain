"""
Here are some functions for comparing
how well two crystals represent the
two color data
"""

from cxid9114.spots import spot_utils
import numpy as np


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



