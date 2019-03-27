
import sys
from cxid9114 import utils
from cxid9114.geom import geom_utils
from dxtbx.model.experiment_list import ExperimentListFactory
from cxid9114.spots import spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from copy import deepcopy
import numpy as np
import scipy.ndimage
from cxid9114.refine import metrics
from cctbx import miller, sgtbx
from cxid9114 import utils
import pandas
from scipy.spatial import cKDTree
from dials.array_family import flex


# which spacey groupy
sg96 = sgtbx.space_group(" P 4nw 2abw")

exp_name = sys.argv[1]
data_name = sys.argv[2]
ofile = sys.argv[3]
try:
    best_fname = sys.argv[4]
    has_best = True
except IndexError:
    has_best = False
hkl_tol = .15

run = int(exp_name.split("/")[1].split("run")[1])
shot_idx = int(exp_name.split("_")[1])

ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
FF = [10000, None]  # Setting structure factors takes long time in nanoBragg, so
# unless you want energy-dependent structure factors
# you need only provide one number -or- one structure factor flex miller table
# and the computer will know to preserve that for all beam colors

FLUX = [1e12, 1e12]  # fluxes of the beams

flux_frac = np.random.uniform(.2, .8)
chanA_flux = flux_frac * 1e12
chanB_flux = (1. - flux_frac) * 1e12
FLUXdat = [chanA_flux, chanB_flux]
GAIN = np.random.uniform(0.5, 3)

waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
exp_lst = ExperimentListFactory.from_json_file(exp_name)  # , check_format=False)
iset = exp_lst.imagesets()[0]
detector = iset.get_detector(0)
data = utils.open_flex(data_name)
beamA = deepcopy(iset.get_beam())
beamB = deepcopy(iset.get_beam())
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

crystalAB = data["crystalAB"]


if has_best:
    szx = szy = 5
    df = pandas.read_pickle( best_fname)
    idxmax = df.ave_score.idxmax()
    best_Amat = tuple(df.Amat.iloc[ idxmax])
    best_Ncell_abc = tuple(df[ ["Na", "Nb", "Nc"]].iloc[idxmax].values)
    best_mos_spread = df["mos_spread"].iloc[idxmax]
    best_shape = df["xtals_shape"].iloc[idxmax]
    best_Nmos_dom = df["Nmos_domain"].iloc[idxmax]
    best_shape = "gauss"
    crystalAB.set_A(best_Amat)

    simsAB = sim_utils.sim_twocolors2(
        crystalAB, detector, iset.get_beam(0), FF,
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        FLUX, pids=None, profile=best_shape, oversample=0, 
        Ncells_abc=best_Ncell_abc, 
        mos_dom=best_Nmos_dom, 
        mos_spread=best_most_spread)
    good_refl_data = utils.open_flex(best_fname.replace(".pkl", "_best_dump.pkl") )
    refl_data = good_refl_data["good_refls"]
else:
    simsAB = sim_utils.sim_twocolors2(
        crystalAB, detector, iset.get_beam(0), FF,
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        FLUX, pids=None, Gauss=False, oversample=0, 
        Ncells_abc=(22, 22, 22), mos_dom=1, mos_spread=0.0)

    refl_data = data["refls_strong"]
    
print "\n\n\n#######\nUsing %d reflections\n#####\n\n" % len( refl_data)

refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA)
refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB)

residA = metrics.check_indexable2(
    refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
residB = metrics.check_indexable2(
    refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)


# NOTE load some dummie structure factor data
# these are calculated using the PDB 4bs7.pdb, Lyso Yterbium derivitive
# solved at a synchrotron, room temp
FA = utils.open_flex('SA.pkl')  # This goes from being ground truth in the simulation to a best-guess in the data fitting
FB = utils.open_flex('SB.pkl')  
HA = tuple([hkl for hkl in FA.indices()]) 
HB = tuple([hkl for hkl in FB.indices()])
HA_val_map = {hkl: FA.value_at_index(hkl) for hkl in HA}
HB_val_map = {hkl: FB.value_at_index(hkl) for hkl in HB}

d = {"crystalAB": crystalAB,
     "residA": residA,
     "residB": residB,
     "beamA": beamA,
     "beamB": beamB,
     "detector": detector,
     "refls_simA": refl_simA,
     "refls_simB": refl_simB,
     "flux_data": FLUXdat,
     "gain": GAIN,
     "refls_data": refl_data}


def get_val_at_hkl(hkl, val_map):
    poss_equivs = [i.h() for i in
                   miller.sym_equiv_indices(sg96, hkl).indices()]
    for hkl2 in poss_equivs:
        if hkl2 in val_map:  # fast lookup
            break
    return hkl2, val_map[hkl2]

# before processing we need to check edge cases
filt=True
if filt:
    _, all_HiA, _ = spot_utils.refls_to_hkl(
        refl_simA, detector, beamA,
        crystal=crystalAB, returnQ=True)
    all_treeA = cKDTree(all_HiA)
    nnA = all_treeA.query_ball_point(all_HiA, r=1e-7)

    _, all_HiB, _ = spot_utils.refls_to_hkl(
        refl_simB, detector, beamB,
        crystal=crystalAB, returnQ=True)
    all_treeB = cKDTree(all_HiB)
    nnB = all_treeB.query_ball_point(all_HiB, r=1e-7)

    NreflA = len(refl_simA)
    NreflB = len(refl_simB)

    drop_meA = []
    for i, vals in enumerate(nnA):
        if i in drop_meA:
            continue
        if len(vals) > 1:
            pids = [refl_simA[v]['panel'] for v in vals]
            if len(set(pids)) == 1:
                refl_vals = refl_simA.select(flex.bool([i_v in vals
                                        for i_v in np.arange(NreflA)]))
                x, y, z = spot_utils.xyz_from_refl(refl_vals)
                allI = [r['intensity.sum.value'] for r in refl_vals]
                allI = sum(allI)
                xm = np.mean(x)
                ym = np.mean(y)
                zm = np.mean(z)
                drop_meA.extend(vals[1:])
                x1b, x2b, y1b, y2b, z1b, z2b = zip(*[r['bbox'] for r in refl_vals])
                keep_me = vals[0]
                # indexing order is important to modify as reference
                refl_simA['intensity.sum.value'][keep_me] = allI
                refl_simA['xyzobs.px.value'][keep_me] = (xm, ym, zm)
                refl_simA['bbox'][keep_me] = (min(x1b), max(x2b),\
                                min(y1b), max(y2b), min(z1b), max(z2b))
            else:
                drop_meA.append(vals)
            print vals

    if drop_meA:
        keep_meA = np.array([i not in drop_meA for i in range(NreflA)])
        refl_simA = refl_simA.select(flex.bool(keep_meA))
        NreflA = len( refl_simA)

    drop_meB = []
    for i, vals in enumerate(nnB):
        if i in drop_meB:
            continue
        if len(vals) > 1:
            pids = [refl_simB[v]['panel'] for v in vals]
            if len(set(pids)) == 1:
                print vals
                # merge_spots(vals)
                refl_vals = refl_simB.select(flex.bool([i_v in vals 
                                                for i_v in np.arange(NreflB)]))
                x, y, z = spot_utils.xyz_from_refl(refl_vals)
                allI = [r['intensity.sum.value'] for r in refl_vals]
                allI = sum(allI)
                xm = np.mean(x)
                ym = np.mean(y)
                zm = np.mean(z)
                drop_meB.extend( vals[1:])
                x1b, x2b, y1b, y2b, z1b, z2b = zip(*[r['bbox'] for r in refl_vals])
                keep_me = vals[0]
                refl_simB['intensity.sum.value'][keep_me] = allI
                refl_simB['xyzobs.px.value'][keep_me] = (xm, ym, zm)
                refl_simB['bbox'][keep_me] = (min(x1b), max(x2b), min(y1b),\
                                max(y2b), min(z1b), max(z2b))
            else:
                drop_meB.append(vals)
            print vals
    if drop_meB:
        keep_meB = [i not in drop_meB for i in range(NreflB)]
        refl_simB = refl_simB.select(flex.bool(keep_meB))
        NreflB = len( refl_simB)

## remake the trees given the drops
    _, all_HiA = spot_utils.refls_to_hkl(
        refl_simA, detector, beamA,
        crystal=crystalAB, returnQ=False)
    all_treeA = cKDTree(all_HiA)

    _, all_HiB = spot_utils.refls_to_hkl(
        refl_simB, detector, beamB,
        crystal=crystalAB, returnQ=False)
    #all_treeB = cKDTree(all_HiB)

## CHECK if same HKL, indexed by both colors
# exists on multiple panels, and if so, delete...
    nnAB = all_treeA.query_ball_point(all_HiB, r=1e-7)  
    drop_meA = []
    drop_meB = []
    for iB, iA_vals in enumerate(nnAB):
        if len(iA_vals) > 0:
            assert (len(iA_vals) == 1)
            iA = iA_vals[0]
            pidA = refl_simA[iA]['panel']
            pidB = refl_simB[iB]['panel']
            if pidA != pidB:
                drop_meA.append(iA)
                drop_meB.append(iB)

    if drop_meA:
        keep_meA = [i not in drop_meA for i in range(NreflA)]
        refl_simA = refl_simA.select(flex.bool(keep_meA))
    if drop_meB:
        keep_meB = [i not in drop_meB for i in range(NreflB)]
        refl_simB = refl_simB.select(flex.bool(keep_meB))

# ----  Done with edge case filters#



# reflections per panel
rpp = spot_utils.refls_by_panelname(refl_data)
rppA = spot_utils.refls_by_panelname(refl_simA)
rppB = spot_utils.refls_by_panelname(refl_simB)

DATA = {"D": [], "IA": [], "IB": [], "h2": [], "k2": [], "l2": [],
        "h": [], "k": [], "l": [], "PA": [], "PB": [], "FA": [],
        "FB": [], "iA": [], "iB": [], "Nstrong": [], "pid": [],
        "delta_pix": []  } # NOTE: added in the delta pix 
                           # for comparing sim and data center of masses

all_int_me = []
sz_fudge = 5  # integration fudge factor to include spots that dont overlap perfectly with predictions

# now set up boundboxes and integrate
for pid in rpp:
    R = rpp[pid]
    if pid in rppA:  # are there A-channel reflections on this panel
        inA = True
        RA = rppA[pid]
        xA, yA, _ = spot_utils.xyz_from_refl(RA)
        pointsA = np.array(zip(xA, yA))
        HA, HiA, QA = spot_utils.refls_to_hkl(
            RA, detector, beamA,
            crystal=crystalAB, returnQ=True)
    else:
        inA = False

    if pid in rppB:  # are there B channel reflections on this channel
        inB = True
        RB = rppB[pid]
        xB, yB, _ = spot_utils.xyz_from_refl(RB)
        pointsB = np.array(zip(xB, yB))
        HB, HiB, QB = spot_utils.refls_to_hkl(
            RB, detector, beamB,
            crystal=crystalAB, returnQ=True)
    else:
        inB = False

    x, y, _ = spot_utils.xyz_from_refl(R)
    x = np.array(x)
    y = np.array(y)

    panX, panY = detector[pid].get_image_size()

    mergesA = []
    mergesB = []
    if inA and inB:  # are there both A and B channel reflections ? If so, lets find out which ones have same hkl
        # make tree structure for merging the spots
        treeA = cKDTree(pointsA)
        treeB = cKDTree(pointsB)
        # how far apart should the two color spots be ? 
        rmax = geom_utils.twocolor_deltapix(detector[pid], beamA, beamB)
        merge_me = treeA.query_ball_tree(treeB, r=rmax + sz_fudge) # slap on some fudge
        # if pixels points in treeA are within rmax + sz_fugde of 
        # points in treeB, then these points are assumed to be overlapped
        for iA, iB in enumerate(merge_me):
            if not iB:
                continue
            iB = iB[0]

            # check that the miller indices are the same
            if not all([i == j for i, j in zip(HiA[iA], HiB[iB])]):
                continue
            x1A, x2A, y1A, y2A, _, _ = RA[iA]['bbox']  # shoebox'].bbox
            x1B, x2B, y1B, y2B, _, _ = RB[iB]['bbox']  # shoebox'].bbox

            xlow = max([0, min((x1A, x1B)) - sz])
            xhigh = min([panX, max((x2A, x2B)) + sz])
            ylow = max([0, min((y1A, y1B)) - sz])
            yhigh = min([panY, max((y2A, y2B)) + sz])

            # integrate me if I am in the bounding box!
            int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
            if not int_me.size:
                continue
            mergesA.append(iA)
            mergesB.append(iB)

            # integrate the spot, this will change depending on data or simulation
            #NOTE : adding in the data-spot center of mass here as well
            totalCOM = np.zeros(3)  # NOTE: x,y,z
            totalI = 0
            for ref_idx in int_me:
                # TODO implement the spot intensity version here
                # which fits the background plane!
                totalI += rpp[pid][ref_idx]["intensity.sum.value"]
                totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
            totalCOM /= len( int_me)

            PA = RA[iA]['intensity.sum.value']
            PB = RB[iB]['intensity.sum.value']

            # NOTE: added the simulated spot(s) center of mass
            posA = RA[iA]['xyzobs.px.value']
            posB = RB[iB]['xyzobs.px.value']
            simCOM = np.mean( [posA , posB], axis=0)

            # get the hkl structure factor, and the sym equiv hkl
            (h, k, l) = HiA[iA]  # NOTE: same for A and B channels
            (h2, k2, l2), FA = get_val_at_hkl((h, k, l), HA_val_map)
            _, FB = get_val_at_hkl((h, k, l), HB_val_map)  # NOTE: no need to return h2,k2,l2 twice

            DATA['h'].append(h)
            DATA['k'].append(k)
            DATA['l'].append(l)
            DATA['h2'].append(h2)
            DATA['k2'].append(k2)
            DATA['l2'].append(l2)
            DATA['D'].append(totalI)
            DATA['PA'].append(PA)
            DATA['PB'].append(PB)
            DATA['FA'].append(FA)
            DATA['FB'].append(FB)
            DATA['IA'].append(abs(FA) ** 2)
            DATA['IB'].append(abs(FB) ** 2)

            DATA['pid'].append(pid)
            DATA["Nstrong"].append(int_me.size)
            DATA["iA"].append(iA)
            DATA["iB"].append(iB)
            all_int_me.append(int_me)

            # NOTE: stash the sim-data distance (COM to COM)
            DATA["delta_pix"].append(distance.euclidean(totalCOM, simCOM)  )
            # this spot was both colors, overlapping
            # find center of mass of all spots inside the integration box
            # and find its distance to the center of mass of the simulation spots
    
    if inA:
        for iA, ref in enumerate(RA):
            if iA in mergesA:
                # this sim spot was already treated above
                continue
            x1A, x2A, y1A, y2A, _, _ = RA[iA]['bbox']  # ['shoebox'].bbox
            xlow = max((0, x1A - sz))
            xhigh = min((panX, x2A + sz))
            ylow = max((0, y1A - sz))
            yhigh = min((panY, y2A + sz))
            int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
            if not int_me.size:
                continue

            # NOTE: added in the total sim calc
            totalCOM = np.zeros(3)
            totalI = 0
            for ref_idx in int_me:
                # TODO implement the spot intensity version here
                # which fits the background plane!
                totalI += rpp[pid][ref_idx]["intensity.sum.value"]
                totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
            totalCOM /= len( int_me)

            PA = RA[iA]['intensity.sum.value']
            PB = 0  # crucial ;)
            
            # NOTE: added the simulated spot center of mass, for spotA
            simCOM = RA[iA]['xyzobs.px.value']

            # get the hkl structure factor, and the sym equiv hkl
            (h, k, l) = HiA[iA]  # NOTE: same for A and B channels
            (h2, k2, l2), FA = get_val_at_hkl((h, k, l), HA_val_map)
            _, FB = get_val_at_hkl((h, k, l), HB_val_map)  # NOTE: no need to return h2,k2,l2 twice

            DATA['h'].append(h)
            DATA['k'].append(k)
            DATA['l'].append(l)
            DATA['h2'].append(h2)
            DATA['k2'].append(k2)
            DATA['l2'].append(l2)
            DATA['D'].append(totalI)
            DATA['PA'].append(PA)
            DATA['PB'].append(PB)
            DATA['FA'].append(FA)
            DATA['FB'].append(FB)
            DATA['IA'].append(abs(FA) ** 2)
            DATA['IB'].append(abs(FB) ** 2)

            DATA['pid'].append(pid)
            DATA["Nstrong"].append(int_me.size)
            DATA["iA"].append(iA)
            DATA["iB"].append(np.nan)
            all_int_me.append(int_me)
            
            # NOTE: stash the sim-data distance (COM to COM)
            DATA["delta_pix"].append(distance.euclidean(totalCOM, simCOM)  )

    if inB:
        for iB, ref in enumerate(RB):
            if iB in mergesB:
                continue
            x1B, x2B, y1B, y2B, _, _ = RB[iB]['bbox']  # shoebox'].bbox
            xlow = max((0, x1B - sz))
            xhigh = min((panX, x2B + sz))
            ylow = max((0, y1B - sz))
            yhigh = min((panY, y2B + sz))
            # subimg = simsDataSum[pid][ylow:yhigh, xlow:xhigh]
            # bg = 0
            int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
            if not int_me.size:
                continue

            # NOTE: added im the total COM calc
            totalCOM = np.zeros(3)
            totalI = 0
            for ref_idx in int_me:
                # TODO implement the spot intensity version here
                # which fits the background plane!
                totalI += rpp[pid][ref_idx]["intensity.sum.value"]
                totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
            totalCOM /= len( int_me)

            PA = 0  # crucial ;)
            PB = RB[iB]['intensity.sum.value']
            
            # NOTE: added the simulated spot center of mass, for spotB only
            simCOM = RB[iB]['xyzobs.px.value']

            # get the hkl structure factor, and the sym equiv hkl
            (h, k, l) = HiB[iB]  # NOTE: same for A and B channels
            (h2, k2, l2), FB = get_val_at_hkl((h, k, l), HB_val_map)
            _, FA = get_val_at_hkl((h, k, l), HA_val_map)  # NOTE: no need to return h2,k2,l2 twice

            DATA['h'].append(h)
            DATA['k'].append(k)
            DATA['l'].append(l)
            DATA['h2'].append(h2)
            DATA['k2'].append(k2)
            DATA['l2'].append(l2)
            DATA['D'].append(totalI)
            DATA['PA'].append(PA)
            DATA['PB'].append(PB)
            DATA['FA'].append(FA)
            DATA['FB'].append(FB)
            DATA['IA'].append(abs(FA) ** 2)
            DATA['IB'].append(abs(FB) ** 2)

            DATA['pid'].append(pid)
            DATA["Nstrong"].append(int_me.size)
            DATA["iA"].append(np.nan)
            DATA["iB"].append(iB)
            all_int_me.append(int_me)
            
            # NOTE: stash the sim-data distance (COM to COM)
            DATA["delta_pix"].append(distance.euclidean(totalCOM, simCOM)  )

df = pandas.DataFrame(DATA)
df["run"] = run
df["shot_idx"] = shot_idx
df['gain'] = d['gain']
df['LA'] = d['flux_data'][0]
df["LB"] = d['flux_data'][1]
df['K'] = FF[0] ** 2 * FLUX[0]
df["rhs"] = df.gain * (df.IA * df.LA * (df.PA / df.K) + df.IB * df.LB * (df.PB / df.K))
df["lhs"] = df.D

df.to_pickle(ofile)

## all_int_me

