#!/usr/bin/env libtbx.python

import sys
import argparse
from copy import deepcopy
import glob
import os

import numpy as np
import pandas
from scipy.spatial import cKDTree, distance
from IPython import embed
import h5py

from cctbx import miller, sgtbx
from cxid9114 import utils
from dials.array_family import flex
import dxtbx

from cxid9114 import utils
from cxid9114.geom import geom_utils
from cxid9114.spots import integrate, spot_utils
from cxid9114 import parameters
from cxid9114.sim import sim_utils
from cxid9114.solvers import setup_inputs
from cxid9114.refine import metrics

from LS49.sim.step4_pad import microcrystal


spec_f = h5py.File("simMe_data_run62.h5")
spec_data = spec_f["hist_spec"][()]
sg96 = sgtbx.space_group(" P 4nw 2abw")
run = 62

parser = argparse.ArgumentParser("Integrate and consolidate data")
parser.add_argument('-j', dest='j', type=int, help="number of jobs", default=1)
parser.add_argument('-m', dest='max_files', type=int, help="max files to process", default=None)
parser.add_argument('-thresh', dest='thresh', type=float, help="spot thresh", default=1e-2)
parser.add_argument('--hkl-tol', dest='htol', type=float, help="hkl tolerance", default=0.33)
parser.add_argument('-jid', dest='jid', type=int, help='rank ID', default=0) 
parser.add_argument('-cpusim', dest='cpusim', action='store_true', help='nanoBragg on CPU')
parser.add_argument('--force-twocolor', dest='force2', action='store_true', help='sim only 2 colors')

parser.add_argument('--plot-overlap', dest='plot_overlap', action='store_true', help='plot indexing overlap between data and basic simulation')
parser.add_argument('-G', dest='G', type=float, help='ADU per photon (gain)', default=1)
parser.add_argument('-iglob', dest='iglob', type=str, help='input file glob', required=True)
parser.add_argument('-t', type=str, dest='t', default='rockets', help='output file tag')
parser.add_argument('-o', type=str, dest='o', default='bigsim_rocketships', 
                    help='output directory')
args = parser.parse_args()


Njobs = args.j
jid = args.jid

cuda = True 
if args.cpusim:
    cuda = False

nom_gain = args.G  # quantum gain used in the bigsim 

odir = args.o
if not os.path.exists( odir):
    os.makedirs( odir)
ofile = "%s_liftoff_betelgeuse%d.%d.pdpkl" % (args.t, jid+1, Njobs)
ofile = os.path.join( odir, ofile)
print (ofile)

file_list = glob.glob(args.iglob)
Nfiles = len(file_list)

# Load some model params    
thresh=args.thresh # threshold for spot finding in simulated 'noiseless' images
hkl_tol = args.htol  # this is only used for initial testing of idnexability
ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH] 
FF = [1e4, None]  
FLUX = [1e12, 1e12]  
beamsize_mm = 0.001
Deff_A = 2200
length_um = 2.2
detector = utils.open_flex("bigsim_detect.pkl")
beam = utils.open_flex("bigsim_beam.pkl")

beamA = deepcopy(beam) 
beamB =  deepcopy(beam) 
waveA = parameters.ENERGY_CONV / ENERGIES[0]
waveB = parameters.ENERGY_CONV / ENERGIES[1]
beamA.set_wavelength(waveA)
beamB.set_wavelength(waveB)

file_list_idx = np.array_split( np.arange(Nfiles), Njobs)
    
crystal = microcrystal(Deff_A=Deff_A, length_um=length_um, 
            beam_diameter_um=beamsize_mm*1000, 
            verbose=False) 

all_dfs = []

idxmax = file_list_idx[-1]
idxstart = file_list_idx[jid]
Nfiles = len(file_list_idx[jid])
if args.max_files is not None:
    Nfiles = min( args.max_files, Nfiles)

for idx in range( idxstart, idxstart+Nfiles): 

    data_name = file_list[idx]
    data = utils.open_flex(data_name)
    shot_idx = int(data["img_f"].split("_")[-1].split(".")[0])
    
    print "Data file %s" % data_name

    shot_idx = int(shot_idx)

    shot_spectrum = spec_data[shot_idx]

    chanA_flux = shot_spectrum[10:25].sum()
    chanB_flux = shot_spectrum[100:115].sum()

    crystalAB = data["crystalAB"]

    print "Doing the basic simulation.."
    simsAB = sim_utils.sim_twocolors2(
        crystalAB, 
        detector, 
        beam, 
        FF,
        [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
        FLUX, 
        Gauss=True, 
        oversample=0,
        Ncells_abc=(25, 25, 25), 
        mos_dom=1000, 
        mos_spread=0.015, 
        cuda=cuda,
        device_Id=0,
        beamsize_mm=beamsize_mm,
        boost=crystal.domains_per_crystal,
        exposure_s=1)
    
    print "Done!"

    refl_data = data["refls_strong"]

    #    
    print "\n\n\n#######\nProcessing %d reflections read from the data file \n#####\n\n" % len( refl_data)

    refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA, thresh=thresh)
    refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB, thresh=thresh)

    residA = metrics.check_indexable2(
        refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
    residB = metrics.check_indexable2(
        refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)


    print "INitial metrics suggest that:"
    print "\t %d reflections could be indexed by channeL A" % residA['indexed'].sum()
    print "\t %d reflections could be indexed by channeL B" % residB['indexed'].sum()
    print "\t NOw we can check for outliers.. "

    if args.plot_overlap:
        spot_utils.plot_overlap(refl_simA, refl_simB, refl_data, detector) 

    d = {"crystalAB": crystalAB,
         "residA": residA,
         "residB": residB,
         "beamA": beamA,
         "beamB": beamB,
         "detector": detector,
         "refls_simA": refl_simA,
         "refls_simB": refl_simB,
         "refls_data": refl_data}

    # integrate with tilt plane subtraction 
    print("LOADING THE FINE IMAGE")
    loader = dxtbx.load(data["img_f"])
    pan_data = np.array([ loader.get_raw_data().as_numpy_array()])
    print (data["img_f"]) 
    
    # make a dummie mask
    mask = np.ones_like( pan_data).astype(np.bool)

    # before processing we need to check edge cases
    print "Checking the simulations edge cases, basically to do with the spot detection of simulations... \n\t such a pain.. "

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

        ##  remake the trees given the drops
        _, all_HiA = spot_utils.refls_to_hkl(
            refl_simA, detector, beamA,
            crystal=crystalAB, returnQ=False)
        all_treeA = cKDTree(all_HiA)

        _, all_HiB = spot_utils.refls_to_hkl(
            refl_simB, detector, beamB,
            crystal=crystalAB, returnQ=False)
        #all_treeB = cKDTree(all_HiB)

        ##  CHECK if same HKL, indexed by both colors
        #   exists on multiple panels, and if so, delete...
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
    print "<><><><>\nI am doing checking the simulations for edge cases!\n<><><><>"

    # reflections per panel
    rpp = spot_utils.refls_by_panelname(refl_data)
    rppA = spot_utils.refls_by_panelname(refl_simA)
    rppB = spot_utils.refls_by_panelname(refl_simB)

    
    DATA = {"D": [], "Dnoise": [],
            "h":[],"k":[],"l":[], "is_pos": [],
            "hAnom": [], "kAnom": [], "lAnom": [],
            "horig": [], "korig": [], "lorig": [], "PA": [], "PB": [], 
            "iA": [], "iB": [], "Nstrong": [], "pid": [],
            "delta_pix": []  } # NOTE: added in the delta pix 
                               # for comparing sim and data center of masses

    all_int_me = []
    sz_fudge = sz = 5  # integration fudge factor to include spots that dont overlap perfectly with predictions
    # double define for convenience cause sz is easier to type than sz_fudge

    #  now set up boundboxes and integrate
    for idx_pid, pid in enumerate(rpp):
        # NOTE: integrate the spots for this panel
        Is, Ibk, noise, pix_per = integrate.integrate3(rpp[pid], mask[pid], pan_data[pid], gain=nom_gain)
        
        
        print  "Processing peaks on CSPAD panel %d (%d / %d)" % (pid, idx_pid, len( rpp))
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
            # NOTE: this is the critical step - are the spots within rmax - and if so they are considered indexed.. 
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
                totalNoise = 0
                for ref_idx in int_me:
                    # TODO implement the spot intensity version here
                    # which fits the background plane!
                    totalI +=  Is[ref_idx]  #rpp[pid][ref_idx]["intensity.sum.value"]
                    totalNoise += noise[ref_idx]**2
                    totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
                totalCOM /= len( int_me)
                totalNoise = np.sqrt(totalNoise)

                PA = RA[iA]['intensity.sum.value']
                PB = RB[iB]['intensity.sum.value']

                # NOTE: added the simulated spot(s) center of mass
                posA = RA[iA]['xyzobs.px.value']
                posB = RB[iB]['xyzobs.px.value']
                simCOM = np.mean( [posA , posB], axis=0)

                # get the hkl structure factor, and the sym equiv hkl
                (horig, korig, lorig) = HiA[iA]  # NOTE: same for A and B channels
                h,k,l = setup_inputs.single_to_asu((horig,korig,lorig), ano=False)
                hAnom,kAnom,lAnom = setup_inputs.single_to_asu((horig,korig,lorig), ano=True)
                if h==hAnom and k==kAnom and l==lAnom:
                    is_pos = True
                else:
                    is_pos = False
                DATA['is_pos'].append( is_pos)
                DATA['horig'].append(horig)
                DATA['korig'].append(korig)
                DATA['lorig'].append(lorig)
                DATA['h'].append(h)
                DATA['k'].append(k)
                DATA['l'].append(l)
                DATA['hAnom'].append(hAnom)
                DATA['kAnom'].append(kAnom)
                DATA['lAnom'].append(lAnom)

                DATA['D'].append(totalI)
                DATA['Dnoise'].append(totalNoise)
                DATA['PA'].append(PA)
                DATA['PB'].append(PB)

                DATA['pid'].append(pid)
                DATA["Nstrong"].append(int_me.size)
                DATA["iA"].append(iA)
                DATA["iB"].append(iB)
                all_int_me.append(int_me)

                # NOTE: stash the sim-data distance (COM to COM)
                DATA["delta_pix"].append(distance.euclidean(totalCOM[:2], simCOM[:2])  )
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
                totalNoise = 0
                for ref_idx in int_me:
                    # TODO implement the spot intensity version here
                    # which fits the background plane!
                    totalI += Is[ref_idx]  #rpp[pid][ref_idx]["intensity.sum.value"]
                    totalNoise += noise[ ref_idx]**2
                    totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
                totalCOM /= len( int_me)
                totalNoise = np.sqrt( totalNoise)
                PA = RA[iA]['intensity.sum.value']
                PB = 0  # crucial ;)
                
                # NOTE: added the simulated spot center of mass, for spotA
                simCOM = np.array(RA[iA]['xyzobs.px.value'])

                # get the hkl structure factor, and the sym equiv hkl
                (horig, korig, lorig) = HiA[iA]  
                h,k,l = setup_inputs.single_to_asu((horig,korig,lorig), ano=False)
                hAnom,kAnom,lAnom = setup_inputs.single_to_asu((horig,korig,lorig), ano=True)
                if h==hAnom and k==kAnom and l==lAnom:
                    is_pos = True
                else:
                    is_pos = False
                DATA['is_pos'] .append(is_pos)
                DATA['horig'].append(horig)
                DATA['korig'].append(korig)
                DATA['lorig'].append(lorig)
                DATA['h'].append(h)
                DATA['k'].append(k)
                DATA['l'].append(l)
                DATA['hAnom'].append(hAnom)
                DATA['kAnom'].append(kAnom)
                DATA['lAnom'].append(lAnom)
                
                
                DATA['D'].append(totalI)
                DATA['Dnoise'].append(totalNoise)
                DATA['PA'].append(PA)
                DATA['PB'].append(PB)

                DATA['pid'].append(pid)
                DATA["Nstrong"].append(int_me.size)
                DATA["iA"].append(iA)
                DATA["iB"].append(np.nan)
                all_int_me.append(int_me)
                
                # NOTE: stash the sim-data distance (COM to COM)
                DATA["delta_pix"].append(distance.euclidean(totalCOM[:2], simCOM[:2])  )

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

                # NOTE: added in the total COM calc
                totalCOM = np.zeros(3)
                totalI = 0
                totalNoise = 0
                for ref_idx in int_me:
                    # TODO implement the spot intensity version here
                    # which fits the background plane!
                    totalI += Is[ref_idx]  #rpp[pid][ref_idx]["intensity.sum.value"]
                    totalNoise += noise[ref_idx]**2
                    totalCOM += np.array(rpp[pid][ref_idx]["xyzobs.px.value"])
                totalCOM /= len( int_me)
                totalNoise = np.sqrt(totalNoise)

                PA = 0  # crucial ;)
                PB = RB[iB]['intensity.sum.value']
                
                # NOTE: added the simulated spot center of mass, for spotB only
                simCOM = np.array(RB[iB]['xyzobs.px.value'])

                # get the hkl structure factor, and the sym equiv hkl
                (horig, korig, lorig) = HiB[iB]  
                h,k,l = setup_inputs.single_to_asu((horig,korig,lorig), ano=False)
                hAnom,kAnom,lAnom = setup_inputs.single_to_asu((horig,korig,lorig), ano=True)
                if h==hAnom and k==kAnom and l==lAnom:
                    is_pos = True
                else:
                    is_pos = False

                DATA['is_pos'].append( is_pos)
                DATA['horig'].append(horig)
                DATA['korig'].append(korig)
                DATA['lorig'].append(lorig)
                DATA['h'].append(h)
                DATA['k'].append(k)
                DATA['l'].append(l)
                DATA['hAnom'].append(hAnom)
                DATA['kAnom'].append(kAnom)
                DATA['lAnom'].append(lAnom)
                
                DATA['D'].append(totalI)
                DATA['Dnoise'].append(totalNoise)
                DATA['PA'].append(PA)
                DATA['PB'].append(PB)

                DATA['pid'].append(pid)
                DATA["Nstrong"].append(int_me.size)
                DATA["iA"].append(np.nan)
                DATA["iB"].append(iB)
                all_int_me.append(int_me)
                
                # NOTE: stash the sim-data distance (COM to COM)
                DATA["delta_pix"].append(distance.euclidean(totalCOM[:2], simCOM[:2])  )

    df = pandas.DataFrame(DATA)
    df["run"] = run
    df["shot_idx"] = shot_idx
    df['LA'] = chanA_flux
    df["LB"] = chanB_flux
    df['K'] = FF[0] ** 2 * FLUX[0]
    df['nominal_gain'] = nom_gain
    all_dfs.append(df)
    print ("Saved %d partial structure factor measurements in file %s" % (len(df), ofile))

DF = pandas.concat( all_dfs)
DF.to_pickle(ofile)



