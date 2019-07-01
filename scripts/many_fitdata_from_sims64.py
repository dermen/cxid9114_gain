#!/usr/bin/env libtbx.python

import argparse
import os
parser = argparse.ArgumentParser("make dadat")
parser.add_argument("-iglob", dest='datanames', type=str, help="name of data file",
    required=True )
parser.add_argument("-o", dest='ofile', type=str, help="file out")
parser.add_argument("-odir", dest='odir', type=str, help="file outdir", default="_sims64res")
parser.add_argument("--gpu", dest='gpu', action='store_true', help='sim with GPU')
parser.add_argument("--plot", dest='plot', action='store_true', help='plot Fobs vs Fmod')
parser.add_argument("-rel-dir", dest='reldir', default=None, type=str, help="relative dir for image file loading")
parser.add_argument("--truth-crystal", dest='truth_cryst', action='store_true',
    help="use the truth crystal")
parser.add_argument("--add-bg", dest="add_bg",action='store_true',help="add background" )
parser.add_argument("--add-noise", dest="add_noise",action='store_true',help="add noise" )
parser.add_argument("--gauss", dest="Gauss", action='store_true', help="use gaussian profile")
parser.add_argument("--fudge", dest='fudge', default=5, type=int, 
    help='integration fudge factor to allow predicted and observed spots to not overlap perfectly')
parser.add_argument("--real-spec",dest='real_spec', action='store_true', 
        help="use a real spectrum")
parser.add_argument("-thresh", dest='thresh', default=0, type=float, 
        help="spot threshold for simple spot detection")
parser.add_argument("--make-background", dest='make_bg', action='store_true',
    help="Just make the background image and quit")
parser.add_argument("-bg-name", dest='bg_name', default='background64.h5',
    type=str, help="name of the background file, either to make/overwrite, or load (default)")
parser.add_argument("--dials-spotter", dest='dials_spot', 
        action='store_true', help="use DIALS to find spots")
parser.add_argument("-g", dest='ngpu', type=int, default=1,help='number of gpu' )
parser.add_argument("--overwrite", dest='overwrite',action='store_true',help='overwrite files' )
args = parser.parse_args()

use_dials_spotter = args.dials_spot
smi_stride = 3
thresh=args.thresh
Gauss = args.Gauss
iglob = args.datanames
cuda = args.gpu
add_background = args.add_bg
add_noise = args.add_noise
use_truth_crystal = args.truth_cryst
rel_dir = args.reldir
sz = args.fudge

ofile = args.ofile
beamsize_mm = 0.001
boost=467
use_data_spec = args.real_spec
exposure_s = 1
Ncells_abc =(23,23,23)
mos_doms = 1 #000
mos_spread = 0 #.015
make_background = args.make_bg
bg_name = args.bg_name
overwrite = args.overwrite
ngpu = args.ngpu
oversample = 0

from joblib import Parallel,delayed

def main(rank):

    import os
    import sys
    from copy import deepcopy
    import glob
    from itertools import izip

    import h5py
    import scipy.ndimage
    from IPython import embed
    import numpy as np
    import pandas
    from scipy.spatial import cKDTree
   
    from simtbx.nanoBragg import shapetype, nanoBragg 
    from libtbx.phil import parse 
    import dxtbx
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dxtbx.model.crystal import CrystalFactory
    from dials.algorithms.indexing.compare_orientation_matrices \
            import rotation_matrix_differences
    from dials.array_family import flex
    from dials.command_line.find_spots import phil_scope as find_spots_phil_scope

    from cxid9114.refine import metrics
    from cxid9114 import utils
    from cxid9114.geom import geom_utils
    from cxid9114.spots import spot_utils
    from cxid9114 import parameters
    from cxid9114.sim import sim_utils
    from cctbx import miller, sgtbx
    from cxid9114 import utils
    from cxid9114.bigsim import sim_spectra

    spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
    spot_par.spotfinder.threshold.dispersion.global_threshold = 40
    spot_par.spotfinder.threshold.dispersion.gain = 28 
    spot_par.spotfinder.threshold.dispersion.kernel_size = [2,2]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 1 
    spot_par.spotfinder.threshold.dispersion.sigma_background = 6 
    spot_par.spotfinder.filter.min_spot_size = 3
    spot_par.spotfinder.force_2d = True

    odir = args.odir
    odirj = os.path.join(odir, "job%d" % rank)

    if not os.path.exists(odirj):
        os.makedirs(odirj)

    hkl_tol = .15
    run = 61 
    shot_idx = 0 
    ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
    FF = [10000, None]  

    sfall_main = sim_spectra.load_spectra("../bigsim/test_sfall.h5")
    FFdat = [sfall_main[19], sfall_main[110]]

    FLUX = [1e11, 1e11]  # fluxes of the beams

    np.random.seed(41107)
    chanA_flux = 1e11 
    chanB_flux = 1e11 
    FLUXdat = [chanA_flux, chanB_flux]
    GAIN = 1 

    waveA = parameters.ENERGY_CONV / ENERGIES[0]
    waveB = parameters.ENERGY_CONV / ENERGIES[1]

    #from cxid9114.bigsim.bigsim_geom import DET,BEAM
    DET = utils.open_flex('ref1_det.pkl')
    BEAM = utils.open_flex('ref3_beam.pkl')

    detector = DET
    
    data_names = glob.glob(iglob)
    data_names_rank = np.array_split(data_names, ngpu)[rank]
    Ndatas = len( data_names_rank)
    print("Rank %d Begin" % rank)
    for i_data, data_name in enumerate(data_names_rank):
        pklname = "%s_rank%d_data%d.pkl" % (ofile, rank, i_data)
        pklname = os.path.join( odirj, pklname)
        if os.path.exists(pklname) and not overwrite:
            print ("Rank %d; Pkl name exists! %s" % (rank, pklname))
            continue
        
        print("<><><><><><><")
        print("Job %d data name %s (%d / %d)" % ( rank, data_name, i_data+1, Ndatas ))
        print("<><><><><><><")
        

        if (rank==0 and i_data % smi_stride==0):
            print("GPU status")
            os.system("nvidia-smi")

            print("\n\n")
            print("CPU memory usage")
            mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
            os.system(mem_usg)

        data = utils.open_flex(data_name)
        beamA = deepcopy(BEAM)
        beamB = deepcopy(BEAM)
        beamA.set_wavelength(waveA)
        beamB.set_wavelength(waveB)

        crystalAB = data["crystalAB"]

        img_f = data['img_f']
        if rel_dir is not None:
            img_f = os.path.join( rel_dir, img_f)
            if not os.path.exists(img_f):
                print ("Image file %s does not exists" % img_f)
                sys.exit()
        loader = dxtbx.load(img_f)

        cryst_descr = {'__id__': 'crystal',
                      'real_space_a': loader._h5_handle["real_space_a"][()],
                      'real_space_b': loader._h5_handle["real_space_b"][()],
                      'real_space_c': loader._h5_handle["real_space_c"][()],
                      'space_group_hall_symbol': \
                            loader._h5_handle["space_group_hall_symbol"][()]}

        if use_data_spec:
            print "Using a real spectrum to simulate the data"
            data_fluxes = loader._h5_handle["fluxes"][()]
            data_energies = loader._h5_handle["energies"][()]
            data_ff = sfall_main 
        else:
            print "Using a phony two color spectrum to simulate the data"
            data_fluxes = FLUXdat
            data_energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]
            data_ff = FFdat

        Ctruth = CrystalFactory.from_dict(cryst_descr)
        init_comp = rotation_matrix_differences((Ctruth, crystalAB))
        init_rot = float(init_comp.split("\n")[-2].split()[2])

        print  ("Truth crystal Misorientation deviation: %f deg" % init_rot )
        if args.truth_cryst:
            print "Using truth crystal"
            dataCryst = Ctruth
        else:
            print "Not using truth crystal"
            dataCryst = crystalAB

        if not make_background:
            print "SIMULATING Flat-Fhkl IMAGES"
            simsAB = sim_utils.sim_twocolors2(
                crystalAB, detector, BEAM, FF,
                [parameters.ENERGY_LOW, parameters.ENERGY_HIGH],
                FLUX, pids=None, Gauss=Gauss, cuda=cuda, oversample=oversample, 
                Ncells_abc=Ncells_abc, mos_dom=mos_doms, mos_spread=mos_spread,
                exposure_s=exposure_s, beamsize_mm=beamsize_mm, device_Id=rank)
       
        if make_background:
            print("MAKING BACKGROUND")
            spec_file = h5py.File("../bigsim/simMe_data_run62.h5", "r") 
            ave_spec = np.mean( spec_file["hist_spec"][()], axis=0)
            data_fluxes=[ave_spec[19], ave_spec[110] ]
            data_energies = spec_file["energy_bins"][()][[19,110]]
            data_ff = [1,1] #*len(data_energies)
            only_water=True
        else:
            only_water=False

        print "SIULATING DATA IMAGE"
        print data_fluxes
        simsDataSum = sim_utils.sim_twocolors2(
            dataCryst, detector, BEAM, data_ff, 
            data_energies, 
            data_fluxes, pids=None, Gauss=Gauss, cuda=cuda,oversample=oversample,
            Ncells_abc=Ncells_abc, accumulate=True, mos_dom=mos_doms, 
            mos_spread=mos_spread,
            exposure_s=exposure_s, beamsize_mm=beamsize_mm,
            only_water=only_water, device_Id=rank)
            
        simsDataSum = np.array(simsDataSum)
        
        if make_background:
            bg_out = h5py.File(bg_name, "w")
            bg_out.create_dataset("sim64_d9114_images",data=[simsDataSum])
            print "Background made! Saved to file %s" % bg_name
            sys.exit()
        
        if add_background:
            print("ADDING BG")
            background = h5py.File(bg_name, "r")['sim64_d9114_images'][()]
            simsDataSum += background[0]

        if add_noise:
            print("ADDING NOISE")
            for pidx in range(64):
                SIM = nanoBragg(detector=DET, beam=BEAM, panel_id=pidx)
                SIM.exposure_s = exposure_s
                SIM.beamsize_mm = beamsize_mm
                SIM.flux = np.sum(data_fluxes)
                SIM.detector_psf_kernel_radius_pixels=5;
                SIM.detector_psf_type=shapetype.Unknown  # for CSPAD
                SIM.detector_psf_fwhm_mm=0
                SIM.quantum_gain = 28
                SIM.raw_pixels = flex.double(simsDataSum[pidx].ravel())
                SIM.add_noise()
                simsDataSum[pidx] = SIM.raw_pixels.as_numpy_array()\
                    .reshape(simsDataSum[0].shape)    
                SIM.free_all()
                del SIM

        print "SAVING DATAFILE"
        h5name = "%s_rank%d_data%d.h5" % (ofile, rank, i_data)
        h5name = os.path.join( odirj, h5name)
        fout = h5py.File(h5name,"w" ) 
        fout.create_dataset("sim64_d9114_images", data=[simsDataSum])
        fout.close()  

        print "RELFS FROM SIMS"
        refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA, thresh=thresh)
        refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB, thresh=thresh)

        if use_dials_spotter:
            print("DIALS SPOTTING")
            loader = dxtbx.load(h5name)     
            iset = loader.get_imageset( loader.get_image_file() )[:1]
            #iset.set_detector(DET)
            iset.set_beam(BEAM)
            El = ExperimentListFactory.from_imageset_and_crystal( iset, crystal=None)
            refl_data = flex.reflection_table.from_observations(El, spot_par)
            print("Found %d refls using DIALS spot finder" % len(refl_data)) 
        else:
            refl_data = spot_utils.refls_from_sims(simsDataSum, detector, beamA,\
                            thresh=thresh)

            print ("Found %d refls using threshold" % len(refl_data))
        
        if len(refl_data)==0:
            print "Rank %d: No reflections found! %s" % (rank, data_name)
            continue
        
        residA = metrics.check_indexable2(
            refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
        residB = metrics.check_indexable2(
            refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)

        sg96 = sgtbx.space_group(" P 4nw 2abw")
        FA = sfall_main[19] # utils.open_flex('SA.pkl')  # ground truth values
        FB = sfall_main[110] #utils.open_flex('SB.pkl')  # ground truth values
        HA = tuple([hkl for hkl in FA.indices()])
        HB = tuple([hkl for hkl in FB.indices()])
        
        HA_val_map = { h:data for h,data in izip(FA.indices(), FA.data())}
        HB_val_map = { h:data for h,data in izip(FB.indices(), FB.data())}
        
        def get_val_at_hkl(hkl, val_map):
            poss_equivs = [i.h() for i in
                           miller.sym_equiv_indices(sg96, hkl).indices()]
            in_map=False
            for hkl2 in poss_equivs:
                if hkl2 in val_map:  # fast lookup
                    in_map=True
                    break
            if in_map:
                return hkl2, val_map[hkl2]
            else:
                return (None,None,None),-1

        filt=1 #True #`False #True
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

##          remake the trees given the drops
            _, all_HiA = spot_utils.refls_to_hkl(
                refl_simA, detector, beamA,
                crystal=crystalAB, returnQ=False)
            all_treeA = cKDTree(all_HiA)

            _, all_HiB = spot_utils.refls_to_hkl(
                refl_simB, detector, beamB,
                crystal=crystalAB, returnQ=False)
            #all_treeB = cKDTree(all_HiB)

##          CHECK if same HKL, indexed by both colors
#           exists on multiple panels, and if so, delete...
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
                "FB": [], "iA": [], "iB": [], "Nstrong": [], "pid": []}
        all_int_me = []

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
                rmax = geom_utils.twocolor_deltapix(detector[pid], beamA, beamB)
                merge_me = treeA.query_ball_tree(treeB, r=rmax + sz)

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

                    #if iA==79:
                    #    embed()
                    # integrate me if I am in the bounding box!
                    int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
                    if not int_me.size:
                        continue
                    mergesA.append(iA)
                    mergesB.append(iB)

                    # integrate the spot, this will change depending on data or simulation
                    totalI = 0
                    for ref_idx in int_me:
                        totalI += rpp[pid][ref_idx]["intensity.sum.value"]
                    PA = RA[iA]['intensity.sum.value']
                    PB = RB[iB]['intensity.sum.value']

                    # get the hkl structure factor, and the sym equiv hkl
                    (h, k, l) = HiA[iA]  # NOTE: same for A and B channels
                    (h2, k2, l2), FA = get_val_at_hkl((h, k, l), HA_val_map)
                    
                    _, FB = get_val_at_hkl((h, k, l), HB_val_map)  # NOTE: no need to return h2,k2,l2 twice
                    #if FB==-1 or FA==-1:
                    #    continue

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
            if inA:
                for iA, ref in enumerate(RA):
                    if iA in mergesA:
                        continue
                    x1A, x2A, y1A, y2A, _, _ = RA[iA]['bbox']  # ['shoebox'].bbox
                    xlow = max((0, x1A - sz))
                    xhigh = min((panX, x2A + sz))
                    ylow = max((0, y1A - sz))
                    yhigh = min((panY, y2A + sz))
                    int_me = np.where((xlow < x) & (x < xhigh) & (ylow < y) & (y < yhigh))[0]
                    if not int_me.size:
                        continue

                    totalI = 0
                    for ref_idx in int_me:
                        totalI += rpp[pid][ref_idx]["intensity.sum.value"]
                    PA = RA[iA]['intensity.sum.value']
                    PB = 0  # crucial ;)

                    # get the hkl structure factor, and the sym equiv hkl
                    (h, k, l) = HiA[iA]  # NOTE: same for A and B channels
                    (h2, k2, l2), FA = get_val_at_hkl((h, k, l), HA_val_map)
                    _, FB = get_val_at_hkl((h, k, l), HB_val_map)  # NOTE: no need to return h2,k2,l2 twice
                    #if FA==-1 or FB==-1:
                    #    continue
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

                    totalI = 0
                    for ref_idx in int_me:
                        totalI += rpp[pid][ref_idx]["intensity.sum.value"]

                    PA = 0  # crucial ;)
                    PB = RB[iB]['intensity.sum.value']

                    # get the hkl structure factor, and the sym equiv hkl
                    (h, k, l) = HiB[iB]  # NOTE: same for A and B channels
                    (h2, k2, l2), FB = get_val_at_hkl((h, k, l), HB_val_map)
                    _, FA = get_val_at_hkl((h, k, l), HA_val_map)  # NOTE: no need to return h2,k2,l2 twice
                    #if FA==-1 or FB==-1:
                    #    continue
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

        df = pandas.DataFrame(DATA)
        df["run"] = run
        df["shot_idx"] = shot_idx
        df['gain'] = GAIN

        if use_data_spec:
            print "Setting LA, LB as sums over flux regions A,B"
            df['LA'] = data_fluxes[:75].sum()
            df['LB'] = data_fluxes[75:].sum()
        else:
            print "Setting LA LB as data_fluxes"
            df['LA'] = data_fluxes[0] 
            df["LB"] = data_fluxes[1] 

        df['K'] = FF[0] ** 2 * FLUX[0]
        df["rhs"] = df.gain * (df.IA * df.LA * (df.PA / df.K) + df.IB * df.LB * (df.PB / df.K))
        df["lhs"] = df.D
        df['data_name'] = data_name
        df['init_rot'] = init_rot
        df.to_pickle(pklname)
        embed()
        print("PLOT")
        if args.plot:
            import pylab as plt
            plt.plot( df.lhs, df.rhs, '.')
            plt.show()
        print("DonDonee")

#if __name__=="__main__":
Parallel(n_jobs=ngpu)(delayed(main)(rank) for rank in range(ngpu) )


