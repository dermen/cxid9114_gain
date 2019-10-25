#!/usr/bin/env libtbx.python

import argparse
import os
parser = argparse.ArgumentParser("make dadat")
parser.add_argument("-o", dest='ofile', type=str, help="file out")
parser.add_argument("-odir", dest='odir', type=str, help="file outdir", default="_sims64res")
parser.add_argument("--seed",type=int, dest='seed', default=None, help='random seed for orientation' )
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
parser.add_argument("--write-img", dest='write_img', action='store_true')

parser.add_argument('-N', dest='nodes', type=int, default=[1,0], nargs=2, help="Number of nodes, and node id")

parser.add_argument("--write-sim-img", 
    dest='write_sim_img', action='store_true')
parser.add_argument("-deltaq", type=float, default=0.065,
            help="q spacing for integration boxes (2PI convention)")

parser.add_argument("-trials", dest='num_trials', help='trials per worker', 
    type=int, default=1 )
args = parser.parse_args()

use_dials_spotter = args.dials_spot
smi_stride = 100
thresh=args.thresh
Gauss = args.Gauss
cuda = args.gpu
add_background = args.add_bg
add_noise = args.add_noise
use_truth_crystal = args.truth_cryst
rel_dir = args.reldir
sz = args.fudge
kernels_per_gpu = 1
num_nodes, node_id = args.nodes

GAIN=28
ofile = args.ofile
beamsize_mm = 0.001
boost = 1.0
use_data_spec = args.real_spec
exposure_s = 1
Ncells_abc =(22,22,22) #(23,23,23)
mos_doms = 100
mos_spread = .015
make_background = args.make_bg
bg_name = args.bg_name
overwrite = args.overwrite
ngpu = args.ngpu
oversample = 0

from joblib import Parallel,delayed


def random_rotation(deflection=1.0, randnums=None):
    r"""
    Creates a random rotation matrix.

    TODO: documentation

    Arguments:
        deflection (float): the magnitude of the rotation. For 0, no rotation; for 1, competely random
                            rotation. Small deflection => small perturbation.
        randnums (numpy array): 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    Returns:
        (numpy array) Rotation matrix
    """
    import numpy as np
    # from
    # http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    vec = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    rot = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    mat = (np.outer(vec, vec) - np.eye(3)).dot(rot)
    return mat.reshape(3, 3)


def main(rank):

    device_Id = rank % ngpu

    worker_Id = node_id*ngpu + rank

    import os
    import sys
    from copy import deepcopy
    import glob
    from itertools import izip

    from scipy.spatial import distance
    import h5py
    import scipy.ndimage
    from IPython import embed
    import numpy as np
    import pandas
    from scipy.spatial import cKDTree
   
    from simtbx.nanoBragg import shapetype, nanoBragg 
    from libtbx.phil import parse 
    from scitbx.matrix import sqr
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
    from cxid9114.spots import integrate, spot_utils
    from cxid9114 import parameters
    from cxid9114.sim import sim_utils
    from cctbx import miller, sgtbx
    from cxid9114 import utils
    from cxid9114.bigsim import sim_spectra
    from cxid9114.refine.jitter_refine import make_param_list

    spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
    spot_par.spotfinder.threshold.dispersion.global_threshold = 40
    spot_par.spotfinder.threshold.dispersion.gain = GAIN
    spot_par.spotfinder.threshold.dispersion.kernel_size = [2,2]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 1 
    spot_par.spotfinder.threshold.dispersion.sigma_background = 6 
    spot_par.spotfinder.filter.min_spot_size = 3
    spot_par.spotfinder.force_2d = True

    odir = args.odir
    odirj = os.path.join(odir, "job%d" % worker_Id)
    
    #all_pkl_files = [s for sl in \
    #    [ files for _,_, files in  os.walk(odir)]\
    #        for s in sl if s.endswith("pkl")]
    
    #print "Found %d pkl files already in %s!" \
    #    % (len(all_pkl_files), odir)

    if not os.path.exists(odirj):
        os.makedirs(odirj)

    hkl_tol = .15
    run = 61 
    shot_idx = 0 
    ENERGIES = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # colors of the beams
    FF = [10000, None]  

    cryst_descr = {'__id__': 'crystal',
                  'real_space_a': (79, 0, 0),
                  'real_space_b': (0, 79, 0),
                  'real_space_c': (0, 0, 38),
                  'space_group_hall_symbol': '-P 4 2'}
    crystalAB = CrystalFactory.from_dict(cryst_descr)

    sfall_main = sim_spectra.load_spectra("../bigsim/test_sfall.h5")
    FFdat = [sfall_main[19], sfall_main[110]]

    FLUX = [1e11, 1e11]  # fluxes of the beams

    chanA_flux = np.random.uniform(1e11,1e12) 
    chanB_flux = np.random.uniform(1e11,1e12) 
    FLUXdat = [chanA_flux, chanB_flux]

    waveA = parameters.ENERGY_CONV / ENERGIES[0]
    waveB = parameters.ENERGY_CONV / ENERGIES[1]

    from cxid9114.bigsim.bigsim_geom import DET,BEAM

    detector = DET
    
    print("Rank %d Begin" % worker_Id)
    for i_data in range( args.num_trials):
        pklname = "%s_rank%d_data%d.pkl" % (ofile, worker_Id, i_data)
        pklname = os.path.join( odirj, pklname) 

        print("<><><><><><><")
        print("Job %d:  trial  %d / %d" % ( worker_Id, i_data+1, args.num_trials ))
        print("<><><><><><><")
        
        if (worker_Id==0 and i_data % smi_stride==0 and cuda):
            print("GPU status")
            os.system("nvidia-smi")

            print("\n\n")
            print("CPU memory usage")
            mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
            os.system(mem_usg)

        beamA = deepcopy(BEAM)
        beamB = deepcopy(BEAM)
        beamA.set_wavelength(waveA)
        beamB.set_wavelength(waveB)

        SCALE = np.random.uniform(0.1,10)

        np.random.seed(args.seed)
        crystalAB = CrystalFactory.from_dict(cryst_descr)
        randnums = np.random.random(3)
        Rrand = random_rotation(1, randnums)
        crystalAB.set_U(Rrand.ravel())
        
        #pert = np.random.uniform(0.0001/2/np.pi, 0.0003 / 2. /np.pi)
        #print("PERT %f" % pert)
        #Rsmall = random_rotation(0.00001, randnums ) #pert)
       
       
        params_lst = make_param_list(crystalAB, DET, BEAM, 
            1, rot=0.08, cell=.0000001, eq=(1,1,0),
            min_Ncell=23, max_Ncell=24, 
            min_mos_spread=0.02, 
            max_mos_spread=0.08)
        Ctruth = params_lst[0]['crystal']
          
        print Ctruth.get_unit_cell().parameters()
        print crystalAB.get_unit_cell().parameters()
         
        init_comp = rotation_matrix_differences((Ctruth, crystalAB))
        init_rot = float(init_comp.split("\n")[-2].split()[2])

        if use_data_spec:
            print "NOT IMPLEMENTED, Using a phony 2col spectrum to simulate the data"
            data_fluxes = FLUXdat
            data_energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]
            data_ff = FFdat
        else:
            print "Using a phony two color spectrum to simulate the data"
            data_fluxes = FLUXdat
            data_energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]
            data_ff = FFdat

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
                exposure_s=exposure_s, beamsize_mm=beamsize_mm, device_Id=device_Id,
                boost=boost)
       
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
            mos_spread=mos_spread, boost=boost,
            exposure_s=exposure_s, beamsize_mm=beamsize_mm,
            only_water=only_water, device_Id=device_Id)
            
        simsDataSum = SCALE*np.array(simsDataSum)
        
        if make_background:
            bg_out = h5py.File(bg_name, "w")
            bg_out.create_dataset("bigsim_d9114",data=simsDataSum[0])
            print "Background made! Saved to file %s" % bg_name
            sys.exit()
        
        if add_background:
            print("ADDING BG")
            background = h5py.File(bg_name, "r")['bigsim_d9114'][()]
            bg_scale = np.sum([39152412349.12075, 32315440627.406036] )
            bg_scale = np.sum(data_fluxes) / bg_scale
            print "%.3e backgorund scale" % bg_scale 
            print "BG shape", background.shape
            simsDataSum[0] += background * bg_scale

        if add_noise:
            print("ADDING NOISE")
            for pidx in range(1):
                SIM = nanoBragg(detector=DET, beam=BEAM, panel_id=pidx)
                SIM.exposure_s = exposure_s
                SIM.beamsize_mm = beamsize_mm
                SIM.flux = np.sum(data_fluxes)
                SIM.detector_psf_kernel_radius_pixels=5;
                #SIM.detector_psf_type=shapetype.Gauss
                SIM.detector_psf_type=shapetype.Unknown  # for CSPAD
                SIM.detector_psf_fwhm_mm=0
                SIM.quantum_gain = GAIN
                SIM.raw_pixels = flex.double(simsDataSum[pidx].ravel())
                SIM.add_noise()
                simsDataSum[pidx] = SIM.raw_pixels.as_numpy_array()\
                    .reshape(simsDataSum[0].shape)    
                SIM.free_all()
                del SIM

        if args.write_img:
            print "SAVING DATAFILE"
            h5name = "%s_rank%d_data%d.h5" % (ofile, worker_Id, i_data)
            h5name = os.path.join( odirj, h5name)
            fout = h5py.File(h5name,"w" ) 
            fout.create_dataset("bigsim_d9114", data=simsDataSum[0])
            fout.create_dataset("crystalAB", data=crystalAB.get_A() )
            fout.create_dataset("dataCryst", data=dataCryst.get_A() )
            fout.close()  

        if args.write_sim_img:
            print "SAVING DATAFILE"
            for i_sim in simsAB:
                sim_h5name = "%s_rank%d_sim%d_%d.h5" % (ofile, worker_Id, i_data, i_sim)
                sim_h5name = os.path.join( odirj, sim_h5name)
                
                fout = h5py.File(sim_h5name,"w" ) 
                fout.create_dataset("bigsim_d9114", 
                    data=simsAB[i_sim][0])
                fout.create_dataset("crystalAB", data=crystalAB.get_A() )
                fout.create_dataset("dataCryst", data=dataCryst.get_A() )
                fout.close()  

        print "RELFS FROM SIMS"
        refl_simA = spot_utils.refls_from_sims(simsAB[0], detector, beamA, thresh=thresh)
        refl_simB = spot_utils.refls_from_sims(simsAB[1], detector, beamB, thresh=thresh)

        if use_dials_spotter:
            print("DIALS SPOTTING")
            El = utils.explist_from_numpyarrays(simsDataSum,DET,beamA)
            refl_data = flex.reflection_table.from_observations(El, spot_par)
            print("Found %d refls using DIALS spot finder" % len(refl_data)) 
        else:
            refl_data = spot_utils.refls_from_sims(simsDataSum, detector, beamA,\
                            thresh=thresh)

            print ("Found %d refls using threshold" % len(refl_data))
        
        if len(refl_data)==0:
            print "Rank %d: No reflections found! " % (worker_Id)
            continue
        
        residA = metrics.check_indexable2(
            refl_data, refl_simA, detector, beamA, crystalAB, hkl_tol)
        residB = metrics.check_indexable2(
            refl_data, refl_simB, detector, beamB, crystalAB, hkl_tol)

        FA = sfall_main[19] # utils.open_flex('SA.pkl')  # ground truth values
        FB = sfall_main[110] #utils.open_flex('SB.pkl')  # ground truth values
        HA = tuple([hkl for hkl in FA.indices()])
        HB = tuple([hkl for hkl in FB.indices()])
        
        HA_val_map = { h:data for h,data in izip(FA.indices(), FA.data())}
        HB_val_map = { h:data for h,data in izip(FB.indices(), FB.data())}
        Hmaps = [HA_val_map, HB_val_map] 

        def get_val_at_hkl(hkl, val_map):
            sg96 = sgtbx.space_group(" P 4nw 2abw")
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
        
        if use_data_spec:
            print "Setting LA, LB as sums over flux regions A,B"
            LA = data_fluxes[:75].sum()
            LB = data_fluxes[75:].sum()
        else:
            print "Setting LA LB as data_fluxes"
            LA = data_fluxes[0] 
            LB = data_fluxes[1] 
        
        K=FF[0] ** 2 * FLUX[0]
        
        L_at_color = [LA,LB] 

        out = spot_utils.integrate_boxes(
            refl_data, simsDataSum[0], [refl_simA, refl_simB], DET,
            [beamA,beamB], crystalAB, delta_q=args.deltaq, gain=GAIN )

        Nh = len(out[0])
        rhs = []
        lhs = []
        all_H2 = []
        all_PA = []
        all_PB = []
        all_FA = []
        all_FB = []
        for i in range(Nh):

            HKL = out[0][i]
            yobs = out[1][i]
            Pvals = out[2][i]
            ycalc = 0 

            for i_P, P in enumerate(Pvals):
                L = L_at_color[i_P]
                H2, F = get_val_at_hkl(HKL, Hmaps[i_P])
                if i_P==0:
                    all_FA.append(F)
                else:
                    all_FB.append(F)

                ycalc += SCALE*L*P*abs(F)**2/K
            all_PA.append( Pvals[0])
            all_PB.append( Pvals[1]) 
            all_H2.append(H2)
            rhs.append(ycalc)
            lhs.append(yobs)
        
        df = pandas.DataFrame({"rhs":rhs, "lhs": lhs, 
            "PA":all_PA, "PB":all_PB, "FA": all_FA, 
            "FB": all_FB})

        df["run"] = run
        df["shot_idx"] = shot_idx
        df['gain'] = SCALE 

        df['LA'] = LA
        df['LB'] = LB
        df['K'] = K 
        df['init_rot'] = init_rot
       
        h,k,l = zip(*all_H2) 
        df['h2'] = h
        df['k2'] = k
        df['l2'] = l

        df.to_pickle(pklname)

        if args.plot:
            print("PLOT")
            import pylab as plt
            plt.plot( df.lhs, df.rhs, '.')
            plt.show()
        print("DonDonee")

if __name__=="__main__":
    Parallel(n_jobs=ngpu*kernels_per_gpu)(\
        delayed(main)(rank) for rank in range(ngpu*kernels_per_gpu) )


