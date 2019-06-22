import logging

logging.basicConfig(filename="_msi_dump.log", level=logging.INFO)

def msi(n_jobs, jid, out_dir, tag, glob_str ):
    """
    msi stands for mad --> spot --> index

    Here we load a data image that has crystal diffraction
    then we spot peaks and index the crystal image
    """
    import sys
    import numpy as np
    import glob
    import os
    from copy import deepcopy
    
    from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
    import dxtbx
    from dxtbx.datablock import DataBlockFactory
    from dxtbx.model.experiment_list import ExperimentListFactory, ExperimentList, Experiment
    from dials.array_family import flex
    from libtbx.utils import Sorry as Sorry
    from libtbx.phil import parse
    from cxi_xdr_xes.two_color import two_color_indexer 
    indexer_two_color = two_color_indexer.indexer_two_color
    
    from cxid9114 import utils
    from cxid9114 import parameters
    from cxid9114.index.ddi import params as mad_index_params
    from cxid9114.spots import spot_utils
    
    fnames = glob.glob(glob_str)
    #fnames = glob.glob("./bigsimnoiseless/job*/*.img.h5")
    #fnames = glob.glob("./bigsimh5/job*/*.img.h5")

    out_dir =os.path.join( out_dir , "job%d" % jid)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#   track shots that indexed, or shots that 
#   had too few spots to index, so can change parameters and try again
    def load_tracker_f(fname):
        data = []
        if os.path.exists(fname):
            data = np.loadtxt(fname, str)
            if data.size and not data.shape:
                data = list(set(data[None].astype(int)))
            else:
                data = list(set(data.astype(int)))
        return data

    skip_weak = False #True #False
    skip_failed = False
    skip_indexed = False #True #False
    weak_shots_f = os.path.join(out_dir, "weak_shots.txt")
    failed_idx_f = os.path.join(out_dir, "failed_shots.txt")
    indexed_f = os.path.join(out_dir, "indexed_shots.txt")
    weak_shots = load_tracker_f(weak_shots_f)
    failed_shots = load_tracker_f(failed_idx_f)
    indexed_shots = load_tracker_f(indexed_f)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#   --- spotting parameters
    spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
    spot_par.spotfinder.threshold.dispersion.global_threshold = 1 #1
    spot_par.spotfinder.threshold.dispersion.gain = 1 #28.
    spot_par.spotfinder.threshold.dispersion.kernel_size = [4,4]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 1 #5 #2.25
    spot_par.spotfinder.threshold.dispersion.sigma_background = 6 #1 #6.
    spot_par.spotfinder.filter.min_spot_size = 1 #3
    spot_par.spotfinder.force_2d = True

#   ------ indexing parameters
    #mad_index_params.indexing.two_color.spiral_method = (1., 100000) # 1000000)
    mad_index_params.indexing.two_color.n_unique_v = 22
    #mad_index_params.indexing.two_color.block_size = 25
    #mad_index_params.indexing.two_color.filter_by_mag = (10,3)
    #mad_index_params.indexing.refinement_protocol.mode = "repredict_only" # ignore"
#   ------

    from cctbx import crystal
    KNOWN_SYMMETRY = crystal.symmetry("78.95,78.95,38.13,90,90,90", "P43212")
    mad_index_params.indexing.known_symmetry.space_group = KNOWN_SYMMETRY.space_group_info()
    mad_index_params.indexing.known_symmetry.unit_cell = KNOWN_SYMMETRY.unit_cell()
    mad_index_params.indexing.real_space_grid_search.characteristic_grid = 0.02
    mad_index_params.indexing.two_color.high_energy = parameters.ENERGY_HIGH
    mad_index_params.indexing.two_color.low_energy = parameters.ENERGY_LOW
    mad_index_params.indexing.two_color.avg_energy = parameters.ENERGY_LOW * .5 + parameters.ENERGY_HIGH * .5

    N = len(fnames)

    idx_split = np.array_split( np.arange(N), n_jobs)

    n_idx = 0 # number indexed

    for idx in idx_split[jid]:
        img_f = fnames[idx]
        
        loader = dxtbx.load(img_f)
        iset = loader.get_imageset( loader.get_image_file() )    
        DET = loader.get_detector()
        BEAM = loader.get_beam()
        El = ExperimentListFactory.from_imageset_and_crystal( iset, crystal=None)
        
        if idx in weak_shots and skip_weak:
            print("Skipping weak shots %d" % idx)
            continue
        if idx in failed_shots and skip_failed:
            print("Skipping failed idx shots %d" % idx)
            continue
        if idx in indexed_shots and skip_indexed:
            print("Skipping already idx shots %d" % idx)
            continue

        refls_strong = flex.reflection_table.from_observations(El, spot_par)
        
        if len(refls_strong) < 10:
            print("Not enough spots shot %d, continuing!" % idx)
            weak_shots.append(idx)
            try:
                np.savetxt(weak_shots_f, weak_shots, fmt="%d")
            except:
                pass
            continue
     
        # ==================================
        # 2 color indexer of 2 color pattern
        # ==================================
        try:
            orientAB = indexer_two_color(
                reflections=spot_utils.as_single_shot_reflections(refls_strong, inplace=False),
                experiments=El,
                params=mad_index_params)
            orientAB.index()
        except (Sorry, RuntimeError) as error:
            print("####\nIndexingFailed  T_T \n####")
            print (error)
            failed_shots.append( idx)
            try:
                np.savetxt(failed_idx_f, failed_shots, fmt="%d")
            except:
                pass
            continue
        print("####\n *_* IndexingWORKED %d *_* \n####" % idx)
        n_idx += 1
        indexed_shots.append(idx)
        try:
            np.savetxt(indexed_f, indexed_shots, fmt="%d")
        except:
            pass
        crystalAB = orientAB.refined_experiments.crystals()[0]
        
        #El.crystal = crystalAB
        #exp_json = os.path.join(out_dir, "exp_%d_%s.json" % (idx, tag) )
        refl_pkl = os.path.join(out_dir, "refl_%d_%s.pkl" % (idx, tag))
        #orientAB.export_as_json(El, exp_json)
        utils.save_flex(refls_strong, refl_pkl)
        
        #beamA = deepcopy(iset.get_beam())
        #beamB = deepcopy(iset.get_beam())
        #waveA = parameters.ENERGY_CONV / mad_index_params.indexing.two_color.low_energy
        #waveB = parameters.ENERGY_CONV / mad_index_params.indexing.two_color.high_energy
        #beamA.set_wavelength(waveA)
        #beamB.set_wavelength(waveB)
        dump = {"crystalAB": crystalAB, #"best_ucell": orientAB.best_crystal,
                #"bestA":orientAB.best_crystalA,
                #"bestU":orientAB.best_crystalU,
                #"bestB":orientAB.best_crystalB,
                "img_f":img_f, #"beamA":beamA, "beamB":beamB,
                "refined_refls_v1": orientAB.refined_reflections,
                 #"rmsd": orientAB.best_rmsd,
                 "refls_strong": refls_strong}

        dump_pkl = os.path.join(out_dir, "dump_%d_%s.pkl" % (idx, tag))
        utils.save_flex(dump,  dump_pkl)

if __name__=="__main__":
    from joblib import Parallel,delayed
    import sys
    n_jobs = int(sys.argv[1])
    outdir = sys.argv[2]

    tag = "data"
    glob_str = sys.argv[3]
    #outdir = "/global/project/projectdirs/lcls/dermen/indexined"
    #tag = sys.argv[2]  # tag to attach to output files
    #outdir = outdir + tag  
    Parallel(n_jobs=n_jobs)(\
        delayed(msi)(n_jobs, jid, outdir, tag, glob_str) \
        for jid in range(n_jobs))

