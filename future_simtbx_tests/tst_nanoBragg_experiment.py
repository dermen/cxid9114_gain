"""
This test is designed to explore
what happens when repeatedly
simulating and indexing a crystal model

After each indexing step, the crystal
model is updated to the indexing solution
and then simulation, and indexing follow

We track the evolution of the unit cell

"""
from __future__ import division

import numpy as np
import copy
import io, sys

import simtbx.nanoBragg
from scitbx.matrix import sqr
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from dxtbx.model.detector import DetectorFactory

from dials.algorithms.spot_finding.factory import FilterRunner
from dials.model.data import PixelListLabeller, PixelList
from dials.algorithms.spot_finding.finder import PixelListToReflectionTable
from dxtbx.imageset import MemReader, MemMasker
from dxtbx.datablock import DataBlockFactory
from dxtbx.imageset import ImageSet, ImageSetData
from libtbx.utils import Sorry
from scitbx.array_family import flex

from scitbx.matrix import col
from dials.algorithms.indexing.indexer import indexer_base
from dials.algorithms.indexing.indexer import master_phil_scope \
    as indexer_phil_scope


from dials.algorithms.indexing import index_reflections
from dials.algorithms.indexing.stills_indexer import e_refine

import dxtbx_cspad

# -------------------
# adjust these
n_sim = 20  # how many times to iterate the simulation - index  step
lab_geom = "canonical"   # can be canonical or cspad
silence_indexer = True  # silence stills indexer so we can watch the unit cell evolve
abc_tol = .5  # Allowed Angstrom deviation in the unit cell lengths
# might also want to adjust, add indexing params below, see variable idxpar
# -------------------

class FormatInMemory:
    def __init__(self, image, mask=None):
        """
        :param image: numpy array of an image or images
        :param mask: mask same shape as image
        """
        self.image = image
        if image.dtype != np.float64:
            self.image = self.image.astype(np.float64)
        if mask is None:
            self.mask = np.ones_like( self.image).astype(np.bool)
        else:
            assert (mask.shape==image.shape)
            assert( mask.dtype == bool)
            self.mask = mask

    def get_raw_data(self):
        if len(self.image.shape)==2:
            return flex.double(self.image)
        else:
            return tuple( [flex.double(panel) for panel in self.image])

    def get_mask(self, goniometer=None):
        if len(self.image.shape)==2:
            return flex.bool(self.mask)
        else:
            return tuple( [flex.bool(panelmask) for panelmask in self.mask])


def datablock_from_numpyarrays(image, detector, beam, mask=None):
    """
    put the numpy array image(s) into a dials datablock
    :param image:  numpy array of images or an image
    :param detector: dxtbx det
    :param beam: dxtbx beam
    :param mask: mask same shape as image , 1 is not masked, boolean
    :return:
    """
    if isinstance( image, list):
        image = np.array( image)
    if mask is not None:
        if isinstance( mask, list):
            mask = np.array(mask).astype(bool)
    I = FormatInMemory(image=image, mask=mask)
    reader = MemReader([I])
    masker = MemMasker([I])
    iset_Data = ImageSetData(reader, masker)
    iset = ImageSet(iset_Data)
    iset.set_beam(beam)
    iset.set_detector(detector)
    dblock = DataBlockFactory.from_imageset([iset])[0]
    return dblock


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, **kwargs ):
    """
    gets a reflection table from simulated panels
    :param panel_imgs: list of numpy arrays , each array is a simulated panel image
    :param detector: dxtbx detector, can have multiple nodes
    :param beam: dxtbx beam
    :param thresh: threshol
    :param filter:
    :param kwargs:
    :return: reflection table, with id coloumn set to 0
    """
    pxlst_labs = []
    for i in range(len(detector)):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append( plab)

    pixlst_to_reftbl = PixelListToReflectionTable(
        min_spot_size=1,
        max_spot_size=194*184,
        filter_spots=FilterRunner(),
        write_hot_pixel_mask=False)

    dblock = datablock_from_numpyarrays( panel_imgs, detector, beam)
    iset = dblock.extract_imagesets()[0]
    refls = pixlst_to_reftbl(iset, pxlst_labs)[0]
    refls['id'] = flex.int(len(refls),0)
    return refls

def sim_cryst_on_det(crystal, detector, beam, recenter=True):
    """
    return the basic simulation of the crystal scattering into
    each detector node

    sim parameters are hard set below
    :param crystal:  dxtbx crystal
    :param detector:  dxtbx detector
    :param beam:  dxtbx beam
    :param recenter: whether to recenter, this is only necessary for a messy cspad
        where individual panels are out of plane from one another.
    :return:  simulated panels as numpy arrays in a list
    """
    pad_pix = []
    for pid in range(len(detector)):
        #print("Simulating into panel %d " % pid)
        SIM = simtbx.nanoBragg.nanoBragg( detector, beam, verbose=1,panel_id=pid)

        SIM.Amatrix = sqr(crystal.get_A()).transpose().elems
        if recenter:
            SIM.beam_center_mm = detector[pid].get_beam_centre(beam.get_s0())
        SIM.default_F = 100
        SIM.F000 = 0
        SIM.xtal_shape = simtbx.nanoBragg.shapetype.Tophat  # makes nice spots with no shape transforms
        SIM.oversample = 5
        SIM.Ncells_abc = (10,10,10)
        SIM.add_nanoBragg_spots()
        pad_pix.append( SIM.raw_pixels.as_numpy_array() )
    return pad_pix

def sim_simple(crystal, detector, beam,):
    """
    simulate a single slab in canonical setup
    :param crystal:  dxtbx crystal
    :param detector:  dxtbx detector
    :param beam:  dxtbx beam
    :return: simulated panel image pixels, as a list of 1 element
        ( in line with above simulator)
    """
    S = simtbx.nanoBragg.nanoBragg(detpixels_slowfast=detector[0].get_image_size(),
                                   pixel_size_mm=detector[0].get_pixel_size()[0],
                                   unit_cell_Adeg=crystal.get_unit_cell(),
                                   distance_mm=detector[0].get_distance(),
                                   oversample=2,
                                   wavelength_A = beam.get_wavelength(),
                                   verbose=0)

    S.beamcenter_convention = simtbx.nanoBragg.convention.DIALS

    S.default_F = 100
    S.F000 = 0
    S.xtal_shape = simtbx.nanoBragg.shapetype.Tophat  # makes nice spots with no shape transforms
    S.oversample=2
    S.Ncells_abc = (10,10,10)
    S.add_nanoBragg_spots()
    return [S.raw_pixels.as_numpy_array() ]

# Here is our dxtbx crystal description
# this will be our ground truth
cryst_descr = {'__id__': 'crystal',
               'real_space_a': (150., 0, 0),
               'real_space_b': (0, 200., 0),
               'real_space_c': (0, 0, 100.),
               'space_group_hall_symbol': '-P 2 2'}
cryst = CrystalFactory.from_dict(cryst_descr)


# Lets define the indexing parameters for when we need them later:
idxpar = indexer_phil_scope.extract()
idxpar.indexing.known_symmetry.space_group = cryst.get_space_group().info()
idxpar.indexing.known_symmetry.unit_cell = cryst.get_unit_cell()
idxpar.indexing.method = "fft1d"
idxpar.indexing.fft1d.characteristic_grid = 0.029
idxpar.indexing.multiple_lattice_search.max_lattices = 1
idxpar.indexing.stills.indexer = 'stills'
idxpar.indexing.stills.refine_all_candidates = True
idxpar.indexing.stills.refine_candidates_with_known_symmetry = True 
idxpar.indexing.stills.candidate_outlier_rejection = False
idxpar.indexing.debug = False
idxpar.refinement.verbosity = 0
#idxpar.indexing.refinement_protocol.mode = "repredict_only"
idxpar.refinement.parameterisation.beam.fix = "all"
idxpar.refinement.parameterisation.detector.fix_list = ["origin"]
idxpar.refinement.parameterisation.crystal.fix = "all"

# ------------------------


if lab_geom == "canonical":
    s = 1  # scale factor, divide pixel size by this factor 
    pixsize = .10992/s # mm
    detdist = 125  # mm
    wavelen = 1.385
    orig = col((-s*1536*pixsize/2.,
                s*1536*pixsize/2.,
                -detdist))
    # Initialise detector frame
    fast = col((1.0, 0.0, 0.0))
    slow = col((0.0, -1.0, 0.0))
    det = DetectorFactory.make_detector(
        "", fast, slow, orig,
        (pixsize, pixsize), (s*1536,s*1536)) #, trusted_range=(0, 10000000))
    beam_descr = {
        'direction': (7.010833160725592e-06, -3.710515413340211e-06, 0.9999999999685403),
        'divergence': 0.0,
        'flux': 0.0,
        'polarization_fraction': 0.999,
        'polarization_normal': (0.0, 1.0, 0.0),
        'sigma_divergence': 0.0,
        'transmission': 1.0,
        'wavelength': 1.385}
    #beam = BeamFactory.simple(wavelen)
    beam = BeamFactory.from_dict(beam_descr)

elif lab_geom == "cspad":
    det = DetectorFactory.from_dict(dxtbx_cspad.cspad)

    # beam pointing off z-axis
    beam_descr = {
        'direction': (7.010833160725592e-06, -3.710515413340211e-06, 0.9999999999685403),
        'divergence': 0.0,
        'flux': 0.0,
        'polarization_fraction': 0.999,
        'polarization_normal': (0.0, 1.0, 0.0),
        'sigma_divergence': 0.0,
        'transmission': 1.0,
        'wavelength': 1.385}
    beam = BeamFactory.from_dict(beam_descr)


pix = sim_cryst_on_det(cryst, det, beam)
truth_refls = refls_from_sims(pix, det, beam)
dblock = datablock_from_numpyarrays(pix, det, beam)
isets = dblock.extract_imagesets()
iset = isets[0]


# current reflection table
curr_refls = copy.deepcopy(truth_refls)
curr_det = copy.deepcopy(det)

all_crysts = []
_stdout = sys.stdout
for _ in range(n_sim):
    #for i in range(len(curr_refls)):
    #    x,y,z = curr_refls[i]['xyzobs.px.value']
    #    curr_refls['xyzobs.px.value'][i] = x-1, y-1,z
    iset.set_detector( curr_det)
    
    orient = indexer_base.from_parameters(
        reflections=curr_refls,
        imagesets=[iset],  # there is only one
        params=idxpar)

    try:
        if silence_indexer:
            sys.stdout = io.BytesIO()
            orient.index()
            sys.stdout = _stdout
        else:
            orient.index()
    except (Sorry, RuntimeError):
        sys.stdout = _stdout
        print("Exiting prematurely because indexing failed")
        break

    new_cryst = orient.refined_experiments.crystals()[0]
    
    curr_det = orient.refined_experiments.detectors()[0]

    pix = sim_cryst_on_det(new_cryst, curr_det, beam)
    curr_refls = refls_from_sims(pix, curr_det,beam)

    # now lets check the overlap
    # and that its reasonable and centered on 0
    all_crysts.append( new_cryst)
    print np.round( new_cryst.get_unit_cell().parameters(), 2)
    print curr_det[0].get_beam_centre(beam.get_s0())
    print
# assert the cell lengths havent changed too much
a,b,c,_,_,_ = cryst.get_unit_cell().parameters()
a2,b2,c2,_,_,_ = new_cryst.get_unit_cell().parameters()
assert(np.allclose( (a,b,c), (a2,b2,c2), atol=abc_tol))

if __name__=="__main__":
 print "OK"
