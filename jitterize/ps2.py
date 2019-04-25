
import simtbx.nanoBragg
from scitbx.matrix import sqr
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from dxtbx.model.detector import DetectorFactory, Detector, Panel
from simtbx.nanoBragg.dxtbx_cspad import cspad
from cctbx import sgtbx

# dxtbx beam model description
beam_descr = {'direction': (0.0, 0.0, 1.0),
             'divergence': 0.0,
             'flux': 1e12,
             'polarization_fraction': 1.,
             'polarization_normal': (0.0, 1.0, 0.0),
             'sigma_divergence': 0.0,
             'transmission': 1.0,
             'wavelength': 1.4}

# dxtbx crystal description
hall_sym = sgtbx.space_group_info(number=19).type().hall_symbol()  # ' P 2ac 2ab'

cryst_descr = {'__id__': 'crystal',
              'real_space_a': (127.7, 0, 0),
              'real_space_b': (0, 225.4, 0),
              'real_space_c': (0, 0, 306.1),
              'space_group_hall_symbol': hall_sym}

beam = BeamFactory.from_dict(beam_descr)
whole_det = DetectorFactory.from_dict(cspad)
cryst = CrystalFactory.from_dict(cryst_descr)

from cxid9114 import utils
utils.save_flex( whole_det, "ps2.det.pkl")
utils.save_flex( cryst, "ps2.cryst.pkl")
utils.save_flex( beam, "ps2.beam.pkl")


