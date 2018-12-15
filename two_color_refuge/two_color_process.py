from __future__ import division
# -*- Mode: Python; c-basic-offset: 2; indent-tabs-mode: nil; tab-width: 8 -*-
#
# LIBTBX_SET_DISPATCHER_NAME cxi_xdr_xes.two_color_process
#
"""
cctbx.xfel.process
  |
  |
  v
cxi_xdr_xes.two_color_process (override index,integrate)
"""
import libtbx.load_env
from dials.util.options import OptionParser
from iotbx.phil import parse
from cxi_xdr_xes.two_color.reflection_utils import remove_invalid_reflections
phil_scope = parse('''
  include scope xfel.command_line.xfel_process.phil_scope
''', process_includes=True)

two_color_phil_scope = parse('''
  indexing {
    two_color {
      debug = False
        .type = bool
        .help = Reflections for both wavelengths at all hkls
      low_energy = 7400
        .type = float
        .help = Low energy value in eV
      high_energy = 7500
        .type = float
        .help = High energy value in eV
      avg_energy = 7450
        .type = float
        .help = The average of both energies (eV) used for overlapped spots
      metal_foil_correction
        .help = Use if the detector is partially obscured by a metal foil designed to \
                absorb one of the energies completely and the other partially
      {
        absorption_edge_energy = None
          .type = float
          .help = Reflections whose energy is higher than this energy are discarded. \
                  Reflections whose energy is lower than this energy are attenuated
        transmittance = None
          .type = float
          .help = Fractional transmittance for the partially absorbed energy, assuming \
                  normal incidence with respect to the detector
        two_theta_deg = None
          .type = float
          .help = Reflections with two theta angles less than this value (in degrees) \
                  will be corrected.
      }
    }
  }
  calc_G_and_B {
    do_calc = False
      .type = bool
      .help = Can choose to postrefine G and B factors for a still
    include scope xfel.command_line.cxi_merge.master_phil
  }
''', process_includes=True)

phil_scope.adopt_scope(two_color_phil_scope)

from xfel.command_line.xfel_process import Script as StillsProcessScript
class InMemScript(StillsProcessScript):
  """ Script to process two color stills """
  def __init__(self):
    """ Set up the option parser. Arguments come from the command line or a phil file """
    self.usage = \
    """ %s input.data=filename
    """%libtbx.env.dispatcher_name

    self.parser = OptionParser(
      usage = self.usage,
      phil = phil_scope,
      #epilog=help_message,
      read_datablocks=True,
      read_datablocks_from_images=True)

  def index(self, datablock, reflections):
    from time import time
    from logging import info
    import copy
    st = time()

    info('*' * 80)
    info('Indexing Strong Spots')
    info('*' * 80)

    imagesets = datablock.extract_imagesets()

    params = copy.deepcopy(self.params)
    # don't do scan-varying refinement during indexing
    params.refinement.parameterisation.crystal.scan_varying = False
    from cxi_xdr_xes.two_color.two_color_indexer import indexer_two_color
    idxr = indexer_two_color(reflections, imagesets, params=params)
    indexed = idxr.refined_reflections
    experiments = idxr.refined_experiments

    if self.params.output.indexed_filename:
      self.save_reflections(indexed, self.params.output.indexed_filename)

    info('')
    info('Time Taken = %f seconds' % (time() - st))
    return experiments, indexed

  def integrate(self, experiments, indexed):
    from logging import info

    info('*' * 80)
    info('Integrating Reflections')
    info('*' * 80)

    from xfel.command_line.xfel_process import Script as ProcessScript
    assert len(experiments) == 2
    integrated = ProcessScript.integrate(self, experiments, indexed)
    if 'intensity.prf.value' in integrated:
      method = 'prf' # integration by profile fitting
    elif 'intensity.sum.value' in integrated:
      method = 'sum' # integration by simple summation
    integrated = integrated.select(integrated['intensity.' + method + '.variance'] > 0) # keep only spots with sigmas above zero
    integrated = remove_invalid_reflections(integrated)
    self.save_reflections(integrated, self.params.output.integrated_filename)

    def write_integration_pickles_callback(params, outfile, frame):
      from cxi_xdr_xes.two_color.two_color_dump import correction_for_metal_foil_absorption, derive_scale_and_B_to_model
      correction_for_metal_foil_absorption(params, frame)
      if params.calc_G_and_B.do_calc:
        derive_scale_and_B_to_model(params.calc_G_and_B, outfile, frame)

    self.write_integration_pickles(integrated, experiments, callback = write_integration_pickles_callback)
    from dials.algorithms.indexing.stills_indexer import calc_2D_rmsd_and_displacements
    rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
    rmsd_integrated, _ = calc_2D_rmsd_and_displacements(integrated)
    crystal_model = experiments.crystals()[0]
    print "Integrated. RMSD indexed,", rmsd_indexed, "RMSD integrated", rmsd_integrated, \
      "Final ML model: domain size angstroms: %f, half mosaicity degrees: %f"%(crystal_model._ML_domain_size_ang, crystal_model._ML_half_mosaicity_deg)

    return integrated

if __name__ == "__main__":
  from dials.util import halraiser
  try:
    script = InMemScript()
    script.run()
  except Exception as e:
    halraiser(e)
