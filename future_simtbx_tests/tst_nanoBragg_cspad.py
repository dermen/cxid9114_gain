from __future__ import division
import numpy as np

import simtbx.nanoBragg
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import DetectorFactory


import dxtbx_cspad

# flat CSPAD
cspad = DetectorFactory.from_dict(dxtbx_cspad.cspad)  # loads a 64 panel dxtbx cspad
# beam along Z
beam = BeamFactory.simple(1.3)  #  make a simple beam along z

print "# --- beam centers comparison for canonical setup -- #"
net_error_flat = 0
for pid in range(64):
    print pid
    SIM = simtbx.nanoBragg.nanoBragg(detector=cspad, beam=beam, verbose=0, panel_id=pid)
    b1 = SIM.beam_center_mm
    b2 = cspad[pid].get_beam_centre(beam.get_s0())
    print np.round(b1,2)
    print np.round(b2,2)
    print

    net_error_flat +=  np.sum(np.subtract(b1,b2)**2)

# tilted cspad (components of the cspad panels along the beam axis)
# easily accomplish this with mis-aligned beam

# this is a realistic tilted beam
# birthed by dials.refine
beam_descr = {
    'direction': (7.010833160725592e-06, -3.710515413340211e-06, 0.9999999999685403),
    'divergence': 0.0,
    'flux': 0.0,
    'polarization_fraction': 0.999,
    'polarization_normal': (0.0, 1.0, 0.0),
    'sigma_divergence': 0.0,
    'transmission': 1.0,
    'wavelength': 1.385}
tilted_beam = BeamFactory.from_dict(beam_descr)

net_error_tilt = 0
print "# --- beam centers comparison for tilted setup -- #"
for pid in range(64):
    print pid
    SIM = simtbx.nanoBragg.nanoBragg(detector=cspad, beam=tilted_beam, verbose=0, panel_id=pid)

    b1 = SIM.beam_center_mm
    b2 = cspad[pid].get_beam_centre(beam.get_s0())
    print np.round(b1,2)
    print np.round(b2,2)
    print
    net_error_tilt +=  np.sum(np.subtract(b1,b2)**2)

# Now add to the mix a CSPAD whose 64
# fdet and sdet vectors each have some
# small variable components along the beam dir

cspad_mess = DetectorFactory.from_dict(dxtbx_cspad.distorted_cspad)

net_error_messy = 0
print "# --- beam centers comparison for distorted cspad -- #"
for pid in range(64):
    print pid
    SIM = simtbx.nanoBragg.nanoBragg(detector=cspad_mess, beam=tilted_beam, verbose=0, panel_id=pid)

    b1 = SIM.beam_center_mm
    b2 = cspad_mess[pid].get_beam_centre(beam.get_s0())
    print np.round(b1,2)
    print np.round(b2,2)
    print

    net_error_messy +=  np.sum(np.subtract(b1,b2)**2)

print("Net error for canonical setup: %.4f" % net_error_flat)
print("Net error for tilted setup: %.4f" % net_error_tilt)
print("Net error for messy setup: %.4f" % net_error_messy)

assert( np.allclose(net_error_flat, 0))
assert( np.allclose(net_error_tilt, 0))
assert( np.allclose(net_error_messy, 0))

if __name__=="__main__":
 print "OK"

