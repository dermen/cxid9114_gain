from __future__ import division
from libtbx import test_utils
import libtbx.load_env

tst_list = (
  "$D/two_color/tst_indexed_hkl.py",
  "$D/two_color/tst_close_spot_res.py",
  "$D/two_color/tst_hkl_unique.py",
  "$D/two_color/tst_angle.py",
  "$D/two_color/two_color_test.py",
  )

def run_standalones():
  build_dir = libtbx.env.under_build("cxi_xdr_xes")
  dist_dir = libtbx.env.dist_path("cxi_xdr_xes")
  test_utils.run_tests(build_dir, dist_dir, tst_list)

if (__name__ == "__main__"):
  run_standalones()