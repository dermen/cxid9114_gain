import numpy as np
import pandas, cctbx, scitbx
from dials.array_family import flex

from cctbx import sgtbx, crystal
# load the data!
print "loading data"
df = pandas.read_hdf( "big_refls_feb14th.h5","df")
shuff = False
                
sg = sgtbx.space_group(" P 4nw 2abw")
Symm = crystal.symmetry( unit_cell=(79,79,38,90,90,90), space_group=sg)

print "querying"
df = df.query("BnotA")  
df = df.query("intens2 < 5000")

print "hkl"
hkls = tuple( map( tuple, df[['h','k','l']].values))

intens = np.ascontiguousarray(df.intens2.values)
if shuff:
    print("shuffling")
    np.random.shuffle( intens)
data = flex.double(intens)
sigmas = flex.double( np.sqrt(intens))

mil_idx = flex.miller_index(hkls)
mill_set = cctbx.miller.set( crystal_symmetry=Symm, 
                indices=mil_idx, anomalous_flag=True)
mill_ar = cctbx.miller.array(mill_set, data=data, sigmas=sigmas)\
            .set_observation_type_xray_intensity()

from IPython import embed
embed()
merged = mill_ar.merge_equivalents().array() #.sort(by_value="data")
merged.show_comprehensive_summary()

