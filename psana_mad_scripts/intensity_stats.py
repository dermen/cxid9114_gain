import numpy as np
import pandas, cctbx, scitbx
from dials.array_family import flex
from cctbx import sgtbx, crystal

# load the data!
print "loading data"
data_f = "/reg/d/psdm/cxi/cxid9114/res/dermen/reflection_2colorspec.hdf5"
df = pandas.read_hdf( data_f,"reflections")
                
sg = sgtbx.space_group(" P 4nw 2abw")
Symm = crystal.symmetry( unit_cell=(79,79,38,90,90,90), space_group=sg)

print "querying"
df = df.query("BnotA")  
#df = df.query("intens2 < 5000")

print "hkl"
hkls = tuple( map( tuple, df[['hB','kB','lB']].values))

intens = np.ascontiguousarray(df.intens5.values)
data = flex.double(intens)
sigmas = flex.double( np.sqrt(intens))

mil_idx = flex.miller_index(hkls)


mill_set = cctbx.miller.set( crystal_symmetry=Symm, 
                indices=mil_idx, anomalous_flag=True)
mill_ar = cctbx.miller.array(mill_set, data=data, sigmas=sigmas)\
            .set_observation_type_xray_intensity()

merged = mill_ar.merge_equivalents().array() #.sort(by_value="data")
merged.show_comprehensive_summary()

