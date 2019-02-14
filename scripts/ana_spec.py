# coding: utf-8
import pandas
from cxid9114 import spectrum
import psana
import sys
run = int(sys.argv[1])
ds = psana.DataSource("exp=cxid9114:run=%d:idx" % run)
spec = psana.Detector("FeeSpec-bin")
R = ds.runs().next()
times = R.times()

df = pandas.read_pickle("ana_result/run%d/run%d_overview.pdpkl" % (run,run))

N = len(df)
all_spec_hist = []
all_rawspec = []
has_spec = []
for i_df in range( N):
    print i_df,N     
    idx = df.shot_idx[i_df]
    
    ev = R.event (times[idx])
    if ev is None:
        all_spec_hist.append( None)
        all_rawspec.append(None)
        has_spec.append(False)
        continue

    spec_img = spec.image( ev)
    if spec_img is None:
        all_spec_hist.append( None)
        all_rawspec.append(None)
        has_spec.append(False)
        continue

    has_spec.append(True)
    out = spectrum.get_spectrum(spec_img)
   
    spec_hist = out[-3]
    raw_spec = out[-1].data
    all_spec_hist.append( list(spec_hist))
    all_rawspec.append( list(raw_spec))


df['spec_hist'] = all_spec_hist
df['raw_spec'] = all_rawspec
df['has_spec'] = has_spec
df.to_pickle("ana_result/run%d/run%d_overview_wspec.pdpkl" % (run,run))

