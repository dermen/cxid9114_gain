
import numpy as np
import h5py
import sys

fname = sys.argv[1]
f = h5py.File(fname, "r+")
Efit = np.array([1.45585875e-01, 8.92420032e+03])
proc_spec_name = "hist_spec"
energy_name = "energy_bins"
ave_flux = 2e11
if proc_spec_name in f:
    del f[proc_spec_name]
if energy_name in f:
    del f[energy_name]

raw_specs = f['raw_spec']
Nspec_bins = raw_specs.shape[1] # 1024
Nspecs = raw_specs.shape[0]
ev_width = 1
Edata = np.polyval( Efit, np.arange( Nspec_bins))
en_bins = np.arange( Edata[0],Edata[-1]+1, ev_width)
en_bin_cent = 0.5*en_bins[1:] + 0.5*en_bins[:-1]
spec_hists = np.zeros((Nspecs, en_bin_cent.shape[0]))
for i_spec, spec in enumerate(raw_specs):
    if i_spec %50==0:
        print ('processing spec %d / %d') % (i_spec+1, Nspecs)
    spec_hists[i_spec] = np.histogram(Edata, en_bins, weights=spec)[0]

# normalize the shots
K = ave_flux/(spec_hists.sum(axis=1).mean())

f.create_dataset(proc_spec_name, data=spec_hists*K, compression="lzf")
f.create_dataset(energy_name, data=en_bin_cent, compression="lzf")

