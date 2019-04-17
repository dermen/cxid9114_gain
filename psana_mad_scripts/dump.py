# coding: utf-8
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
df.intens5.max()
df.intens5.min()
#df.intens5.min()
#F
sigmas = flex.double( np.sqrt(intens))

mil_idx = flex.miller_index(hkls)


mill_set = cctbx.miller.set( crystal_symmetry=Symm,
                indices=mil_idx, anomalous_flag=True)
mill_ar = cctbx.miller.array(mill_set, data=data, sigmas=sigmas)            .set_observation_type_xray_intensity()
df.resB
df_r = df.query("4.2 < resB < 4.7")
df_r
df_r.BnotA
np.all(df_r.BnotA)
df_r.BnotA
df_r.intens5
np.sqrt(df_r.intens5)
np.sqrt(df_r.intens5).hist(bins=logspace(1,5,100), log=1)
get_ipython().magic(u'pylab')
np.sqrt(df_r.intens5).hist(bins=logspace(1,5,100), log=1)
np.sqrt(df_r.intens5).hist(bins=100, log=1)
df_r.intens5 /= df_r.partiality
df_r.partiality
#df_r = df.query("4.2 < resB < 4.7")
list(df_r)
df.HS_ratio
df_r = df.query("4.2 < resB < 4.7").query("HS_ration < 0.9")
df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9")
df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9").partiality
df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9").partiality
df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9").partiality.min()
df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9").partiality.max()
df.partiality.max()
df
df.partiality.max()
get_ipython().magic(u'ls ')
get_ipython().magic(u'pwd ')
df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9")
df_r.intens5/ df_r.partialityB
df_r.intens5/ df_r.partialityB / df_r.transmission
df_r.transmission
df_r.transmission <0.1
np.sum(df_r.transmission <0.1)
df_r
len( df_r)
np.sum(df_r.transmission < 0.1)
np.sum(df_r.transmission < 0.03)
np.sum(df_r.transmission < 0.1)
bad = df_r.query('transmission < 0.1')
bad
bad.run
bad.transmission
bad.thick
bad.Si_thicknes
bad.run
bad.shot_idx
bad.run
import psana
ds = psana.DataSource("exp=cxid9114:run=120")
bad.shot_idx
ds = psana.DataSource("exp=cxid9114:run=120:idx")
run = ds.runs().next()
times = run.times()
t = times[60909]
ev = run.event(t)
ev
det_ids = range(2,12)  # need reference to the motors themselves
atten_dets = { det_id:psana.Detector("XRT:DIA:MMS:%02d.RBV" % det_id, ds.env())
            for det_id in det_ids}
# each motor represents a piece of Silicon foil, of varying thickness
atten_vals = {det_id: 20*2**i  for i, det_id in enumerate(det_ids)}
for i in atten_dets:
    print i, atten_vals[i], atten_dets[i].get(ev)
    
for i in atten_dets:
    print i, atten_vals[i], atten_dets[i](ev)
    
bad.intens5 / bad.partiality
bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)
bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens).hist(bins=100,log=1)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=100,log=1)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=100,log=1)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=300,log=1)
#(df_r.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=300,log=1)
bad = df_r.query('transmission < 0.1')
good = df_r.query('transmission >.1')
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=300,log=1)
(good.intens5 / good.partiality/(good.channA_intens+good.channB_intens)).hist(bins=300,log=1)
(good.intens5 / good.partiality/(good.channA_intens+good.channB_intens)).hist(bins=300,log=1, normed=True)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=300,log=1, normed=True)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=(0,100,100),log=1, normed=True)
(bad.intens5 / bad.partiality/(bad.channA_intens+bad.channB_intens)).hist(bins=linspace(0,100,100),log=1, normed=True)
(good.intens5 / good.partiality/(good.channA_intens+good.channB_intens)).hist(bins=linspace(0,100,100),log=1, normed=True)
df.transmission
bad.Si_thicknes
for i in atten_dets:
    print i, atten_vals[i], atten_dets[i](ev)
    
atten_dets
ds = psana.DataSource("exp=cxid9114:run=120:smd")
#ax.tick_params(labelsize=17, length=9, which='major')
det_ids = range(2,12)  # need reference to the motors themselves
atten_dets = { det_id:psana.Detector("XRT:DIA:MMS:%02d.RBV" % det_id, PSANA_ENV)
            for det_id in det_ids}
# each motor represents a piece of Silicon foil, of varying thickness
atten_vals = {det_id: 20*2**i  for i, det_id in enumerate(det_ids)}
det_ids = range(2,12)  # need reference to the motors themselves
atten_dets = { det_id:psana.Detector("XRT:DIA:MMS:%02d.RBV" % det_id, ds.env())
            for det_id in det_ids}


# each motor represents a piece of Silicon foil, of varying thickness
atten_vals = {det_id: 20*2**i  for i, det_id in enumerate(det_ids)}
events = ds.events()
vals = []
for ev in events:
    if ev is None:
        continue
    si = atten_dets[det_ids[11]](ev)
    if si is None:
        continue
    vals.append(si)
    print si
    
vals = []
for ev in events:
    if ev is None:
        continue
    si = atten_dets[11](ev)
    if si is None:
        continue
    vals.append(si)
    print si
    
vals = []
for i,ev in enumerate(events):
    if ev is None:
        continue
    si = atten_dets[11](ev)
    if si is None:
        continue
    vals.append(si)
    print i, si
    
len( vals)
events = ds.events()
len( vals)
vals = []
for i,ev in enumerate(events):
    if ev is None:
        continue
    si = atten_dets[11](ev)
    if si is None:
        continue
    vals.append(si)
    print i, si
    
ds = psana.DataSource("exp=cxid9114:run=120:smd")
events = ds.events()
vals = []
for i,ev in enumerate(events):
    if ev is None:
        continue
    si = atten_dets[11](ev)
    if si is None:
        continue
    vals.append(si)
    print i, si
    
df_r.bkgrnd5
#df_r.bkgrnd5
R =df_r.groupby("run")
runs = df_r.runs.unique()
runs = df_r.run.unique()
runs
R.get_group( 63)
R.get_group( 63).bkgrnd
R.get_group( 63).bkgrnd5
#R.get_group( 63).bkgrnd5.hist(bins=0,
df_r.bkgrnd5.max()
df_r.bkgrnd5.min()
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,200,100))
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,200,100))
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,10,20))
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,10,50))
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,10,50), normed=True)
R.get_group( 120).bkgrnd5.hist(bins=linspace(0,10,50), normed=True)
#df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9")
#df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9")
runs
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,10,50), normed=True)
#R.get_group( 60).bkgrnd5.hist(bins=linspace(0,10,50), normed=True)
df2 = df.query("3.2 < resB < 5.7").query("HS_ratio < 0.9")
R = df2.groupby("run")
runs = df2.run.unique()
runs
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,100,100), normed=True)
clf()
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,100,100), normed=True, log=1)
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,30,300), normed=True, log=1)
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
clf()
R.get_group( 60).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 62).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 120).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 61).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 63).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
runs
R.get_group( 131).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 171).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group( 166).bkgrnd5.hist(bins=linspace(0,30,200), normed=True, log=1, histtype='step')
R.get_group(166)
R.get_group(166).shot_idx
R.get_group(166).shot_idx.unique()
R.get_group(166).query("shot_idx==31831")
runs
R = df_r.groupby("run")
df_r = df.query("4.2 < resB < 4.7").query("HS_ratio < 0.9")
R = df_r.groupby("run")
#R[383]['shoebox']
#R.get_group(166)
runs = df_r.run.unique()
runs
R.get_group( 63).intens5.hist(bins=linspace(0,30,1000), normed=True, log=1, histtype='step')
R.get_group( 63).intens5.hist(bins=linspace(0,1000,1000), normed=True, log=1, histtype='step')
R.get_group( 63).intens5.hist(bins=linspace(0,200,1000), normed=True, log=1, histtype='step')
R.get_group( 63).intens5.hist(bins=linspace(0,200,1000), normed=False, log=1, histtype='step')
clf()
R.get_group( 63).intens5.hist(bins=linspace(0,200,1000), normed=False, log=1, histtype='step')
R.get_group( 63).intens5.hist(bins=linspace(0,100,1000), normed=False, log=1, histtype='step')
df_r.intens5.max()
df_r.intens5.min()
df_r.intens5.median()
R.get_group( 63).intens5.hist(bins=linspace(0,80,100), normed=False, log=1, histtype='step')
R.get_group( 120).intens5.hist(bins=linspace(0,80,100), normed=False, log=1, histtype='step')
runs
R.get_group( 128).intens5.hist(bins=linspace(0,80,100), normed=False, log=1, histtype='step')
R.get_group( 66).intens5.hist(bins=linspace(0,80,100), normed=False, log=1, histtype='step')
#R.get_group( 66).intens5.hist(bins=linspace(0,80,100), normed=False, log=1, histtype='step')
df_r / df_r.partiality
df_r.intens5 / df_r.partiality
np.sqrt(  df_r.intens5 / df_r.partiality ) 
np.hist(np.sqrt(  df_r.intens5 / df_r.partiality ) )
(np.sqrt(  df_r.intens5 / df_r.partiality ) ).hist(bins=100)
df
df.BnotA
np.all(df.BnotA)
np.all(df.AandB)
np.aany(df.AandB)
np.any(df.AandB)
#df.inten5/ df.partiality
df = df.query("HS_ratio < 0.9")
df
df = df.query("HS_ratio < 0.9")
df
df.intens5 / df.partiality
df.intens5 / df.partialitB
df.intens5 / df.partialityB
df.intens5 / df.partiality
#df.intens5 / df.partiality
df.channB_intens.max()
df.channA_intens + df.channB_intens
(df.channA_intens + df.channB_intens).max()
mx_intens = (df.channA_intens + df.channB_intens).max()
df.intens5 / df.partiality * mx_intens / (df.channA_intens+df.channB_intens)
lvl = df.intens5 / df.partiality * mx_intens / (df.channA_intens+df.channB_intens)
sqrt( lvl).hist(bins=1000)
sqrt( lvl).hist(bins=1000, log=1)
clf();sqrt( lvl).hist(bins=1000, log=1, histtype='step')
#df.intens
sqrt( df.intens5 ).hist(bins=1000, log=1, histtype='step')
mx_intens
mx_intens/(df.channA_intens+df.channB_intens)
(mx_intens/(df.channA_intens+df.channB_intens)).max()
(mx_intens/(df.channA_intens+df.channB_intens)).max()
(mx_intens/(df.channA_intens+df.channB_intens))
np.unique(mx_intens/(df.channA_intens+df.channB_intens))
np.unique(mx_intens/(df.channA_intens+df.channB_intens))[::-10]
np.unique(mx_intens/(df.channA_intens+df.channB_intens))[::-1][:10]
np.unique(mx_intens/(df.channA_intens+df.channB_intens))[::-1][:100]
gb = df.groupby(["hB","kB","lB"])
gb.groups.values[0]
gb.groups.values()[0]
count = [ v.shape[0] for v in gb.groups.values()]
count
order = np.argsort(count)[::-1]
order
#order = np.argsort(count)[::-1]
H = gb.groups.keys()
H[0]
H[order[0]]
df
gb.get_group( H[order[0]])
gb.get_group( H[order[0]]).run.unique()
gb.get_group( H[order[0]]).run
gb.get_group( H[order[0]]).run
gb.get_group( H[order[10]]).run
gb.get_group( H[order[10]]).shape
gb.get_group( H[order[2]])
plot( gb.get_group( H[order[2]]), bins=100)
plot( gb.get_group( H[order[2]]).intens5, '.')
clf()
plot( gb.get_group( H[order[2]]).intens5, '.')
#gb.get_group( H[order[2]])
def norm(d,intens):
    return d[intens]/d.partiality
norm(gb.get_group( H[order[2]]), "intens5")
#norm(gb.get_group( H[order[2]]), "intens5").hist(bins
plot( norm(gb.get_group( H[order[2]]), "intens5"), '.')
plot( norm(gb.get_group( H[order[2]]), "intens5"), '.')
plot( norm(gb.get_group( H[order[2]]), "intens5").values, '.')
plot( norm(gb.get_group( H[order[2]]), "intens3").values, '.')
plot( norm(gb.get_group( H[order[2]]), "intens2").values, '.')
plot( norm(gb.get_group( H[order[2]]), "intens").values, '.')
plot( norm(gb.get_group( H[order[2]]), "intens6").values, '.')
def norm(d,intens):
    return np.sqrt(d[intens]/d.partiality)
plot( norm(gb.get_group( H[order[2]]), "intens6").values, '.')
clf()
plot( norm(gb.get_group( H[order[2]]), "intens6").values, '.')
dfA = pandas.read_hdf("reflection_2colorspec.hdf5", "reflections")
dfA = pandas.read_hdf("../../../../res/dermen/reflection_2colorspec.hdf5", "reflections")
H[order[2]]
gbB = df.groupby(["hB","kB","lB"])
#gbA = df.groupby(["hA","kA","lA"])
dfA = df.query("HS_ratio < 0.9")
dfA = dfA.query("HS_ratio < 0.9")
gbA = dfA.groupby(["hA","kA","lA"])
H[order[2]]
H[order[2]]
gbB.get_group( (3,-25,0))
gbA.get_group( (3,-25,0))
gbA.get_group( (-3,25,0))
gbB.get_group( (-3,25,0))
gbA.get_group( (-3,25,0))
gbB.get_group( (-3,25,0))
#gbA.get_group( (-3,25,0))
#dfA = dfA.query("HS_ratio < 0.9")
dfA = dfA.query("HS_ratio < 0.9").query("AnotB")
gbA = dfA.groupby(["hA","kA","lA"])
gbA.get_group( (-3,25,0))
gbA.get_group( (3,-25,0))
dfA = pandas.read_hdf("../../../../res/dermen/reflection_2colorspec.hdf5", "reflections")
dfA.query("AnotB")
#bA = dfA.query("AnotB").query()
df_mast = pandas.read_hdf("../../../../res/dermen/reflection_2colorspec.hdf5", "reflections")
dfA = df.query("HS_ratio < 0.9").query("AnotB")
dfB = df.query("HS_ratio < 0.9").query("AnotB")
dfA
dfB = df_mast.query("HS_ratio < 0.9").query("AnotB")
dfB = df_mast.query("HS_ratio < 0.9").query("BnotA")
dfA = df_mast.query("HS_ratio < 0.9").query("AnotB")
dfA
dfB
gbA = dfA.groupby(["hA","kA","lA"])
gbB = df.groupby(["hB","kB","lB"])
gbB = dfB.groupby(["hB","kB","lB"])
countA = [ v.shape[0] for v in gbA.groups.values()]
#countB = [ v.shape[0] for v in gbB.groups.values()]
countB = [ v.shape[0] for v in gbB.groups.values()]
orderA = np.argsort(countA)[::-1]
orderB = np.argsort(countB)[::-1]
HB = gbB.groups.keys()
HA = gbA.groups.keys()
HA[orderA[0]]
HB[orderB[:10]]
HA = np.array( HA)
HB = np.array( H)
HB = np.array( HB)
HB[orderB[:10]]
HA[orderA[:10]]
countsA[orderA[:10]]
countA[orderA[:10]]
countA = np.array( countA)
countB = np.array( countB)
countA[orderA[:10]]
countA[orderA[:20]]
#countA[orderA[:]]
countA > 10
where(countA > 10)
goodA = HA[where(countA > 10)[0]]
goodB = HB[where(countB > 10)[0]]
goodA
goodA
goodA.shape
goodB.shape
#[i for i in goodA if 
#tre
from scipy.spatial import cKDTree
treeB = cKDTree( goodB)
treeB.query(goodA[0])
#treeB.query(goodA[0])
get_ipython().magic(u'timeit treeB.query(goodA[0])')
goodA_inB[i for i in goodA if treeB.query(i)[0]==0]
goodA_inB = [i for i in goodA if treeB.query(i)[0]==0]
goodA_inB
goodA_inB = [tuple(i) for i in goodA if treeB.query(i)[0]==0]
goodA_inB
len(goodA_inB)
gbB.get_group( goodA_inB[0])
gbA.get_group( goodA_inB[0])
gbB.get_group( goodA_inB[0])
gbB.get_group(goodA_inB[0])
gbA.get_group(goodA_inB[0])
asu = mill_ar.map_to_asu()
perm = asu.sort_permutation(by="packed_indices")
perm = asu.sort_permutation(by_value="packed_indices")
perm
perm[0]
perm[1]
perm[2]
perm[3]
perm[4]
asu.indices()
asu.indices().select(perm)
#asu.indices().select(perm)
get_ipython().magic(u'paste')
data_type_str = mill_arr.data().__class__.__name__
data_type_str = mil_arr.data().__class__.__name__
mill_ar
data_type_str = mil_ar.data().__class__.__name__
data_type_str = mill_ar.data().__class__.__name__
data_type_str
asu_set = cctbx.miller.set(mill_ar)
asu_set = cctbx.miller.set.map_to_asu(mill_ar)
asu_set
perm = asu_set.sort_permutation(by_value="packed_indices")
asu_set.indices().select(perm)
asu_set.indices()
asu_set.indices
asu_set.indices()
asu_set
get_ipython().magic(u'pinfo asu_set.match_indices')
dir(asu_set)
#cctbx.miller.sym_equiv_indices(
goodA_inB[0]
cctbx.miller.sym_equiv_indices(sg, goodA_inB[0])
cctbx.miller.sym_equiv_indices(sg, goodA_inB[0]).indices()
get_ipython().magic(u'pinfo cctbx.miller.sym_equiv_indices')
cctbx.miller.sym_equiv_indices(sg, goodA_inB[1]).indices()
cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()
cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()[0]
cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()[0][0]
equivs = cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()
e = equivs[0]
e.h
e.h()
[ h.h() for h in cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()]
[ h.h() for h in cctbx.miller.sym_equiv_indices(sg, goodA_inB[0]).indices()]
goodA_inB[0]
goodA_inB[0]
gbA.get_group((-11,17,12))
gbA.get_group((11,-17,-12))
gbA.get_group((-11,17,12))
gbA.get_group((11,-17,-12))
gbA.get_group((-11,17,12))
gbB.get_group((11,-17,-12))
gbB.get_group((-11,17,12))
#gbB.get_group((-11,17,12))
#gbB.get_group((-11,17,12))
[ h.h() for h in cctbx.miller.sym_equiv_indices(sg, goodA_inB[2]).indices()]
[ h.h() for h in cctbx.miller.sym_equiv_indices(sg, (16,-9,6)).indices()]
a1 = [ h.h() for h in cctbx.miller.sym_equiv_indices(sg, (16,-9,6)).indices()]
a2 = [ h.h() for h in cctbx.miller.sym_equiv_indices(sg, (-16,9,-6)).indices()]
a1
a2
set( a1+a2)
a2
treeB.query(a2)
treeB.query(a1)
treeA = cKDTree( goodA)
treeA.query(a1)
treeA.query(a2)
goodA = HA[where(countA > 0)[0]]
#treeB = cKDTree( goodB)
goodB = HB[where(countB > 0)[0]]
treeA = cKDTree( goodA)
treeB = cKDTree( goodB)
goodA.keys()
good
goodA
goodA.shape
set(goodA)
set(list(map,tuple,goodA)))
set(list(map,tuple,goodA))
set(list(map(tuple,goodA) ))
len(set(list(map(tuple,goodA) )))
goodA[0]
cctbx.miller.sym_equiv_indices(sg, goodA[0])
#cctbx.miller.sym_equiv_indices(sg, goodA[0])
a1 = [ h.h() for h in cctbx.miller.sym_equiv_indices(sg, goodA[0]).indices()]
treeA = cKDTree( goodA)
treeA.query(a1)
#treeA.query(a1)
a1 = [treeA.query([ h.h() for h in cctbx.miller.sym_equiv_indices(sg, g).indices()]) for g in goodA]
a1[0]
for a,b in a1:
    print a
    
for a,b in a1:
    idx = b[ a==0]
    
    
for a,b in a1:
    idx = b[ a==0]
    
    print idx
    
for a,b in a1:
    idx = b[ a==0]
    
    print idx, 
    
    
goodA
#gbA.get_group()
goodA
gbA.get_group(goodA[0])
gbA.get_group(tuple(goodA[0]))
gbA.get_group(tuple(goodA[0]))
get_ipython().magic(u'timeit gbA.get_group(tuple(goodA[0]))')
get_ipython().magic(u'timeit gbA.get_group(tuple(goodA[10]))')
get_ipython().magic(u'timeit gbA.get_group(tuple(goodA[100]))')
goodA[10]
idx
godoA[i] for i in idx]
[godoA[i] for i in idx]
[goodA[i] for i in idx]
[tuple(goodA[i]) for i in idx]
[gb.get_group(tuple(goodA[i])) for i in idx]
get_ipython().magic(u'timeit [gb.get_group(tuple(goodA[i])) for i in idx]')
len( a1)
0.00292 * 89024
pandas.concat([gb.get_group(tuple(goodA[i])) for i in idx])
#pandas.concat([gb.get_group(tuple(goodA[i])) for i in idx])
goodA_idx = []
for a,b in a1:
    idx = b[ a==0]
    
    print idx,
    goodA_idx.append( idx)
     
    
    
goodA_idx
pandas.concat([gb.get_group(tuple(goodA[i])) for i in idx])
pandas.concat([gb.get_group(tuple(goodA[i])) for i in goodA_idx[0]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[0]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[1]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[2]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[3]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[4]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[5]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[6]])
pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[10]])
#pandas.concat([gbA.get_group(tuple(goodA[i])) for i in goodA_idx[10]])
#countsA[orderA[:10]]
#countA = [ v.shape[0] for v in gbA.groups.values()]
#gbA = 
out = [pandas.concat([gbA.get_group(tuple(goodA[i])) for i in idx]) for idx in goodA_idx]
N = [len(o) for o in out]
N
argmax(N)
N[1545]
#N = [len(o) for o in out]
out[1545]
d = out[1545]
plot(d.intensB.values[0],'.')
clf()
plot(d.intensB.values[0],'.')
plot(d.intens2.values[0],'.')
d
plot(d.intens2.values,'.')
plot(sqrt(d.intens2.values),'.')
plot(sqrt(d.intens2.values / d.partiality.values),'.')
plot(sqrt(d.intens2.values),'.')
plot(sqrt(d.intens2.values / d.partiality.values / d.channA_intens),'.')
plot(sqrt(d.intens2.values / d.partiality.values / d.channA_intens.values),'.')
plot(sqrt(d.intens2.values / d.partiality.values / d.channA_intens.values),'.')
figure()
plot(sqrt(d.intens2.values),'.')
o
out
#plot(sqrt(d.intens2.values / d.partiality.values / d.channA_intens.values),'.')
order = argsort(N)[::-1]
order
d = out[11922]
d
order
d = out[51021]
d
d = out[order[2]]
d
d = out[order[4]]
d
len(d)
order
out[0]
out[1]
out[2]
N[order]
array(N)[order]
plot(sqrt(d.intens2.values),'.')
ylabel("$\sqrt{I}$", fontsize=24)
ylabel("$\sqrt{I}$", fontsize=20)
xlabel("measurement")
ax = gca()
ax.tick_params(labelsize=17)
xlabel("measurement", fontsize=16)
np.mean( d.intens2.values)
np.sqrt(np.mean( d.intens2.values))
get_ipython().magic(u'save d.py 1-520')
