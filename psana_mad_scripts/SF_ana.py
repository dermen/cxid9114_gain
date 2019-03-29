# coding: utf-8
from cxid9114 import utils
a=utils.open_flex('SA.pkl')
b=utils.open_flex('SB.pkl')
a
dir(a)
a.show_array()
b.show_array()
ba = b.show_array()
ba
b.show_array()
a.amplitudes()
a.amplitudes()[0]
dir(a)
get_ipython().magic(u'pinfo a.value_at_index')
a.value_at_index((1,9,8))
a.indices()
a.indices()[0]
H = [a.indices()[i] for i in len(a)]
H = [a.indices()[i] for i in len(a.indices())]
H = [i for i in a.indices()]
H
a.value_at_index(H[10])
b.value_at_index(H[10])
H[10]
a.value_at_index(H[1000])
b.value_at_index(H[1000])
a.resolution_range
get_ipython().magic(u'pinfo a.resolution_range')
def res_from_idx:
def res_from_idx:
def res_from_idx(h,k,l,a=79.,b=79.,c=38.):
    return 1./ np.sqrt(h*h/a/a+k*k/b/b+l*l/c/c)
res_from_idx(*H[0])
import numpy as np
res_from_idx(*H[0])
res_from_idx(*H[1])
res_from_idx(*H[2])
res_from_idx(*H[3])
res_from_idx(*H[300])
H[300]
H[2]
H[1]
H[0]
def res_from_idx(h,k,l,a=79.,b=79.,c=38.):
    return 1./ np.sqrt( h*h/a/a + k*k/b/b + l*l/c/c)
RES = [res_from_idx(*h) for h in H]
RES
a.resolution_range()
get_ipython().magic(u'paste')
get_ipython().magic(u'pinfo np.digitize')
bin_ass = np.digitize( RES,res_lims)
bin_ass = np.digitize( RES,res_lims)-1
bin_ass
Aval = [a.value_at_index(h) for h in H]
Bval = [b.value_at_index(h) for h in H]
Aval = np.array(Aval)
Bval = np.array(Bval)
Aval**2 - Bval**2
np.sqrt(np.abs(Aval**2 - Bval**2))
np.sqrt(np.abs(Aval**2 - Bval**2))
np.sqrt(np.abs(Aval**2 - Bval**2))
resdiff = np.sqrt(np.abs(Aval**2 - Bval**2))
get_ipython().magic(u'pylab')
#np.histogram( bin_ass, bins=arange
bincount (bin_ass)
bincount( bin_ass, weights=resdiff)
bincount( bin_ass, weights=resdiff) / bincount(bin_ass)
#bar( RES[:-1]*.5+RES, bincount( bin_ass, weights=resdiff) / bincount(bin_ass) )
rescent = RES[:-1]*.5 + RES[1:]*.5
res_cent = array(RES[:-1])*.5 + array(RES[1:])*.5
res_cent
RES
RES[0]
get_ipython().magic(u'paste')
res_lims[:-1]*.5
res_lims[:-1]*.5 + res_lims[1:]*.5
res_x = res_lims[:-1]*.5 + res_lims[1:]*.5
res_x[0] = 50
bar( range( len(res_x)), bincount( bin_ass, weights=resdiff) / bincount(bin_ass), width))
bar( range( len(res_x)), bincount( bin_ass, weights=resdiff) / bincount(bin_ass), width)
bar( range( len(res_x)), bincount( bin_ass, weights=resdiff) / bincount(bin_ass), width=1)
bincount( bin_ass, weights=resdiff) / bincount(bin_ass))
bincount( bin_ass, weights=resdiff) / bincount(bin_ass)
bincount( bin_ass, weights=resdiff) / bincount(bin_ass).shape
bincount( bin_ass, weights=resdiff) / bincount(bin_ass).shape
(bincount( bin_ass, weights=resdiff) / bincount(bin_ass)).shape
bin_ass
np.unique(bin_ass)
ubin = np.unique(bin_ass)
bincount( bin_ass, weights=resdiff) / bincount(bin_ass)
uvals = bincount( bin_ass, weights=resdiff) / bincount(bin_ass)
bar( ubin, uvals, width=.9)
ax = gca()
#ax.yaxis.set_ticklabels
res_x
res_x.shape
[ "%.1f"%x for x in res_x]
[ "%.2f"%x for x in res_x]
xlabs = [ "%.2f"%x for x in res_x[ubin]]
ax.xaxis.set_ticklabels(xlabs)
ax.xaxis.set_ticks(ubins)
ax.xaxis.set_ticks(ubin)
ax.xaxis.set_ticklabels(xlabs)
ax.xaxis.set_ticklabels(xlabs, rot90)
ax.xaxis.set_ticklabels(xlabs)
plt.xticks(rotation=90)
xlabs[0] = "low"
ax.xaxis.set_ticklabels(xlabs)
xlabel("Resolution $\AA^{-1}$", fontsize=20)
ax.tick_params(labelsize=14)
bin
#bincount( bin_ass, weights=resdiff) / bincount(bin_ass)
#resdiff = np.sqrt(np.abs(Aval**2 - Bval**2))
ylabel("$\sqrt{ | F_A^2 - F_B^2|}$", fontsize=20)
get_ipython().magic(u'pinfo a.merge_equivalents')
amerge = a.merge_equivalents()
#amerge.inconsistent_equivalents
dir(amerge)
amerge = a.merge_equivalents().array()
amerge.show_comprehensive_summary()
amerge.half_dataset_anomalous_correlation()
get_ipython().magic(u'pinfo amerge.half_dataset_anomalous_correlation')
a.half_dataset_anomalous_correlation()
a.as_amplitude_array().half_dataset_anomalous_correlation()
a.as_intensity_array().half_dataset_anomalous_correlation()
b.as_intensity_array().half_dataset_anomalous_correlation()
#amerge = a.merge_equivalents().as
len(H)
len(set(H))
H[0]
H[1]
H[2]
H[100]
H[200]
H
#sorted([h.h() for h in miller.sym_equiv_indices(sg, (11,-23,12)).indices()], key=itemgetter(0,1,2))
a.space_group
a.space_group()
sg = a.space_group()
sg
sg.info()
sgi = sg.info()
sgi.show_summary()
#sorted([h.h() for h in miller.sym_equiv_indices(sg96, h).indices()], key=itemgetter(0,1,2))
from operator import itemgetter
#[sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2)) for hkl in H]
H
len(H)
Hb = [i for i in b.indices()]
len(Hb)
#[sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2)) for hkl in H]
sg96 = a.space_group()
Hequiv_map = { hkl: sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2)) for hkl in H}
from cctbx import miller
Hequiv_map = { hkl: sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2)) for hkl in H}
Hequiv_map[H[0]]
Hequiv_map = { hkl: sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2))[0] for hkl in H}
Hequiv_map[H[0]]
Hequiv_map[H[100]]
Hequiv_map = { hkl: sorted([h.h() for h in miller.sym_equiv_indices(sg96, hkl).indices()], key=itemgetter(0,1,2))[:] for hkl in H}
Hequiv_map[H[100]]
Hequiv_map[H[101]]
#vals = d.intens2.values
#Bval = [b.value_at_index(h) for h in H]
a.value_at_index((0,0,4))
a.value_at_index((0,0,-4))
a.value_at_index((0,0,4))
a.value_at_index((0,0,-4))
Hequiv_map[H[101]]
a.value_at_index(Hequiv_map[H[101]][0])
a.value_at_index(Hequiv_map[H[101]][1])
a.value_at_index(Hequiv_map[H[101]][0])
a.value_at_index(Hequiv_map[H[100]][0])
a.value_at_index(Hequiv_map[H[100]][1])
a.value_at_index(Hequiv_map[H[100]][2])
a.value_at_index(Hequiv_map[H[100]][3])
a.value_at_index(Hequiv_map[H[100]][4])
a.value_at_index(Hequiv_map[H[100]][5])
a.value_at_index(Hequiv_map[H[100]][6])
a.value_at_index(Hequiv_map[H[100]][7])
H[0]
H[1]
H[2]
H[3]
H[4]
H[5]
H[6]
H[1000]
a.value_at_index(Hequiv_map[H[1000]][6])
a.value_at_index(Hequiv_map[H[1000]][0])
a.value_at_index(Hequiv_map[H[1000]][1])
a.value_at_index(Hequiv_map[H[1000]][2])
a.value_at_index(Hequiv_map[H[1000]][3])
a.value_at_index(Hequiv_map[H[1000]][4])
a.value_at_index(Hequiv_map[H[1000]][5])
a.value_at_index(Hequiv_map[H[1000]][6])
a.value_at_index(Hequiv_map[H[1000]][7])
Hequiv_map[H[1000]][7]
H[1000]
#miller.sym_equiv_ind()
HA_val_map = { hkl:a.value_at_index(hkl) for hkl in H}
HB_val_map = { hkl:b.value_at_index(hkl) for hkl in H}
hkl = (7,5,13)
#hkl = (7,5,13)
HA_val_map[hkl]
#miller.sym_equiv_ind()
neg_hkl = lambda hkl: tuple([-1*i for i in hkl])
neg_hkl(hkl)
miller.sym_equiv_ind(neg_hkl(hkl))
miller.sym_equiv_indices(neg_hkl(hkl))
neg_hkl
neg_hkl(hkl)
miller.sym_equiv_indices(sg96,neg_hkl(hkl))
[i.index() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl))]
[i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
get_neg_equv = lambda hkl, [i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
get_neg_equv = lambda hkl: [i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
get_neg_equv(hkl)
HA_val_map[get_neg_equv(hkl)[0]]
HA_val_map[get_neg_equv(hkl)[1]]
HA_val_map[get_neg_equv(hkl)[2]]
HA_val_map[get_neg_equv(hkl)[3]]
HA_val_map[get_neg_equv(hkl)[4]]
HA_val_map[get_neg_equv(hkl)[5]]
HA_val_map[get_neg_equv(hkl)[6]]
HA_val_map[get_neg_equv(hkl)[7]]
HA_val_map[get_neg_equv(hkl)[8]]
#HA_val_map[get_neg_equv(hkl)[]]
HA_val_map[get_neg_equv(hkl)[0]]
HA_val_map[hkl]
get_ipython().magic(u'pinfo a.generate_bijvoet_mates')
a.generate_bijvoet_mates()
b = a.generate_bijvoet_mates()
#b = a.generate_bijvoet_mates()
#bincount (bin_ass)
b=utils.open_flex('SB.pkl')
mates = a.generate_bijvoet_mates()
dir(mates)
mates.indices()
[i.h() for i in mates.indices()]
[i for i in mates.indices()]
[i for i in mates.indices()][0]
[i for i in mates.indices()][1]
[i for i in mates.indices()][2]
[i for i in mates.indices()][3]
get_ipython().magic(u'pinfo mates.match_bijvoet_mates')
mates.match_bijvoet_mates()
HA_val_map[get_neg_equv(hkl)[8]]
HA_val_map[get_neg_equv(hkl)[0]]
get_neg_equv = lambda hkl: [i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
get_ipython().magic(u'hist')
def get_neg_equiv(hkl):
    poss_equivs = [i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
    for hkl2 in poss_equivs:
        if hkl2 in HA_val_map:
            break
    return hkl2
get_neg_equiv(hkl)
HA_val_map[get_neg_equiv(hkl)]
get_ipython().magic(u'timeit HA_val_map[get_neg_equiv(hkl)]')
HA_val_map[get_neg_equiv(H[0])]
HA_val_map[get_neg_equiv(H[1])]
HA_val_map[get_neg_equiv(H[0])]
HA_val_map[H[0]]
get_ipython().magic(u'pinfo a.centric_flags')
a.centric_flags()
a.centric_flags()[0]
a.centric_flags().flags
a.centric_flags().flags()
get_ipython().magic(u'pinfo a.centric_flags')
aa = a.centric_flags()
dir(aa)
aa.show_array()
for i,j in aa.array():
    print i,j
    
aa.array()
for i,j in aa.indices():
    print i,j
    
    
aa.show_array()
dir(aa)
a.select_acentric()
a2 = a.select_acentric()
a2.indices()[0]
a.indices()[0]
H2 = [i for i in a2.indices()]
HA_val_map[H2[0]]
HA_val_map[get_neg_equv(H2[0])]
def get_neg_equiv(hkl):
    poss_equivs = [i.h() for i in miller.sym_equiv_indices(sg96,neg_hkl(hkl)).indices()]
    for hkl2 in poss_equivs:
        if hkl2 in HA_val_map:
            break
    return hkl2
get_neg_equ
get_neg_equiv(H2[0])
HA_val_map[get_neg_equiv(H2[0])]
HA_val_map[H2[0]]
FdelA = []
FdelB = []
for i,hkl in enumerate(H2):
    if i%200==0:
        print i, len(H2)
    F1 = HA_val_map[hkl]
    F0 = HA_val_map[get_neg_equiv(hkl)]
    
FdelA = []
FdelB = []
for i,hkl in enumerate(H2):
    if i%200==0:
        print i, len(H2)
    F1 = HA_val_map[hkl]
    F0 = HA_val_map[get_neg_equiv(hkl)]
    FdelA.append( F1-F0)
    F1 = HB_val_map[hkl]
    F0 = HB_val_map[get_neg_equiv(hkl)]
    FdelB.append( F1-F0)
    
FdelA
FdelB
FdelA
FdelB
FdelA
FdelA = []
FdelB = []
for i,hkl in enumerate(H2):
    hkl2 = get_neg_equiv(hkl)
    if i%200==0:
        print i, len(H2), hkl, hkl2
    F1 = HA_val_map[hkl]
    F0 = HA_val_map[hkl2]
    FdelA.append( F1-F0)
    F1 = HB_val_map[hkl]
    F0 = HB_val_map[hkl2]
    FdelB.append( F1-F0)
    
    
F1
F0
F1 - F0
F0
F1-F0
type(F1)
len(H2)
#bin_ass = np.digitize( RES,res_lims)-1
RES2 = [res_from_idx(*h) for h in H2]
bin_ass2 = np.digitize( RES2,res_lims)-1
for i in range(len(res_lims)):
    g = bin_ass2==i
    ("")
    
sum(g)
FdelA = np.array( FdelA)
FdelB = np.array( FdelB)
bin_ass2==10
sum(bin_ass2==10)
FdelA[bin_ass2==10]
FdelA[bin_ass2==10]
FdelB[bin_ass2==10]
FdelA[bin_ass2==10]
FdelA[bin_ass2==10] * FdelB[bin_ass2==10]
FdelA[bin_ass2==10] * FdelB[bin_ass2==10]
FdelB.conjugate
FdelB.conjugate()
FdelB
FdelA[bin_ass2==10] * FdelB[bin_ass2==10].conjugate()
FdelA[bin_ass2==10] * FdelB[bin_ass2==10].conjugate()
F1
F0
#FdelA[bin_ass2==10] * FdelB[bin_ass2==10].conjugate()
(F1-F0)**@
np.abs(F1-F0)
np.abs(F1-F0) / np.abs(.5*(F1+F0))
F1
np.abs(F1)
np.abs(F0)
np.abs(F0-F1)
np.abs(F1-F0)
np.abs(F1)-np.abs(F0)
FdelA = []
FdelB = []
for i,hkl in enumerate(H2):
    hkl2 = get_neg_equiv(hkl)
    if i%200==0:
        print i, len(H2), hkl, hkl2
    F1 = HA_val_map[hkl]
    F0 = HA_val_map[hkl2]
    FdelA.append( np.abs(F1)-np.abs(F0))
    F1 = HB_val_map[hkl]
    F0 = HB_val_map[hkl2]
    FdelB.append( np.abs(F1)-np.abs(F0))
    
    
    
FdelA
FdelB
FdelB[bin_ass2==10]
np.array(FdelB)[bin_ass2==10]
FdelB = np.array( FdelB)
FdelA = np.array( FdelA)
np.array(FdelB)[bin_ass2==10]
FdelB[bin_ass2==10]
FdelB[bin_ass2==10]*FdelA[bin_ass2==10]
np.mean(FdelB[bin_ass2==10]*FdelA[bin_ass2==10] ) 
#np.mean(FdelB[bin_ass2==10]*FdelA[bin_ass2==10] )  / np.sqrt
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    gA = FdelA[bin_ass2==g]
    gB = FdelB[bin_ass2==g]
    top = np.mean( gA*gB)
    bottom= np.sqrt(gA**2)*np.sqrt(gb**2)
    coeffs.append(top/bottom)
    
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    gA = FdelA[bin_ass2==g]
    gB = FdelB[bin_ass2==g]
    top = np.mean( gA*gB)
    bottom= np.sqrt(gA**2)*np.sqrt(gB**2)
    coeffs.append(top/bottom)
    
top
bottom
gA
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[bin_ass2==g]
    gB = FdelB[bin_ass2==g]
    top = np.mean( gA*gB)
    bottom= np.sqrt(gA**2)*np.sqrt(gB**2)
    coeffs.append(top/bottom)
    
coeffs[0]
coeffs[-1]
coeffs[10]
FdelA
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[bin_ass2==g]
    gB = FdelB[bin_ass2==g]
    top = np.mean( gA*gB)
    bottom= np.sqrt( np.mean(gA**2))*np.sqrt(np.mean(gB**2))
    coeffs.append(top/bottom)
    
bottom
top
gA
sum(g)
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[bin_ass2==g]
    gB = FdelB[bin_ass2==g]
    top = np.mean( gA*gB)
    bottom= np.sqrt( np.mean(gA**2))*np.sqrt(np.mean(gB**2))
    coeffs.append(top/bottom)
    print top, bottom
    
bin_ass2
FdelA[bin_ass2==1]
FdelA[bin_ass2==2]
FdelA[bin_ass2==10]
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[g]
    gB = FdelB[g]
    top = np.mean( gA*gB)
    bottom= np.sqrt( np.mean(gA**2))*np.sqrt(np.mean(gB**2))
    coeffs.append(top/bottom)
    print top, bottom
    
coeffs = []
for i in range(len(res_lims)):
    g = bin_ass2==i
    if sum(g)==0:
        coeffs.append(0)
        continue
    gA = FdelA[g]
    gB = FdelB[g]
    top = np.mean( gA*gB)
    bottom= np.sqrt( np.mean(gA**2))*np.sqrt(np.mean(gB**2))
    coeffs.append(top/bottom)
    print top/bottom
    
    
FdelA
np.mean(FdelA)
FdelA
np.mean(np.abs(FdelA))
Ha_vals
#HA_val_map = { hkl:a.value_at_index(hkl) for hkl in H}
[a.value_at_index(hkl) for hkl in H]
Fvals = np.array([a.value_at_index(hkl) for hkl in H])
Fvals.abs()
np.abs(Fvals)
np.mean(np.abs(Fvals))
np.mean(np.abs(FdelA)) / np.mean(np.abs(Fvals))
get_ipython().magic(u'hist')
np.mean(np.abs(FdelB)) / np.mean(np.abs(Fvals))
np.mean(np.abs(FdelA)) / np.mean(np.abs(Fvals))
get_ipython().magic(u'save SF_ana.py 1-380')
