figure()

bar( list(set(pids)), sigx_vals, width=.8, label="$\sigma_{\Delta_X}$")


outA = metrics.check_indexable2( refls_strong, spot_dataA_2, detector, beamA, optCrystal)
outB = metrics.check_indexable2( refls_strong, spot_dataB_2, detector, beamB, optCrystal)

dbls = where(outA['idxd']*outB['idxd'])[0]
pids = [ refls_strong[i]['panel'] for i in dbls]
pan_imgs = {i:iset.get_raw_data(0)[i].as_numpy_array() for i in set(pids)}
rois = [spot_utils.get_spot_roi( refls_strong, detector[0].get_image_size())[i] for i in dbls]

fig,axs = subplots( nrows=2, ncols=2, figsize=(4,4))
subplots_adjust(left=.02, right=.98, top=.98, bottom=.02, wspace=.06, hspace=.06)
a0 = axs[0,0]
a1 = axs[0,1]
a2 = axs[1,0]
a3 = axs[1,1]
outdir = "out"
for i in range(len(dbls)):

    pid = pids[i]
    roi = rois[i]
    (x1,x2),(y1,y2) = roi
    pimg = pan_imgs[pid][y1:y2, x1:x2]
    spotA = simsAB_2[0][pid][y1:y2, x1:x2]
    spotB = simsAB_2[1][pid][y1:y2, x1:x2]
    spotAB = spotA+spotB
    m = spotAB[spotAB>0].mean()
    s = spotAB[spotAB>0].std()
    vmax=m+1*s
    vmin = m-1*s
    show_args = {'vmin':None, 'vmax':None, 'cmap': 'gnuplot'}
    a0.clear()
    a1.clear()
    a2.clear()
    a3.clear()
    a0.axis('off')
    a1.axis('off')
    a2.axis('off')
    a3.axis('off')
    a0.imshow(spotA,**show_args)
    a1.imshow(spotB ,**show_args)
    a2.imshow(spotB+spotA,**show_args)
    a3.imshow( pimg, cmap=show_args['cmap'], vmax=130 )
    figname = os.path.join(outdir, "fat_overlap_%d.png" % i)
    savefig(figname)




#dbs
#rois[dbs[0]]
#rois[dbls[0]]
x1,x2,y1,y2 = rois[dbls[0]]
x1
spotA
figure()
imshow( spotA)
imshow( spotB)
imshow( spotB+spotA)
figure()
#pan_imgs = {i:iset.get_raw_data(0)[i].as_numpy_array() for i in pids}
#pan_imgs[pids[0]][
#pids = [ refls_strong[i]['panel'] for i in dbls]
pan_imgs[pids[0]][y1:y2, x1:x2]
pimg = pan_imgs[pids[0]][y1:y2, x1:x2]
imshow( pimg, vmax=100)
%hist



# code to search for neighboring doubles
dist_out2 = []
for i_d,d in enumerate(all_d):
    print i_d, len( all_d)
    R = d['refls_strong']
    idxA = d['residA']['indexed']
    idxB = d['residB']['indexed']
    idxAB= np.logical_or( idxA, idxB)
    Rpp = spot_utils.refls_by_panelname(R.select(flex.bool(idxAB)))
    for pid in Rpp:
        r = Rpp[pid]
        x,y ,_ = spot_utils.xyz_from_refl(r)
        C = distance.pdist(zip(x,y))
        nC = np.sum( (4 < C) & (C < 7))
        dist_out2.append( [nC, i_d, pid] )


for i_d,d in enumerate(all_d):
    print i_d, len( all_d)
    R = d['refls_strong']
    idxA = d['residA']['indexed']
    idxB = d['residB']['indexed']
    idxA_not_B = idxA * np.logical_not( idxB)
    idxB_not_A = idxB * np.logical_not( idxA)







####
# plot the hists
bins = np.arange(-8.05,8.05,.1)
binsY = arange(-8.05, 7.05, .1)
bins_cent = bins[1:] * .5 + bins[:-1] * .5
bins_centY = binsY[1:] * .5 + binsY[:-1] * .5
X,Y = np.meshgrid( bins_cent,bins_centY)

img = histogram2d(dvecs[d<5, 1], dvecs[d<5, 0], bins=[binsY, bins], normed=True)[0]


sigX = {}
sigY = {}
x0 = {}
y0 = {}
outs = {}
success = {}
imgs = {}
gimgs = {}
Amp = {}
for pid in set(pids):
    img = histogram2d(dvecsP[pid][:, 1], dvecsP[pid][:, 0], bins=[binsY, bins], normed=True)[0]
    out = minimize( metrics.gauss2d_resid, (.75,.5,.5,1,1), args=(X,Y,img), method='Nelder-Mead')
    success[pid] = out['success']
    x0[pid] = out['x'][1]
    y0[pid] = out['x'][2]
    sigX[pid] = out['x'][3]
    sigY[pid] = out['x'][4]
    outs[pid] = out
    imgs[pid] = img
    Amp[pid] = out['x'][0]
    gimgs[pid] = metrics.gauss2d(X,Y, *out['x'])











# code to go from saved refl table to
# other one

#def add_rlp(refls, detector, beam):
from dials.algorithms.indexing.indexer import indexer_base
from dxtbx.model.experiment_list import Experiment, ExperimentList
from cxi_xdr_xes.two_color.two_color_indexer import index_reflections_detail

# make the 3 experiments
e = Experiment()
e.beam = beamA
e.crystal = optCrystal
e.detector = detector
e2 = deepcopy(e)
e2.beam = beamB
e2.crystal = e.crystal
e3 = deepcopy(e)
beamAB = deepcopy(beamA)
waveAB = beamA.get_wavelength()*.5 + beamB.get_wavelength()*.5
beamAB.set_wavelength( waveAB)
e3.beam = beamAB
e3.crystal = e.crystal
el = ExperimentList()
el.append(e)
el.append(e2)
el.append(e3)
####

# make the rlps
refls_w_mm = indexer_base.map_spots_pixel_to_mm_rad(refls, detector, scan=None)
indexer_base.map_centroids_to_reciprocal_space(refls_w_mm, detector, beamA, goniometer=None)
rlps1 = refls_w_mm['rlp']
indexer_base.map_centroids_to_reciprocal_space(refls_w_mm, detector, beamB, goniometer=None)
rlps2 = refls_w_mm['rlp']

for i in range(len(refls_w_mm)):
    refls_w_mm['id'][i] = -1
spot_utils.as_single_shot_reflections(refls_w_mm)
index_reflections_detail(None, el, refls_w_mm, detector, rlps1, rlps2)
