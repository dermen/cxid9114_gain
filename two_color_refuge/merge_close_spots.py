from __future__ import division
from six.moves import range
from cctbx.crystal import symmetry
from cctbx.array_family import flex
import cctbx.miller
from math import sqrt
from dxtbx.model.detector import DetectorFactory as detector_factory
from dials.array_family import flex
from cctbx import crystal
from cxi_xdr_xes.two_color import two_color_still_sim
from dxtbx.model.beam import BeamFactory as beam_factory

'''

This program is designed to merge spot positions when two spots are separated
by a distance of less than two pixels to create a more realistic simulation of
measured data. Uses the spots array from the two color still simulation to
calculate the Euclidean distance in pixels between spot pairs. The spots that
are less than two pixels are combined into a single spot position and the
associated pixel variance is set to 1.0 (the maximim for shift of a single
spot to the average position).

The reflection id; used only for checking the indexing solution; is set to 2.
The miller indices are each reset to (0,0,0) prior to two color indexing.

A plot of resolution vs. spot distance is generated for spots that have the
same hkl. The close spots are in the medium to low resolution range, 14 - 35
angstroms.

'''


class merge_close_spots(object):
  ''' class designed to merge close spots and plot spot distance vs. resolution '''
  def __init__(self):

    #unit cell and space group for lysozyme
    known_symmetry=crystal.symmetry("78,78,37,90,90,90","P43212")

    data = two_color_still_sim.two_color_still_sim(symm = known_symmetry, rot = None)
    self.data =data
    wavelength1=12398/7400 #wavelength for 2 color experiment in Angstroms
    wavelength2=12398/7500 #wavelength for 2 color experiment in Angstroms


    #get detector with a single panel
    detector=detector_factory.simple('SENSOR_UNKNOWN',125,(97.075,97.075),'+x','-y',(0.11,0.11),(1765,1765))
    self.detector=detector

    beam1 = beam_factory.simple_directional((0,0,1), wavelength1)
    beam2 = beam_factory.simple_directional((0,0,1), wavelength2)

    beams = [beam1, beam2]
    self.beams =beams

    hkl_list = data.hkl_list
    self.hkl_list = hkl_list

    A = data.A
    self.A =A

    two_color_sim = data.ewald_proximity_test(beams, hkl_list, A, detector)
    self.two_color_sim = two_color_sim



  def spot_proximity(self, res):
    ''' calculates Euclidean distance in pixels between spots
    and reasigns the spots to a single spot at the average of the two spot
    positions. Returns a mapping between average spot positions and original
    spots and a new reflection table with the experiment ids reset to
    0 and miller indices set to (0,0,0).'''

    spots_a = res['wavelength_1_spots']
    spots_b = res['wavelength_2_spots']
    refl = res['reflection_table']
    refl_a = refl.select(refl['set_id']==0)
    refl_b = refl.select(refl['set_id']==1)
    a=flex.bool(len(refl_a)*[True])
    b=flex.bool(len(refl_b)*[True])
    spots_a=refl_a['xyzobs.px.value']
    spots_b=refl_b['xyzobs.px.value']
    mapping=[]
    ax,ay,az=spots_a.parts()
    bx,by,bz=spots_b.parts()
    spot_dist=[]
    a_hkl=refl_a['miller_index']
    b_hkl=refl_b['miller_index']
    spot_pair_ind=[]
    spot_dist_close=[]

    for i,hkl in enumerate(a_hkl):
        for j in range(len(b_hkl)):

            if hkl==b_hkl[j] and i>=j:
                mapping.append((i,j))


    for k in range(len(mapping)):
        h,l=mapping[k]
        diff0sq=(ax[h] - bx[l])**2
        diff1sq=(ay[h] - by[l])**2
        diff2sq=(az[h] - bz[l])**2
        spot_dist.append(sqrt(diff0sq + diff1sq + diff2sq))
        #calculate average position of close spots
        if spot_dist[k]<2:
            ave1=(ax[h]+bx[l])/2
            ave2=(ay[h]+by[l])/2
            ave3=(az[h]+bz[l])/2
            b[l]=False
            spots_a[h]=(ave1,ave2,ave3)
            refl_a['xyzobs.px.variance'][h] = (1.0, 1.0, 0.0)
            refl_a['set_id'][h]=2
            refl_b['set_id'][h]=2
            spot_dist_close.append(spot_dist[k])

    #now we create a new reflection table with close spots replaced by average spot position
    import copy
    new_refl=copy.deepcopy(refl_a)

    new_refl.extend(refl_b.select(b))

    #keep track of simulated miller indices for comparing in indexer by making new reflection table column
    miller_index = new_refl['miller_index']
    new_refl['set_miller_index'] = miller_index

    #set all miller indices to 0 prior to indexing
    new_refl['miller_index'] == flex.miller_index(len(new_refl), (0,0,0))

    # tell the indexer that all the reflections come from the zero-th imageset
    new_refl['id'] = flex.int(len(new_refl), 0)




    return new_refl, mapping



  def spot_resolution(self,detector,beams, spots_a, spots_b, mapping):
    ''' uses mapping of pix spots indices with matching hkl values in reflection
    tables for each wavelength to calculate spot distances in pixels and the
    average resolution of the spot pairs'''

    pannel=detector[0]

    res_ave=[]
    a_res=[]
    b_res=[]

    dists=[]
    ax,ay=spots_a.parts()
    bx,by=spots_b.parts()



    for i in range(len(spots_a)):
        a_res.append(pannel.get_resolution_at_pixel(beams[0].get_s0(), (ax[i],ay[i])))

    for j in range(len(spots_b)):
        b_res.append(pannel.get_resolution_at_pixel(beams[1].get_s0(), (bx[j],by[j])))

    #this will give all distances between unique reflection pairs
    #with the same hkl measured at two distinct wavelengths
    for k in range(len(mapping)):
        i,j=mapping[k]
        diff0sq=(ax[i] - bx[j])**2
        diff1sq=(ay[i] - by[j])**2
        dists.append(sqrt(diff0sq+diff1sq) )
        res_ave.append((a_res[i]+b_res[j])/2)




    return res_ave, dists

def plot_spot_sep_vs_res(self, res,dist):
    ''' plots the spot distance (separation in pixels) vs. the average resolution
    of the two spots of spots that have the same hkl '''
    from matplotlib import pyplot as plt
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.set_xlim((0,max(res)+1))
    ax.set_ylim((0,max(dist)+1))
    ax.set_title('Average Resolution vs. Spot Distance')
    ax.set_ylabel('spot distance (pix)')
    ax.set_xlabel('average resolution')

    plt.tight_layout()

    plt.scatter([res],[dist],c='blue',linewidth=0)

    plt.show()


if __name__=="__main__":

  #unit test average close spots should be in the low resolution region >=15 angstroms
  for j in range(100):
    data = merge_close_spots.merge_close_spots()
    res = data.two_color_sim
    info = data.spot_proximity(res)
    detector = data.detector
    beams = data.beams
    spots_a = res['wavelength_1_spots']
    spots_b = res['wavelength_2_spots']
    mapping = info[1]
    resolution, distances = data.spot_resolution(detector, beams, spots_a, spots_b, mapping)
    for i in range(len(distances)):
      if resolution[i] >=14:
        test = distances[i]
        assert distances[i] <= 2, test

  print "ok"

