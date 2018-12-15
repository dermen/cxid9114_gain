from __future__ import division
from six.moves import range
from cctbx.crystal import symmetry
from cctbx.array_family import flex
import cctbx.miller
from scitbx.matrix import sqr,col
from dxtbx.model.detector import DetectorFactory as detector_factory
from dials.array_family import flex
from cctbx import crystal
from dxtbx.model.beam import BeamFactory as beam_factory
import random

'''

This program is designed to simulate thermolysin diffraction at two
distinct wavelengths in a single image in a random orientation or a known
orientation may be input (especially useful for debugging).
The simulation is meant to be a test of a new method for indexing two "colors"
in a single shot.

The input requires a target unit cell and space group. This target cell
is varied over a Gaussian distribution with a set standard deviation. A list of
potential hkls is generated and corresponding q vectors are calculated
using the orientation matrix and the hkls. The s1 test vectors (EQ) are calculated
for each wavelength and determined to on the Ewald sphere if the ratio of the
magnitudes of S0 and EQ is within 2% of 1.0.

If the ray intersection of EQ with the detector is a valid detector coordinate
it is accepted as an observed spot and placed in the reflection table in both
pixel and mm coordinates.

A dictionary containing a reflection table, spots array, array of spots at wavelength 1,
array of spots at wavelength 2 and the orientation matrix. The spots arrays and
orientation matrix are used for verification of indexing.

'''

#helper functions:

def get_dials_obs(px_positions, px_var, px_sq_err, mm_positions, mm_var, mm_sq_err):
  ''' gets obs array used to create shoeboxes '''
  from dials.model.data import Intensity, Centroid, CentroidData, Observation
  obs = flex.observation()
  for iobs in range(len(px_positions)):
    obs.append( Observation(
                  panel = 0,
                  centroid = Centroid(
                                px = CentroidData(
                                         position=(px_positions[iobs][0],px_positions[iobs][1],0),
                                         variance=(px_var[iobs][0],px_var[iobs][1],px_var[iobs][2]),
                                         std_err_sq=(px_sq_err[iobs][0], px_sq_err[iobs][1],px_sq_err[iobs][2])),
                                mm = CentroidData(
                                         position=(mm_positions[iobs][0],mm_positions[iobs][1],0),
                                         variance=(mm_var[iobs][0],mm_var[iobs][1],mm_var[iobs][2]),
                                         std_err_sq=(px_sq_err[iobs][0], px_sq_err[iobs][1],px_sq_err[iobs][2]))),
                 intensity = Intensity(
                                observed_value=1.0,
                                observed_variance=1.0,
                                observed_success =True)))
  return obs


class two_color_still_sim(object):
  ''' class to simulate a still at two wavelengths '''
  def __init__(self, symm, rot=None):

    uc=symm.unit_cell()
    spgrp=symm.space_group()


    #The U matrix to calculate A^*
    if rot is None:
      rot = flex.random_double_r3_rotation_matrix()
      self.rot = rot
      #print rot

    # add gassian random noise "error" to unit cell, sigma = 0.3
    # change a and c axis using symmetry constraints
    a = uc.parameters()[0] # b will take this same value
    c = uc.parameters()[2]
    v_params= [a,c]

    new_uc_params=[random.gauss(v_params[i], 1.0) for i in range(len(v_params))]
    new_uc=(new_uc_params[0],new_uc_params[0],new_uc_params[1],uc.parameters()[3],
            uc.parameters()[4], uc.parameters()[5])

    print new_uc

    new_sym=crystal.symmetry(new_uc, space_group=spgrp)
    new_uc= new_sym.unit_cell()
    A = sqr(rot) * sqr(new_uc.reciprocal().orthogonalization_matrix())
    self.A = A


    #set of hkls including bijvoet pairs
    hkl_list=cctbx.miller.build_set(symm,True,d_min=3).expand_to_p1()
    self.hkl_list=hkl_list



  #calculate s1 and check if on Ewald sphere
  def ewald_proximity_test(self, beams, hkl_list, A, detector):
    ''' test used to determine which spots are on the Ewald sphere,
    returns a dictionary of results including a reflection table, spots arrays
    and orientation matrix '''

    wavelength1 = beams[0].get_wavelength()
    wavelength2 = beams[1].get_wavelength()
    s0_1 = col((0,0,-1/wavelength1))
    s0_2 = col((0,0,-1/wavelength2))


    #create variables for h arrays
    filtered1 = flex.miller_index()
    #create variables for spots indices
    spots1 = flex.vec2_double()

    #create variables for h arrays
    filtered2 = flex.miller_index()
    #create variables for spots indices
    spots2 = flex.vec2_double()

    self.A=A
    q = flex.mat3_double([A]*len(hkl_list.all_selection()))*hkl_list.indices().as_vec3_double()

    EQ1 = q + s0_1
    EQ2 = q + s0_2
    len_EQ1 = flex.double([col(v).length() for v in EQ1])
    ratio1 = len_EQ1*wavelength1
    len_EQ2 = flex.double([col(v).length() for v in EQ2])
    ratio2 = len_EQ2*wavelength2


    for i in range(len(EQ1)):
      if ratio1[i]> 0.998 and ratio1[i] < 1.002:
        pix = detector[0].get_ray_intersection_px(EQ1[i])
        if detector[0].is_coord_valid(pix):
          spots1.append(pix)
          filtered1.append(hkl_list.indices()[i])

    for i in range(len(EQ2)):
      if ratio2[i]> 0.998 and ratio2[i] < 1.002:
        pix = detector[0].get_ray_intersection_px(EQ2[i])
        if detector[0].is_coord_valid(pix):
          spots2.append(pix)
          filtered2.append(hkl_list.indices()[i])


    #create reflection table
    #filtered s1 and r for reflection table

    spots = spots1.concatenate(spots2)
    lab_coord1=flex.vec3_double([detector[0].get_pixel_lab_coord(i) for i in spots1])
    lab_coord2=flex.vec3_double([detector[0].get_pixel_lab_coord(i) for i in spots2])
    s1_vecs1=lab_coord1.each_normalize()*(1/wavelength1)
    s0_vec1=col((0,0,-1/wavelength1))
    r_vecs1=s1_vecs1-s0_vec1
    s1_vecs2=lab_coord2.each_normalize()*(1/wavelength2)
    s0_vec2=col((0,0,-1/wavelength2))
    r_vecs2=s1_vecs2-s0_vec2

    #create one large reflection table for both experiments
    x_px,y_px = tuple(spots.parts()[0:2])

    px_array = flex.vec3_double(x_px,y_px,flex.double(len(x_px),0.0))
    px_to_mm = detector[0].pixel_to_millimeter(spots)
    x_mm,y_mm = tuple(px_to_mm.parts()[0:2])
    mm_array = flex.vec3_double(x_mm,y_mm,flex.double(len(x_px),0.0))
    px_var = flex.vec3_double(flex.double(len(x_px),0.25),
                  flex.double(len(x_px),0.25),flex.double(len(x_px),0.0))
    px_sq_err = flex.vec3_double(flex.double(len(x_px),0.1),
                  flex.double(len(x_px),0.1),flex.double(len(x_px),0.0))
    mm_var = flex.vec3_double(flex.double(len(x_px),0.05),
                 flex.double(len(x_px),0.05),flex.double(len(x_px),0.05))
    mm_sq_err = flex.vec3_double(flex.double(len(x_px),0.01),
                  flex.double(len(x_px),0.01),flex.double(len(x_px),0.01))

    obs_array =get_dials_obs(px_positions=px_array, px_var=px_var,
        px_sq_err=px_sq_err, mm_positions=mm_array, mm_var=mm_var,
        mm_sq_err=mm_sq_err)
    shoeboxes = flex.shoebox(len(obs_array))
    refl = flex.reflection_table(obs_array,shoeboxes)
    refl['id'] = flex.int(len(refl),-1)
    s1_vecs = s1_vecs1.concatenate(s1_vecs2)

    refl['s1'] = s1_vecs

    refl['xyzobs.px.variance'] = px_var
    refl['xyzobs.mm.value'] = mm_array
    refl['xyzobs.mm.variance'] = flex.vec3_double([(v[0]*0.005, v[1]*0.005, v[2]*0.0) for v in refl['xyzobs.px.variance']])
    filtered = filtered1.concatenate(filtered2)
    refl['miller_index'] = filtered # is reset to all 0s once close spots are merged
    bbox = flex.int6([(0,1,0,1,0,1)]*len(refl))
    refl['bbox'] = bbox
    #refl['rlp'] = r_vecs
    refl['xyzobs.px.value']=px_array

    # define experiment ids for each wavelength to check simulation with
    #indexing results in separate new column of reflection table not used in indexing
    exp_id1 = flex.int(len(spots1),0)
    exp_id2 = flex.int(len(spots2),1)
    exp_id = exp_id1.concatenate(exp_id2)
    refl['set_id'] = exp_id

    self.refl = refl
    sim_res={'reflection_table': refl , 'all_spots': spots , 'wavelength_1_spots': spots1 , 'wavelength_2_spots': spots2, 'input_orientation': A};

    return sim_res



if __name__=="__main__":
  wavelength1=12398/7400 #wavelength for 2 color experiment in Angstroms
  wavelength2=12398/7500 #wavelength for 2 color experiment in Angstroms

  #get detector with a single panel
  detector=detector_factory.simple('SENSOR_UNKNOWN',125,(97.075,97.075),'+x','-y',(0.11,0.11),(1765,1765))


  #unit cell and space group for lysozyme
  known_symmetry=crystal.symmetry("78,78,37,90,90,90","P43212")


  beam1 = beam_factory.simple_directional((0,0,1), wavelength1)
  beam2 = beam_factory.simple_directional((0,0,1), wavelength2)
  beams = [beam1, beam2]

  flex.set_random_seed(42)
  for i in range(5):


    data = two_color_still_sim(known_symmetry,rot=None)
    hkl_list = data.hkl_list
    A = data.A


    sim_data = data.ewald_proximity_test(beams, hkl_list, A, detector)
    refl = sim_data.get('reflection_table')
    refl1 = refl.select(refl['set_id']==0)
    refl2 = refl.select(refl['set_id']==1)


    #unit test to make sure all miller indices are unique for each experiment id

    assert len(refl1['miller_index'])==len(set(refl1['miller_index']))

    assert len(refl2['miller_index'])==len(set(refl2['miller_index']))

    print "ok"
