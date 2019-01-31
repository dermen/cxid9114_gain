from itertools import product, izip
import numpy as np
from scitbx.matrix import sqr, col
from cxid9114.spots import spot_utils
from cxid9114.sim import sim_utils


class JitterFactory:
    def __init__(self, crystal, Patt, refls, data_image, szx=30, szy=30):
        """
        Some tools for jittering the indexing solution
        to try and come up with a better solution
        THis uses simtbx wrapped into the Patt object
        :param crystal: dxtbx crystal
        :param Patt: an instance of PatternFactory that has been
            primed (Patt.primer has been run)
            Note, primer sets the Amatrix, Fcalcs, energy, and flux
        :param refls: strong spots, we will simulate there vicinity
            and use the simultion overlap as a proxy for indexing
            solution quality
        :param data_image: the raw data image that is used to compute
            overlap
        """
        detector = Patt.detector[Patt.panel_id]
        self.FS = detector.get_fast_axis()
        self.SS = detector.get_slow_axis()
        self.A = sqr(crystal.get_A())
        self.U = sqr(crystal.get_U())
        self.B = sqr(crystal.get_B())
        self.s0 = Patt.beam.get_s0()
        self.Patt = Patt
        self.data = data_image
        self.rois = spot_utils.get_spot_roi(refls, dxtbx_image_size=detector.get_image_size(), szx=szx, szy=szy)
        self.spotX, self.spotY,_ = spot_utils.xyz_from_refl(refls)
        self.spot_mask = spot_utils.strong_spot_mask(refls, self.data.shape)
        self.counts = spot_utils.count_roi_overlap(self.rois, img_size=self.data.shape)
        if np.any( self.counts > 0):
            self.should_norm = True
        else:
            self.should_norm = False

    def jitter_Amat(self, anglesX=None, anglesY=None, anglesZ=None, deg=True, overlapper=None,
                plot=False, plot_delay=0.05):
        """
        Rock the Amatrix in the X,Y, Z dimension
        :param anglesX:
        :param anglesY:
        :param deg:
        :param overlapper:
        :return:
        """
        if plot:
            import pylab as plt
        if overlapper is None:
            overlapper = lambda im1,im2: np.sum( im1*im2)

        x = col((1,0,0))
        y = col((0,1,0))
        z = col((0,0,1))

        xR = x.axis_and_angle_as_r3_rotation_matrix
        yR = y.axis_and_angle_as_r3_rotation_matrix
        zR = z.axis_and_angle_as_r3_rotation_matrix

        opX = []
        opY = []
        opZ = []
        I =  sqr( (1,0,0,0,1,0,0,0,1))
        if anglesX is not None:
            for ang in anglesX:
                opX.append( xR(ang, deg=deg))
        else:
            opX = [I]
        if anglesY is not None:
            for ang in anglesY:
                opY.append( yR(ang, deg=deg))
        else:
            opY = [I]
        if anglesZ is not None:
            for ang in anglesZ:
                opZ.append( zR(ang, deg=deg))
        else:
            opZ = [I]

        XYZ_seq = product( opX, opY, opZ)
        A_seq = []
        overlaps = []

        if plot:
            plt.figure()
            ax = plt.gca()
            im=ax.imshow( np.random.random((10,10)), vmax=2)
            ax.plot( self.spotX,self.spotY,'r.')

        for X,Y,Z in XYZ_seq:
            Anew = X*Y*Z*self.U*self.B
            self.Patt.SIM2.Amatrix = Anew.transpose().elems
            sim_img = self.Patt.sim_rois(self.rois, reset=True)
            if self.should_norm:
                sim_img[ self.counts > 1]  /= self.counts[ self.counts > 1]
            overlaps.append(overlapper(self.spot_mask, sim_img))
            A_seq.append( Anew)
            if plot:
                im.set_data( sim_img)
                plt.draw()
                plt.pause(plot_delay)

        maxpos = np.argmax( overlaps)
        Aopt = A_seq[maxpos]

        # reset the A matrix
        self.Patt.Amatrix = self.A
        return {"A_seq": A_seq, "overlaps": np.array(overlaps), "Aopt": Aopt}


def jitter_panels(panel_ids, crystal, refls, det, beam, FF, en, data_imgs, flux,
                  ret_best=False, scanX=None, scanY=None, scanZ=None,
                  mos_dom=1, mos_spread=0.01, **kwargs):
    """
    Helper function for doing fast refinements by rocking the U-matrix
    portion of the crystal A matrix
    NOTE, additional kwargs are passed  to the PatternFactor instantiator

    :param panel_ids: a list of panels ids where refinement
        if None we will refine using all panels with 1 or more reflections
    :param crystal: dxtbx crystal model
    :param refls: reflection table with strong spots
    :param det: dxtxb multi panel detector model
    :param beam: dxtbx beam model. Note this beam is only used to
        specify the beam direction but not the its energy, we will
        override the energy.
    :param FF: a cctbc miller array instance or a list of them
        corresponding to the energies list (same length)
    :param en: a float or list of floats specifying energies
    :param data_imgs: the data imgs corresponding to the
    :param flux: float or list of floats, the flux of each energy
        and it should be same length as en
    :param ret_best, whether to return the best Amatrix or all the data
        from each scan
    :param scanX: scan angles for the X-rotation matrix
        (same for scanY and Z, note these are lab frame notataions)
    :param mos_dom, number os mosaic domains in crystal
    :param mos_spread, angular spread of reflections from mosaicity (tangential component)
    :return: best A matrix , or else a bunch of data for each panel
        that can be used to select best A matrix
    """
    if isinstance( en, float) or isinstance( en, int):
        en = [en]
        FF = [FF]
        flux = [flux]
    else:
        # TODO: assert iterables
        assert( len(en) == len(FF) == len(flux))

    if scanX is None:
        scanX = np.arange( -0.3, 0.3, 0.025)
    if scanY is None:
        scanY = np.arange(-0.3, 0.3, 0.025)
    # leave Z scanning off by default

    R = spot_utils.refls_by_panelname(refls)
    out = {}
    for i_color in range(len(en)):
        for pid,dat in izip(panel_ids, data_imgs):
            print i_color+1 , len(en), pid+1, len( panel_ids)
            P = sim_utils.PatternFactory( detector=det, beam=beam,
                                          panel_id=pid, recenter=True,
                                          **kwargs)
            P.adjust_mosaicity(mos_dom, mos_spread)
            P.primer( crystal, en[i_color], flux[i_color], FF[i_color])
            JR = JitterFactory(crystal, P, R[pid], dat)
            if pid not in out:  # initialize
                out[pid] = JR.jitter_Amat(scanX, scanY, scanZ, plot=False)
            else:  # just add the overlay from the second color
                out_temp = JR.jitter_Amat(scanX, scanY, scanZ, plot=False)
                out[pid]['overlaps'] += out_temp['overlaps']

    if ret_best:
        max_overlay_pos = np.sum( [out[i]['overlaps'] for i in out], axis=0).argmax()
        bestA = out[out.keys()[0]]["A_seq"][ max_overlay_pos]  # its the same sequence for each panel

        return bestA
    else:
        return out

#P = sim_utils.PatternFactory( detector=det, beam=B, panel_id=1)
#F = scattering_factors.get_scattF( B.get_wavelength(), pdb_name="../sim/4bs7.pdb", algo='direct',dmin=1.5, ano_flag=True)
#P.primer( C, F, 8950, 1e14)
#JR = jitter_refine.JitterFactory(C,det[1], B, R[1], dat, P)
#out = JR.scan_xy(arange(-.7,.7,0.05), arange( -.7,.7,0.05) )
#imshow( np.reshape(out["overlaps"],  ( int(sqrt(len(out["overlaps"]))),-1)))
