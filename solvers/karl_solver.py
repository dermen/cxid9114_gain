
import numpy as np
from IPython import embed

from scitbx.lstbx import normal_eqns_solving
from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common
from cctbx import sgtbx, crystal, miller
from cctbx.array_family import flex as cctbx_flex
from cxid9114.parameters import ENERGY_CONV

import cxid9114


class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):

    def __init__(self, initial_estimates, Nh):

        super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
        self.initialize(initial_estimates)
        self.stored_functional = []
        self.Nh = Nh

    def build_up(self, objective_only=False):
        if not objective_only:
            self.counter+=1

        self.reset()
        if not objective_only:
            functional = self.functional_karl(self.x)   # NOTE: Im a cpp function in solvers_ext.cpp
            G = self.x[3*self.Nh:].as_numpy_array()
            Gm =G.mean()
            Gs = G.std()
            print("\n<><><><><><>")
            print("Count=%d Functional value: %.10e, Gain=%.3f (%.3f)" % (self.counter, functional, Gm, Gs))
            print("<><><><><><>\n")
            self.stored_functional.append(functional)

        self.karlize(objective_only, current_values = self.x)


class karl_solver:

    def __init__(self, data, truth=None, conj_grad=True, weights=None):
        self.data = data
        self.truth = truth

        self._prepare_constants()
        self._store_initial_guess()
        self._define_useful_scalars()
        self.counter = 0

        # set dummie weights for now
        if weights is None:
            self.Wobs = flex.double(np.ones(len(self.Yobs)))
        else:
            self.Wobs = weights

        self.helper = eigen_helper(initial_estimates=self.x_init, Nh=self.Nhkl)
        self.helper.eigen_wrapper.conj_grad = conj_grad

        # NOTE: I'm a cpp function defined in solvers_ext.cpp
        self.helper.set_karl_data(self.Yobs, self.Wobs,
            self.Aidx, self.Gidx,
                self.PA, self.PB, self.LA, self.LB, self.EN,
                self.Nhkl, self.Ns, )

        print self.calc_func()[1]
        print self.helper.functional_karl(self.helper.x)
        self._solve()

    def _solve(self):
        self.helper.restart()
        try:
            _ = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
                   non_linear_ls=self.helper,
                   n_max_iterations=200,
                   track_all=True,
                   step_threshold=0.0001)
        except KeyboardInterrupt:
            pass
        print "End of minimization: Converged", self.helper.counter, "cycles"
        print self.helper.get_eigen_summary()
        print "Converged functional: ", self.helper.functional(self.helper.x)

    def _prepare_constants(self):
        """
        LA,LB,PA,PB, a_lambda, b_lambda, c_lambda are all constants in this experiment
        """
        self.LA = flex.double(np.ascontiguousarray(self.data["LA"]))
        self.LB = flex.double(np.ascontiguousarray(self.data["LB"]))
        self.PA = flex.double(np.ascontiguousarray(self.data["PA"]))
        self.PB = flex.double(np.ascontiguousarray(self.data["PB"]))

        # TODO compute the a constants on the fly ?
        self.EN = np.concatenate(
            (self.data["a_enA"],
            self.data["b_enA"],
            self.data["c_enA"],
            self.data["a_enB"],
            self.data["b_enB"],
            self.data["c_enB"]))

        self.EN = flex.double(np.ascontiguousarray(self.EN))
        self.Yobs = flex.double(np.ascontiguousarray(self.data["Yobs"]))

        self.Aidx = flex.size_t(np.ascontiguousarray(self.data["Aidx"]))
        self.Gidx = flex.size_t(np.ascontiguousarray(self.data["Gidx"]))

    def _store_initial_guess(self):
        """
        the X parameter array will consist of a concatenation f 4 vectors
        1 of length Nhkl for the |Fprotein|^2
        1 of length Nhkl for the |Fheavy|^2
        1 of length Nhkl for the alpha
        1 of length Ns for the per-crystal scale factors
        """
        self.x_init = np.concatenate(
            (np.log(self.data["Fprot_prm"]),
             self.data["Fheavy_prm"],
             self.data["alpha_prm"],
             self.data["Gain_prm"]))

        # convert to flex
        self.x_init = flex.double(np.ascontiguousarray(self.x_init))

    def _define_useful_scalars(self):
        """
        stores numbers of things
        """
        self.Nmeas = len(self.Yobs)
        self.Nhkl = self.data["Fprot_prm"].shape[0]
        self.Ns = self.data["Gain_prm"].shape[0]

    def calc_func(self, x=None):
        if x is None:
            x = self.helper.x.as_numpy_array()
        Nh = self.Nhkl
        Aidx = self.Aidx.as_numpy_array()
        Gidx = self.Gidx.as_numpy_array()
        EN = self.EN.as_numpy_array()
        PA = self.PA.as_numpy_array()
        PB = self.PB.as_numpy_array()
        LB = self.LB.as_numpy_array()
        LA = self.LA.as_numpy_array()
        a_enA = EN[:Nh][Aidx]
        b_enA = EN[Nh:2*Nh][Aidx]
        c_enA = EN[2*Nh:3*Nh][Aidx]
        a_enB = EN[3*Nh:4*Nh][Aidx]
        b_enB = EN[4*Nh:5*Nh][Aidx]
        c_enB = EN[5*Nh:][Aidx]

        self.prot = np.exp(x[:Nh])[Aidx]
        self.heav = x[Nh:2*Nh][Aidx]
        self.alpha = x[2*Nh:3*Nh][Aidx]
        self.G = x[3*Nh:][Gidx]

        Aterm = PA*LA*(self.prot**2 + self.heav**2 * a_enA + self.prot*self.heav*b_enA*np.cos(self.alpha) +
                       self.prot*self.heav*c_enA*np.sin(self.alpha))
        Bterm = PB*LB*(self.prot**2 + self.heav**2 * a_enB + self.prot*self.heav*b_enB*np.cos(self.alpha) +
                       self.prot*self.heav*c_enB*np.sin(self.alpha))

        ymodel = self.G*(Aterm+Bterm)
        return ymodel, np.sum((self.Yobs.as_numpy_array() - ymodel)**2)


    def to_mtzA(self, hkl_map, mtz_name, x=None, verbose=False, stride=50):
        """
        Takes the refined Fheavy and Fprot and creates an MTZ file
        for the A-channel (8944 eV) Structure factors

        :param hkl_map:  dictionary to lookup HKL from parameter index
        :param mtz_name:
        :param x: input array of parameters
        :return:
        """
        if x is None:
            x = self.helper.x.as_numpy_array()

        Nh = self.Nhkl
        assert (Nh == len(hkl_map))
        logFprot = x[:Nh]
        Fheavy = x[Nh:2*Nh]
        alpha = x[2*Nh: 3*Nh]

        hkl_map2 = {v: k for k, v in hkl_map.iteritems()}
        hout, Iout = [], []
        for i in range(Nh):
            if verbose:
                if i % stride==0:
                    print ("Reflection %d / %d" % (i+1, Nh))
            a = self.data["a_enA"][i]
            b = self.data["b_enA"][i]
            c = self.data["c_enA"][i]
            h = hkl_map2[i]
            Fp = np.exp(logFprot[i])
            Fa = Fheavy[i]
            COS = np.cos(alpha[i])
            SIN = np.sin(alpha[i])
            IA = Fp**2 + Fa**2 * a + Fa*Fp* (b*COS + c*SIN)

            hout.append(h)
            Iout.append(IA)

        sg = sgtbx.space_group(" P 4nw 2abw")
        Symm = crystal.symmetry(unit_cell=(79, 79, 38, 90, 90, 90), space_group=sg)
        hout = tuple(hout)
        mil_idx = cctbx_flex.miller_index(hout)
        mil_set = miller.set(crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)
        Iout_flex = flex.double(np.ascontiguousarray(Iout))
        mil_ar = miller.array(mil_set, data=Iout_flex).set_observation_type_xray_intensity()

        waveA = ENERGY_CONV / 8944.
        out = mil_ar.as_mtz_dataset(column_root_label="Iobs", title="B", wavelength=waveA)
        out.add_miller_array(miller_array=mil_ar.average_bijvoet_mates(), column_root_label="IMEAN")
        obj = out.mtz_object()
        obj.write(mtz_name)

