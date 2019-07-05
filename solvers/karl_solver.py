
import numpy as np
import cxid9114
from scitbx.lstbx import normal_eqns_solving
from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common


class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):

    def __init__(self, initial_estimates):

        super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
        self.initialize(initial_estimates)
        self.stored_functional = []

    def build_up(self, objective_only=False):
        if not objective_only:
            self.counter+=1

        self.reset()
        if not objective_only:
            functional = self.functional_karl(self.x)   # NOTE: Im a cpp function in solvers_ext.cpp
            print("\n\t<><><><><><>")
            print("Count=%d Functional value: %.10e" % (self.counter, functional))
            print("\t<><><><><><>\n")
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

        self.helper = eigen_helper(initial_estimates=self.x_init) #, Nhkl=self.Nhkl)
        self.helper.eigen_wrapper.conj_grad = conj_grad


        # NOTE: I'm a cpp function defined in solvers_ext.cpp
        self.helper.set_karl_data(self.Yobs, self.Wobs,
            self.Aidx, self.Gidx,
                self.PA, self.PB, self.LA, self.LB, self.EN,
                self.Nhkl, self.Ns, )

        print self.calc_func()[1]
        print self.helper.functional_karl(self.helper.x)
        exit()
        self.helper.restart()

        try:
            _ = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
                   non_linear_ls=self.helper,
                   n_max_iterations=2000,
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
             np.log(self.data["Fheavy_prm"]),
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

        prot = np.exp(x[:Nh])[Aidx]
        heav = np.exp(x[Nh:2*Nh])[Aidx]
        alpha = x[2*Nh:3*Nh][Aidx]
        G = x[3*Nh:][Gidx]

        Aterm = PA*LA*(prot**2 + heav**2 * a_enA + prot*heav*b_enA*np.cos(alpha) +
                       prot*heav*c_enA*np.sin(alpha))
        Bterm = PB*LB*(prot**2 + heav**2 * a_enB + prot*heav*b_enB*np.cos(alpha) +
                       prot*heav*c_enB*np.sin(alpha))

        ymodel = G*(Aterm+Bterm)
        return ymodel, np.sum((self.Yobs.as_numpy_array() - ymodel)**2)
