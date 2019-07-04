
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
            print("Functional value: %.10e" % functional)
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
            (np.log(self.data["Iprot_prm"]),
             np.log(self.data["Iheavy_prm"]),
             self.data["alpha_prm"],
             self.data["Gain_prm"]))

        # convert to flex
        self.x_init = flex.double(np.ascontiguousarray(self.x_init))

    def _define_useful_scalars(self):
        """
        stores numbers of things
        """
        self.Nmeas = len(self.Yobs)
        self.Nhkl = self.data["Iprot_prm"].shape[0]
        self.Ns = self.data["Gain_prm"].shape[0]



