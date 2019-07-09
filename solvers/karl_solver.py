
import numpy as np
from IPython import embed

from scipy.sparse.linalg import lsmr
from scitbx.lstbx import normal_eqns_solving
from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common
from cctbx import sgtbx, crystal, miller
from cctbx.array_family import flex as cctbx_flex
from cxid9114.parameters import ENERGY_CONV
from scipy.sparse import coo_matrix
import cxid9114


class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):

    def __init__(self, initial_estimates, Nh, tom=False):

        super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
        self.initialize(initial_estimates)
        self.stored_functional = []
        self.Nh = Nh
        self.tom=tom

    def build_up(self, objective_only=False):
        if not objective_only:
            self.counter+=1

        self.reset()
        if not objective_only:
            if not self.tom:
                functional = self.functional_karl(self.x)   # NOTE: Im a cpp function in solvers_ext.cpp
            else:
                functional = self.functional_karl_tom(self.x)   # NOTE: Im a cpp function in solvers_ext.cpp

            G = self.x[3*self.Nh:].as_numpy_array()
            Gm =G.mean()
            Gs = G.std()
            print("\n<><><><><><>")
            print("Count=%d Functional value: %.10e, Gain=%.3f (%.3f)" % (self.counter, functional, Gm, Gs))
            print("<><><><><><>\n")
            self.stored_functional.append(functional)
        if not self.tom:
            self.karlize(objective_only, current_values = self.x)
        else:
            self.karlize_tom(objective_only, current_values = self.x)


class karl_solver:

    def __init__(self, data, truth=None, conj_grad=True, weights=None, tom=False):
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

        self.helper = eigen_helper(initial_estimates=self.x_init, Nh=self.Nhkl, tom=tom)
        self.helper.tom = tom
        self.helper.eigen_wrapper.conj_grad = conj_grad

        # NOTE: I'm a cpp function defined in solvers_ext.cpp
        self.helper.set_karl_data(self.Yobs, self.Wobs,
            self.Aidx, self.Gidx,
                self.PA, self.PB, self.LA, self.LB, self.EN,
                self.Nhkl, self.Ns, )

        if not tom:
            print self.calc_func()[1]
            print self.helper.functional_karl(self.helper.x)
        else:
            print self.calc_func_TomT()[1]
            print self.helper.functional_karl_tom(self.helper.x)
        #self._solve()

    def solve(self):
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

        self.x = self.x_init.copy()

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

    def set_values_for_lsqr(self):

        EN = self.EN.as_numpy_array()
        self.Nh = Nh = self.Nhkl
        self.Aidx = self.Aidx.as_numpy_array()
        self.Gidx = self.Gidx.as_numpy_array()
        self.PA = self.PA.as_numpy_array()
        self.PB = self.PB.as_numpy_array()
        self.LB = self.LB.as_numpy_array()
        self.LA = self.LA.as_numpy_array()
        self.a_enA = EN[:Nh][self.Aidx]
        self.b_enA = EN[Nh:2*Nh][self.Aidx]
        self.c_enA = EN[2*Nh:3*Nh][self.Aidx]
        self.a_enB = EN[3*Nh:4*Nh][self.Aidx]
        self.b_enB = EN[4*Nh:5*Nh][self.Aidx]
        self.c_enB = EN[5*Nh:][self.Aidx]

        self.Aterm = self.LA*self.PA
        self.Bterm = self.LB*self.PB

        self.s2_A = (1+self.a_enA +self.b_enA)
        self.s2_B = (1+self.a_enB +self.b_enB)

        # sparse matrix stuff:
        self.dFo_cols = self.Aidx.astype(int)
        self.dFa_cols = (self.Aidx + self.Nhkl).astype(int)
        self.dAlpha_cols = (self.Aidx + 2*self.Nhkl).astype(int)
        self.dG_cols = (self.Gidx + 3*self.Nhkl).astype(int)

        Nmeas = len(self.PB)
        self.sparse_rows = np.array(range(Nmeas)*4, int)
        self.sparse_cols = np.concatenate((self.dFo_cols, self.dFa_cols,
                                          self.dAlpha_cols, self.dG_cols))

        init_data = np.zeros(self.sparse_rows.shape[0], float)  # initial with empty
        self.SPARSE_M = coo_matrix(
            (init_data, (self.sparse_rows, self.sparse_cols)))

        self.n_iters=0
        self.residuals = []

    def buildup_lsmr(self, **kwargs):
        Fo = np.exp(self.x[:self.Nh])[self.Aidx]
        Fa = self.x[self.Nh:2*self.Nh][self.Aidx]
        al = self.x[2*self.Nh:3*self.Nh][self.Aidx]
        G = self.x[3*self.Nh:][self.Gidx]
        GAterm = G*self.Aterm
        GBterm = G*self.Bterm

        Faa = Fa*Fa
        Foo = Fo*Fo
        FaFo = Fa*Fo

        SIN = np.sin(al)
        COS = np.cos(al)

        s1_A = (2+self.b_enA)*COS + self.c_enA*SIN
        s1_B = (2+self.b_enB)*COS + self.c_enB*SIN
        dFo_A = 2*Foo + s1_A * FaFo
        dFo_B = 2*Foo + s1_B * FaFo
        dFo = GAterm*dFo_A + GBterm*dFo_B

        dFa_A = 2*self.s2_A * Fa + s1_A*Fo
        dFa_B = 2*self.s2_B * Fa + s1_B*Fo
        dFa = GAterm*dFa_A + GBterm*dFa_B

        dAl_A = ((-2-self.b_enA)*SIN + self.c_enA*COS)*FaFo
        dAl_B = ((-2-self.b_enB)*SIN + self.c_enB*COS)*FaFo
        dAl = GAterm*dAl_A + GBterm*dAl_B

        dG_A = Foo + self.s2_A*Fa + s1_A*FaFo
        dG_B = Foo + self.s2_B*Fa + s1_B*FaFo
        dG = self.Aterm*dG_A + self.Bterm*dG_B

        sparse_data = np.concatenate((dFo, dFa, dAl, dG))
        print("Setting sparse data")
        self.SPARSE_M.data = sparse_data

        betaA = Foo + self.s2_A*Fa + s1_A*FaFo
        betaB = Foo + self.s2_B*Fa + s1_B*FaFo

        self.Ymodel = GAterm*betaA + GBterm*betaB
        self.Beta = self.Yobs - self.Ymodel

        b = self.SPARSE_M.T.dot(self.Beta)
        A = self.SPARSE_M.T.dot(self.SPARSE_M)
        print("Solving...")
        a = lsmr(A, b, **kwargs)[0]  # solve

        self.n_iters += 1
        self.x += a  # update
        resid = np.dot(self.Beta, self.Beta)
        self.residuals.append(resid)
        print "Iter %d ; Residual: %e" % (self.n_iters, resid)


    def calc_func_TomT(self, x=None, using_numpy=False):
        if x is None:
            x = self.helper.x.as_numpy_array()
        if using_numpy:
            Nh = self.Nhkl
            Aidx = self.Aidx
            Gidx = self.Gidx
            PA = self.PA
            PB = self.PB
            LB = self.LB
            LA = self.LA
            a_enA = self.a_enA
            b_enA = self.b_enA
            c_enA = self.c_enA
            a_enB = self.a_enB
            b_enB = self.b_enB
            c_enB = self.c_enB

        else:
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

        Fo = np.exp(x[:Nh])[Aidx]
        Fa = x[Nh:2*Nh][Aidx]
        al = x[2*Nh:3*Nh][Aidx]
        G = x[3*Nh:][Gidx]

        COS = np.cos(al)
        SIN = np.sin(al)

        Aterm = PA*LA*(Fo*Fo + Fa*Fa*(1 + a_enA + b_enA) +
                                Fo*Fa*(2*COS + b_enA*COS + c_enA*SIN))
        Bterm = PB*LB*(Fo*Fo + Fa*Fa*(1 + a_enB + b_enB) +
                       Fo*Fa*(2*COS + b_enB*COS + c_enB*SIN))


        ymodel = G*(Aterm+Bterm)
        return ymodel, np.sum((self.Yobs.as_numpy_array() - ymodel)**2)





    def to_mtzA(self, hkl_map, mtz_name, x=None, verbose=False, stride=50, tom=False):
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
                if i % stride == 0:
                    print ("Reflection %d / %d" % (i+1, Nh))
            a = self.data["a_enA"][i]
            b = self.data["b_enA"][i]
            c = self.data["c_enA"][i]
            h = hkl_map2[i]
            Fp = np.exp(logFprot[i])
            Fa = Fheavy[i]
            COS = np.cos(alpha[i])
            SIN = np.sin(alpha[i])
            if not tom:
                IA = Fp**2 + Fa**2 * a + Fa*Fp* (b*COS + c*SIN)
            else:
                IA = Fp**2 + (1+a+b)*Fa**2  + ((2+b)*COS + c*SIN)*Fp*Fa

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

