import gen_data
from IPython import embed
from scitbx import lbfgs
data = gen_data.gen_data()

guesses = gen_data.guess_data(data, perturbate=True)
truth = gen_data.guess_data(data, perturbate=False)


from scitbx.array_family import flex
import numpy as np
class LBFGSsolver(object):
    def __init__(self, data, guess, truth):
        self.Gprm_truth = flex.double(np.ascontiguousarray(truth["Gprm"]))
        self.IAprm_truth = flex.double(np.ascontiguousarray(truth["IAprm"]))
        self.IBprm_truth = flex.double(np.ascontiguousarray(truth["IBprm"]))

        self.Yobs = flex.double(np.ascontiguousarray(data["Yobs"]))  # NOTE expanded
        self.PA = flex.double(np.ascontiguousarray(data["PA"]))# NOTE expanded
        self.PB = flex.double(np.ascontiguousarray(data["PB"]))# NOTE expanded
        self.LA = flex.double(np.ascontiguousarray(data["LA"]))# NOTE expanded
        self.LB = flex.double(np.ascontiguousarray(data["LB"]))# NOTE expanded

        self.Nhkl = len(self.IAprm_truth)
        self.Ns = len(self.Gprm_truth)
        self.Nmeas = len(self.Yobs)


        self.x = flex.double(np.ascontiguousarray(guess["IAprm"])).concatenate(
            flex.double(np.ascontiguousarray(guess["IBprm"]))).concatenate( flex.double(np.ascontiguousarray(guess["Gprm"])))
        assert( len(self.x) == self.Nhkl*2 + self.Ns)
        self.n = len(self.x)

        self.Aidx = flex.size_t(np.ascontiguousarray(data["Aidx"]))
        self.Gidx = flex.size_t(np.ascontiguousarray(data["Gidx"]))


    def unpack(self):
        IAprm = self.x[:self.Nhkl]
        IBprm = self.x[self.Nhkl: 2*self.Nhkl]
        Gprm = self.x[2*self.Nhkl:]

        IAexpa = IAprm.select(self.Aidx)
        IBexpa = IBprm.select(self.Aidx)
        Gexpa = Gprm.select(self.Gidx)

        return IAexpa, IBexpa, Gexpa

    def functional(self):
        IAcurr, IBcurr, Gcurr = self.unpack()
        self.resid = self.Yobs - Gcurr * ( IAcurr * self.LA * self.PA + IBcurr*self.LB*self.PB )
        resid_sq = self.resid * self.resid
        return .5 * flex.sum(resid_sq)

    def gradients(self):
        IAcurr, IBcurr, Gcurr = self.unpack()
        grad_vec = flex.double(self.n)
        for i_meas in range( self.Nmeas):

            i_hkl = self.Aidx[i_meas]
            i_s = self.Gidx[i_meas]

            grad_vec[i_hkl] += -self.resid[i_meas] * Gcurr[i_meas] * self.PA[i_meas] * self.LA[i_meas]
            grad_vec[self.Nhkl + i_hkl] += -self.resid[i_meas] * Gcurr[i_meas] * self.PB[i_meas] * self.LB[i_meas]
            grad_vec[2*self.Nhkl + i_s] += -self.resid[i_meas] * ( IAcurr[i_meas]*self.PA[i_meas] * self.LA[i_meas]\
                                                           + IBcurr[i_meas] * self.PB[i_meas] * self.LB[i_meas])

        return grad_vec

    def compute_functional_and_gradients(self):
        f = self.functional()
        print f
        return f, self.gradients()

    def run(self):
        self.minimizer = lbfgs.run(target_evaluator=self)

Solver1 = LBFGSsolver(data=data, guess=truth, truth=truth)
Solver2 = LBFGSsolver(data=data, guess=guesses, truth=truth)

embed()
