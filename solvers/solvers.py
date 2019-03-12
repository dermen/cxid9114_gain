import gen_data
from IPython import embed
from scitbx import lbfgs
import cxid9114
from libtbx.test_utils import approx_equal

data = gen_data.gen_data()

guesses = gen_data.guess_data(data, perturbate=True)
truth = gen_data.guess_data(data, perturbate=False)


import time
from scitbx.array_family import flex
import numpy as np
class LBFGSsolver(object):
    def __init__(self, data, guess, truth):
        self.Gprm_truth = flex.double(np.ascontiguousarray(truth["Gprm"]))
        self.IAprm_truth = flex.double(np.ascontiguousarray(truth["IAprm"]))
        self.IBprm_truth = flex.double(np.ascontiguousarray(truth["IBprm"]))


        self.stored_functional = []
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

    def minimize(self):
        self.minimizer = lbfgs.run(target_evaluator=self)


class LogIsolver(LBFGSsolver):
    def __init__(self, *args, **kwargs):
        LBFGSsolver.__init__(self, *args, **kwargs)
        self.IAprm_truth = flex.log(self.IAprm_truth)
        self.IBprm_truth = flex.log(self.IBprm_truth)
        IAx = flex.log(self.x[:self.Nhkl])
        IBx = flex.log(self.x[self.Nhkl:2*self.Nhkl])
        Gx = self.x[2*self.Nhkl:]
        self.x = IAx.concatenate(IBx)
        self.x = self.x.concatenate(Gx)

    def functional(self):
        IAcurr, IBcurr, Gcurr = self.unpack()
        self.resid = self.Yobs - Gcurr * ( flex.exp(IAcurr) * self.LA * self.PA + flex.exp(IBcurr)*self.LB*self.PB )
        resid_sq = self.resid * self.resid
        return .5 * flex.sum(resid_sq)

    def gradients(self):
        print("Entering gradients")
        IAcurr, IBcurr, Gcurr = self.unpack()

        t = time.time()
        grad_vec1 = cxid9114.grad_vecs_cpp(resid=self.resid,
                                Gcurr=Gcurr,
                                IAcurr=IAcurr,
                               IBcurr=IBcurr,
                                Aidx=self.Aidx,
                               Gidx=self.Gidx,
                                PA=self.PA,
                               PB=self.PB,
                                LA=self.LA,
                               LB=self.LB, Nhkl=self.Nhkl, n=self.n )
        #t2 = time.time()
        #print("Exiting gradients cpp; took %.4f sec" % (t2-t))

        #t = time.time()
        #grad_vec = flex.double(self.n)
        #for i_meas in range( self.Nmeas):

        #    i_hkl = self.Aidx[i_meas]
        #    i_s = self.Gidx[i_meas]

        #    grad_vec[i_hkl] += -self.resid[i_meas] * Gcurr[i_meas] * self.PA[i_meas] * self.LA[i_meas] * np.exp(IAcurr[i_meas])
        #    grad_vec[self.Nhkl + i_hkl] += -self.resid[i_meas] * Gcurr[i_meas] * self.PB[i_meas] * self.LB[i_meas] * np.exp(IBcurr[i_meas])
        #    grad_vec[2*self.Nhkl + i_s] += -self.resid[i_meas] * ( np.exp(IAcurr[i_meas])*self.PA[i_meas] * self.LA[i_meas]
        #                                            + np.exp(IBcurr[i_meas]) * self.PB[i_meas] * self.LB[i_meas])

        #t2 = time.time()
        #print("Exiting gradients; took %.4f sec" % (t2-t))

        #assert( approx_equal(grad_vec, grad_vec1))

        return grad_vec1

from scitbx.lbfgs.tst_curvatures import lbfgs_with_curvatures_mix_in

class LogIsolverCurve(lbfgs_with_curvatures_mix_in, LBFGSsolver):
    def __init__(self, use_curvatures=True, *args, **kwargs):
        LBFGSsolver.__init__(self, *args, **kwargs)
        self.IAprm_truth = flex.log(self.IAprm_truth)
        self.IBprm_truth = flex.log(self.IBprm_truth)
        #self.Gprm_truth = flex.log(self.Gprm_truth)

        IAx = flex.log(self.x[:self.Nhkl])
        IBx = flex.log(self.x[self.Nhkl:2*self.Nhkl])
        Gx = self.x[2*self.Nhkl:]
        #Gx = flex.log(self.x[2*self.Nhkl:])
        self.x = IAx.concatenate(IBx)
        self.x = self.x.concatenate(Gx)

        if use_curvatures:
            self.minimizer = lbfgs_with_curvatures_mix_in.__init__(
              self,
              min_iterations=0,
              max_iterations=None,
              use_curvatures=True)

    def curvatures(self):
        """aren't the gradient and the curvature the same in this case ? """
        IAcurr, IBcurr, Gcurr = self.unpack()

        curva = cxid9114.curvatures2(resid=self.resid,
                                        Gcurr=Gcurr,
                                        IAcurr=IAcurr,
                                        IBcurr=IBcurr,
                                        Aidx=self.Aidx,
                                        Gidx=self.Gidx,
                                        PA=self.PA,
                                        PB=self.PB,
                                        LA=self.LA,
                                        LB=self.LB, Nhkl=self.Nhkl, n=self.n )
        return curva

    def functional(self):
        IAcurr, IBcurr, Gcurr = self.unpack()
        self.resid = self.Yobs - Gcurr * ( flex.exp(IAcurr) * self.LA * self.PA + flex.exp(IBcurr)*self.LB*self.PB )
        #self.resid = self.Yobs - flex.exp(Gcurr) * ( flex.exp(IAcurr) * self.LA * self.PA + flex.exp(IBcurr)*self.LB*self.PB )
        resid_sq = self.resid * self.resid
        return .5 * flex.sum(resid_sq)

    def gradients(self):
        IAcurr, IBcurr, Gcurr = self.unpack()

        grad_vec1 = cxid9114.grad_vecs_cpp(resid=self.resid,
                                Gcurr=Gcurr,
                                IAcurr=IAcurr,
                               IBcurr=IBcurr,
                                Aidx=self.Aidx,
                               Gidx=self.Gidx,
                                PA=self.PA,
                               PB=self.PB,
                                LA=self.LA,
                               LB=self.LB, Nhkl=self.Nhkl, n=self.n )

        return grad_vec1

Solver1 = LBFGSsolver(data=data, guess=truth, truth=truth)
Solver2 = LBFGSsolver(data=data, guess=guesses, truth=truth)

prm = np.load("_temp_3.npz")
guesses["Gprm"] = prm["Gain_final"]
guesses["IAprm"] = prm["AmpA_final"]
guesses["IBprm"] = prm["AmpB_final"]
LogSolve = LogIsolver(data=data, guess=guesses, truth=truth)
LogSolveCurve = LogIsolverCurve(use_curvatures=False, data=data, guess=guesses, truth=truth)

embed()
#LogSolveCurve.run()
