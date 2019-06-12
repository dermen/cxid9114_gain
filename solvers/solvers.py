import gen_data
from IPython import embed
from scitbx import lbfgs
import cxid9114
from itertools import izip
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

import time
from scitbx.array_family import flex
import numpy as np


class LBFGSsolver(object):
    def __init__(self, data, guess, truth=None, lbfgs=True):
        self.Gprm_truth = self.IAprm_truth = self.IBprm_truth = None  # we might not know the truth
        if truth is not None:
            self.Gprm_truth = flex.double(np.ascontiguousarray(truth["Gprm"]))
            self.IAprm_truth = flex.double(np.ascontiguousarray(truth["IAprm"]))
            self.IBprm_truth = flex.double(np.ascontiguousarray(truth["IBprm"]))


        self.stored_functional = []
        self.Yobs = flex.double(np.ascontiguousarray(data["Yobs"]))  # NOTE expanded
        self.PA = flex.double(np.ascontiguousarray(data["PA"]))# NOTE expanded
        self.PB = flex.double(np.ascontiguousarray(data["PB"]))# NOTE expanded
        self.LA = flex.double(np.ascontiguousarray(data["LA"]))# NOTE expanded
        self.LB = flex.double(np.ascontiguousarray(data["LB"]))# NOTE expanded

        self.Nhkl = len(set(data['Aidx']))  # self.IAprm_truth)
        self.Ns = len(set(data['Gidx']))  ##self.Gprm_truth)
        self.Nmeas = len(self.Yobs)

        self.Aidx = flex.size_t(np.ascontiguousarray(data["Aidx"]))
        self.Gidx = flex.size_t(np.ascontiguousarray(data["Gidx"]))

        self.guess = guess

        if lbfgs:
            self.x = flex.double(np.ascontiguousarray(guess["IAprm"])).concatenate(
                flex.double(np.ascontiguousarray(guess["IBprm"]))).concatenate( flex.double(np.ascontiguousarray(guess["Gprm"])))
            assert( len(self.x) == self.Nhkl*2 + self.Ns)
            self.n = len(self.x)



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

    def ideal_functional(self):
        try:
            _ = self.IAprm_truth
        except AttributeError as error:
            print(error)
            print("No stored truth values to compute ideal funcitonal")
            return
        #IAcurr, IBcurr, Gcurr = self.unpack()
        self.resid = self.Yobs - self.Gprm_truth * \
                     ( self.IAprm_truth * self.LA * self.PA + self.IBprm_truth*self.LB*self.PB )
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
        print f, "YEE"
        self.stored_functional.append(f)
        return f, self.gradients()

    def minimize(self):
        self.minimizer = lbfgs.run(target_evaluator=self)


class LogIsolver(LBFGSsolver):
    def __init__(self, *args, **kwargs):
        LBFGSsolver.__init__(self, *args, **kwargs)
        if self.IAprm_truth is not None:
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
        if self.IAprm_truth is not None:
            self.IAprm_truth = flex.log(self.IAprm_truth)
            self.IBprm_truth = flex.log(self.IBprm_truth)
            self.Gprm_truth = flex.log(self.Gprm_truth)

        IAx = flex.log(self.x[:self.Nhkl])
        IBx = flex.log(self.x[self.Nhkl:2*self.Nhkl])
        #Gx = self.x[2*self.Nhkl:]
        Gx = flex.log(self.x[2*self.Nhkl:])
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



class LSMRsolver:
    def __init__(self, data, guess):

        self.Yobs = data["Yobs"]
        self.Gidx = data["Gidx"]
        self.Aidx = data["Aidx"]
        self.LAvals = data["LA"]
        self.LBvals = data["LB"]
        self.PAvals = data["PA"]
        self.PBvals = data["PB"]

        self.Nhkl = np.unique( self.Aidx).shape[0]
        self.Ngain = np.unique( self.Gidx).shape[0]

        self.x = np.hstack( (guess["IAprm"], guess["IBprm"], guess["Gprm"])  )
        self.x[:2*self.Nhkl] = np.log(self.x[:2*self.Nhkl])

        self.Nmeas = len(self.Yobs)
        self.Nprm = len(self.x)
        self.Beta = np.zeros(self.Nmeas)
        self.niters = 0
        self.residuals = []

    def iterate(self, **kwargs):
        BIG_row = []
        BIG_col =  []
        BIG_data = []
        for i_meas, (yobs, i_g, i_hkl, LA, LB, PA, PB) in enumerate(
                izip(self.Yobs, self.Gidx, self.Aidx, self.LAvals, self.LBvals, self.PAvals, self.PBvals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA --- AmplitudesB --- Gains]
            IA_guess = self.x[i_hkl]
            IB_guess = self.x[self.Nhkl + i_hkl]
            G_guess = self.x[2*self.Nhkl + i_g]

            # residual between data and guess
            self.Beta[i_meas] = yobs - (np.exp(IA_guess)*LA*PA + np.exp(IB_guess)*LB*PB) * G_guess

            # partial derivitives
            dA = G_guess * np.exp(IA_guess) * LA*PA
            dB = G_guess * np.exp( IB_guess) * LB*PB
            dG = np.exp(IA_guess)*LA*PA + np.exp(IB_guess)*LB*PB

            # store the data in coordinate format for making sparse array
            BIG_col.extend([i_hkl, self.Nhkl + i_hkl, 2*self.Nhkl + i_g])
            BIG_row.extend([i_meas] * 3)
            BIG_data.extend([dA, dB, dG])

        # make the big sparse array
        BS = coo_matrix((BIG_data, (BIG_row, BIG_col)),
                        shape=(self.Nmeas, self.Nprm))
        BS = BS.tocsr()  # convert to csr for speed gains?

        # continue with Wolfram notation
        # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html
        b = BS.T.dot(self.Beta)
        A = BS.T.dot(BS)
        a = lsmr(A,b, damp=np.random.uniform(.1,100)*0, **kwargs)[0]  # solve

        self.niters += 1
        self.x += a  # update
        resid = np.dot(self.Beta, self.Beta)
        self.residuals.append( resid)
        self.BS = BS  # store for looking

        print "Iter %d ; Residual: %e, Press Ctrl-C to break" % (self.niters, resid)


# =============
# Get a guess
#lsmr_solver = LSMRsolver(data, guesses)
#embed()

# ================

#Solver1 = LBFGSsolver(data=data, guess=truth, truth=truth)
#Solver2 = LBFGSsolver(data=data, guess=guesses, truth=truth)

#prm = np.load("lsmr_solver_x.npy")
#prm[:2*lsmr_solver.Nhkl] = np.exp(prm[:2*lsmr_solver.Nhkl])
#guesses["Gprm"] = prm[2*lsmr_solver.Nhkl:]
# ..

if __name__=="__main__":
    data = gen_data.gen_data(Nshot_max=500)
    guesses = gen_data.guess_data(data, perturbate=True)
    truth = gen_data.guess_data(data, perturbate=False)
    #prm = np.load("_temp_4.npz")
    #guesses["IAprm"] = prm["AmpA_final"]
    #guesses["IBprm"] = prm["AmpB_final"]
    #guesses["Gprm"] = prm["Gain_final"]

    #LogSolve = LogIsolver(data=data, guess=guesses, truth=truth)

    LogSolveCurve = LogIsolverCurve(use_curvatures=False, data=data, guess=guesses, truth=truth)

    embed()

    #prm = np.load("_temp_4.npz")
    prm = LogSolveCurve.x.as_numpy_array()
    Nh = LogSolveCurve.Nhkl
    guesses["IAprm"] = np.exp(prm[:Nh])
    guesses["IBprm"] = np.exp(prm[Nh:2*Nh])
    guesses["Gprm"] = prm[2*Nh:]

    LogSolveCurve = LogIsolverCurve(use_curvatures=True, data=data, guess=guesses, truth=truth)


    #LogSolveCurve.run()
