
"""
"""
import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix
import pandas
import sys
from itertools import izip

np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

Niter = 2
Nsolvers = 1000
K = 10000**2 * 1e12
method = "lsmr"

#Nmeas = 2500000
df = pandas.read_hdf("r62_simdata2_fixed_oversamp_labeled.hdf5","data")
#df.reset_index(inplace=True)

Ndata_total = len(df)
Nmeas = len(df)

#vals = abs(df.lhs  - df.rhs)
#df = df.loc[vals < 1e-7]  # only select those shots with overlaps

ydata = df.D.values
adata = df.adata.values
gdata = df.gdata.values
LAdata = df.LA.values
LBdata = df.LB.values
PAdata = df.PA.values
PBdata = df.PB.values
gains = df.gain.values

# remap adata and gdata
amp_remap = {a:i_a for i_a, a in enumerate(set(adata))}
adata = np.array([amp_remap[a] for a in adata])
gain_remap = {g:i_g for i_g,g in enumerate(set(gdata))}
gdata = np.array([gain_remap[g] for g in gdata])

Nmeas = len( ydata)
Namp = np.unique(adata).shape[0]
Ngain = np.unique(gdata).shape[0]
print "N-unknowns: 2xNhkl + Ngain = %d unknowns," % (2*Namp + Ngain)
print "N-measurements: %d" % Nmeas

class ScaleSolver:
    def __init__(self, guess, yvals, gvals, LAvals, LBvals, PAvals, PBvals, FAvals, FBvals, Ngain, save_iters=True, K=1e20):
        self.PRM = np.array(guess, float)
        self.yvals = yvals
        self.gvals = gvals
        self.LAvals = LAvals
        self.LBvals = LBvals
        self.PAvals = PAvals
        self.PBvals = PBvals
        self.FAvals = FAvals
        self.FBvals = FBvals
        self.Ngain = Ngain

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.niters = 0
        self.residuals = []
        self.save_iters = save_iters
        self.K = K  # default FF for constant structure fator simulation..

    def iterate(self, **kwargs):
        K = self.K
        BIG_row = []
        BIG_col =  []
        BIG_data = []
        for i_meas, (yobs, gdata, LA, LB, PA, PB, FA, FB) in enumerate(
                izip(self.yvals, self.gvals, self.LAvals, self.LBvals, self.PAvals, self.PBvals, self.FAvals, self.FBvals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA --- AmplitudesB --- Gains]
            gain_guess = self.PRM[gdata]

            # residual between data and guess
            self.Beta[i_meas] = yobs - gain_guess * (FA*LA*PA/K + FB*LB*PB/K)

            # partial derivitives
            dG = FA*LA*PA/K + FB*LB*PB/K

            # store the data in coordinate format for making sparse array
            BIG_col.extend([gdata ])
            BIG_row.extend([i_meas])
            BIG_data.extend([dG])

        # make the big sparse array
        BS = coo_matrix((BIG_data, (BIG_row, BIG_col)),
                        shape=(self.Nmeas, self.Nprm))
        BS = BS.tocsr()  # convert to csr for speed gains?

        # continue with Wolfram notation
        # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html
        b = BS.T.dot(self.Beta)
        A = BS.T.dot(BS)
        a = lsmr(A,b, **kwargs)[0]  # solve

        self.niters += 1
        self.PRM += a  # update
        resid = np.dot(self.Beta, self.Beta)
        self.residuals.append( resid)
        self.BS = BS  # store for looking

        print "Iter %d ; Residual: %e, Press Ctrl-C to break" % (self.niters, resid)

        if self.save_iters:
            np.save("_PRM_iter%d" % self.niters, self.PRM)


class AmpSolver:
    def __init__(self, guess, yvals, avals, LAvals, LBvals, PAvals, PBvals, GainVals, Namp, save_iters=True, K=1e20):
        self.PRM = np.array(guess, float)
        self.yvals = yvals
        self.avals = avals
        self.LAvals = LAvals
        self.LBvals = LBvals
        self.PAvals = PAvals
        self.PBvals = PBvals
        self.GainVals = GainVals
        self.Namp = Namp

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.niters = 0
        self.residuals = []
        self.save_iters = save_iters
        self.K = K  # default FF for constant structure fator simulation..

    def iterate(self, **kwargs):
        K = self.K
        BIG_row = []
        BIG_col =  []
        BIG_data = []
        for i_meas, (yobs, adata, LA, LB, PA, PB, G) in enumerate(
                izip(self.yvals, self.avals, self.LAvals, self.LBvals, self.PAvals, self.PBvals, self.GainVals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA --- AmplitudesB --- Gains]
            ampA_guess = self.PRM[adata]
            ampB_guess = self.PRM[self.Namp + adata]

            # residual between data and guess
            self.Beta[i_meas] = yobs - G*(ampA_guess*LA*PA/K + ampB_guess*LB*PB/K)

            # partial derivitives
            dA = G*LA*PA/K
            dB = G*LB*PB/K

            # store the data in coordinate format for making sparse array
            BIG_col.extend([adata, self.Namp+adata ])
            BIG_row.extend([i_meas] * 2)
            BIG_data.extend([dA, dB])

        # make the big sparse array
        BS = coo_matrix((BIG_data, (BIG_row, BIG_col)),
                        shape=(self.Nmeas, self.Nprm))
        BS = BS.tocsr()  # convert to csr for speed gains?

        # continue with Wolfram notation
        # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html
        b = BS.T.dot(self.Beta)
        A = BS.T.dot(BS)
        a = lsmr(A,b, **kwargs)[0]  # solve

        self.niters += 1
        self.PRM += a  # update
        resid = np.dot(self.Beta, self.Beta)
        self.residuals.append( resid)
        self.BS = BS  # store for looking

        print "Iter %d ; Residual: %e, Press Ctrl-C to break" % (self.niters, resid)

        if self.save_iters:
            np.save("_PRM_iter%d" % self.niters, self.PRM)


FAdat = df.FA.abs().values
FBdat = df.FB.abs().values

UAvals = zip(adata, FAdat)
UBvals = zip(adata, FBdat)

UAvals = sorted( set(UAvals), key=lambda x: x[0])
UBvals = sorted( set(UBvals), key=lambda x: x[0])

Avals = np.array(UAvals)[:,1]
Bvals = np.array(UBvals)[:,1]

AmpA_guess = np.random.uniform(Avals*.1, Avals*10, Namp)**2
AmpB_guess = np.random.uniform(Bvals*.1, Bvals*10, Namp)**2
#AmpA_guess = Avals**2
#AmpB_guess = Bvals**2

#AmpA_guess = np.random.uniform(min(Avals),max(Avals), Namp)
#AmpB_guess = np.random.uniform(min(Bvals),max(Bvals), Namp)
Gain_guess = np.random.uniform(gains.min(), gains.max(), Ngain)  # going in blind here on the gain


#prm_data = np.load("_temp.npz")
#AmpA_guess = prm_data["AmpA_final"]
#AmpB_guess = prm_data["AmpB_final"]
#Gain_guess = prm_data["Gain_final"]


# Parameters array, structured as amplitudes then gains:[AmplitudesA --- AmplitudesB --- Gains]
Amp_PRM = np.hstack((AmpA_guess, AmpB_guess))
Gain_PRM = Gain_guess
#PRM = np.load('_PRM_iter7.npy')

#init_amp_sums = np.array(zip(AmpA_guess[adata]*LAdata, AmpB_guess[adata]*(1-fdata))).sum(axis=1)
#init_yfit = init_amp_sums*np.exp(-xdata**2) * Gain_guess[gdata]

#embed()


import signal

class AlarmException(Exception):
    pass

def alarmHandler(signum, frame):
    raise AlarmException

def nonBlockingRawInput(prompt='', timeout=20):
    signal.signal(signal.SIGALRM, alarmHandler)
    signal.alarm(timeout)
    try:
        text = raw_input(prompt)
        signal.alarm(0)
        return text
    except AlarmException:
        print '\nPrompt timeout. Continuing...'
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return ''

import time
def iter_solver(solver, Niter):
    print "Solverize"
    tstart = time.time()
    try:
        for i in range(Niter):
            solver.iterate()
    except KeyboardInterrupt:
        print "Breaking solver"
        pass
    time_solve = time.time() - tstart
    print "Took %f sec to solve" % time_solve
    return solver

GainData = Gain_guess[gdata]
#rhs = GainData * (AmpA_guess[adata]*LAdata*(PAdata/K) + AmpB_guess[adata]*LBdata*(PBdata/K))
for i_solve in range(Nsolvers):

    print("Starting Amplitude solver!")
    solver = AmpSolver(Amp_PRM, ydata, adata, LAdata, LBdata, PAdata, PBdata,GainData, Namp, K=K)
    solver = iter_solver( solver, Niter)
    Amp_PRM = solver.PRM
    FAdata = Amp_PRM[:Namp][adata]
    FBdata = Amp_PRM[Namp:2*Namp][adata]
    #rhs = GainData * (FAdata*LAdata*(PAdata/K) + FBdata*LBdata*(PBdata/K))

    print("Starting Gain/scale solver!")
    solver = ScaleSolver(Gain_PRM, ydata, gdata,LAdata, LBdata, PAdata, PBdata, FAdata, FBdata, Ngain, K=K)
    solver = iter_solver( solver, Niter)
    Gain_PRM = solver.PRM
    GainData = Gain_PRM[gdata]

    #rhs = GainData * (FAdata*LAdata*(PAdata/K) + FBdata*LBdata*(PBdata/K))


    #stop = raw_input("Do you want to stop? (Y/n)" )
    stop = nonBlockingRawInput("Do you want to stop? (Y/n)" , 5)
    if stop == "Y":
        print("Stopping solverize loop")
        break

AmpA_final = Amp_PRM[:Namp]
AmpB_final = Amp_PRM[Namp:2*Namp]
Gain_final = Gain_PRM

counter = 1
ofile = "_temp"
import os
while os.path.exists(ofile + ".npz"):
    ofile = "_temp_%d"%(counter)
    counter += 1
np.savez(ofile, AmpA_final = AmpA_final, AmpB_final=AmpB_final, Gain_final=Gain_final)

# plot
import pylab as plt

#density = float(solver.BS.count_nonzero()) / np.product(solver.BS.get_shape())
#sparsity = 1 - density

#plt.suptitle("Fit analysis; %d Amplitudes (x2 channels) ; %d measurements\n Sparsity=%.5g (%.2g %% occupancy in Jacobian); %d iters; time per iter: %.2f sec" \
#             % (Namp, Nmeas, sparsity, density*100, solver.niters, time_solve/solver.niters))


#
# PLOT THE Yobs vs Ycalc
# ----------------------
plt.subplot(221)
lhs = ydata
IA = FAdat**2
IB = FBdat**2
rhs_final = Gain_final[gdata] * (AmpA_final[adata]*LAdata*(PAdata/K) + AmpB_final[adata]*LBdata*(PBdata/K))
rhs_guess = Gain_guess[gdata] * (AmpA_guess[adata]*LAdata*(PAdata/K) + AmpB_guess[adata]*LBdata*(PBdata/K))

plt.title("Measurements")
plt.plot( lhs, rhs_guess, '.', ms=.5)
plt.plot( lhs, rhs_final, '.', ms=.5)
plt.xlabel("y_obs")
plt.ylabel("y_calc")
plt.legend(("init guess", "final fit"), markerscale=10)

#
# PLOT THE A-amplitudes
# ---------------------
plt.subplot(222)
plt.title("A amps")
plt.plot(FAdat**2, AmpA_guess[adata],'s', ms=.5)
plt.plot(FAdat**2, AmpA_final[adata],'o', ms=.5)
plt.xlabel("actual value")
plt.ylabel("best fit")
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.legend(("init guess", "final fit"), markerscale=10)


#
# PLOT THE B-amplitudes
# ---------------------
plt.subplot(223)
plt.title("B amps")

plt.plot(FBdat**2, AmpB_guess[adata],'s', ms=.5)
plt.plot(FBdat**2, AmpB_final[adata],'o', ms=.5)
plt.xlabel("data")
plt.ylabel("fit")
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.legend(("init guess", "final fit"), markerscale=10)

#
# PLOT THE GAIN
# -------------
plt.subplot(224)
plt.title("Gains")

plt.plot(df.gain.values, Gain_guess[gdata],'s', ms=.5)
plt.plot(df.gain.values, Gain_final[gdata],'o', ms=.5)
plt.xlabel("data")
plt.ylabel("fit")
plt.legend(("init guess", "final fit"), markerscale=10)


plt.show()

embed()

