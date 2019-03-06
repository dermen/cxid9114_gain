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

Niter = 200
K = 10000**2 * 1e12
method = "lsmr"

#Nmeas = 2500000
df = pandas.read_hdf("r62_simdata2_fixed_oversamp_labeled.hdf5","data")
#df.reset_index(inplace=True)

Ndata_total = len(df)
Nmeas = len(df)

#vals = abs(df.lhs  - df.rhs)
#df = df.loc[vals < 1e-7]  # only select those shots with overlaps

adata = df.adata.values
gdata = df.gdata.values
LAdata = df.LA.values
LBdata = df.LB.values
PAdata = df.PA.values
PBdata = df.PB.values
gains = df.gain.values
ydata = df.D.values / gains

# remap adata and gdata
amp_remap = {a:i_a for i_a, a in enumerate(set(adata))}
adata = np.array([amp_remap[a] for a in adata])

Nmeas = len( ydata)
Namp = np.unique(adata).shape[0]
print "N-unknowns: 2xNhkl = %d unknowns," % (2*Namp)
print "N-measurements: %d" % Nmeas


class TestSolver:
    def __init__(self, guess, yvals, avals, LAvals, LBvals, PAvals, PBvals, Namp, save_iters=True, K=1e20):
        self.PRM = np.array(guess, float)
        self.yvals = yvals
        self.avals = avals
        self.LAvals = LAvals
        self.LBvals = LBvals
        self.PAvals = PAvals
        self.PBvals = PBvals
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
        for i_meas, (yobs, adata, LA, LB, PA, PB) in enumerate(
                izip(self.yvals, self.avals, self.LAvals, self.LBvals, self.PAvals, self.PBvals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA --- AmplitudesB --- Gains]
            ampA_guess = self.PRM[adata]
            ampB_guess = self.PRM[self.Namp + adata]

            # residual between data and guess
            self.Beta[i_meas] = yobs - (ampA_guess*LA*PA/K + ampB_guess*LB*PB/K)

            # partial derivitives
            dA = LA*PA/K
            dB = LB*PB/K

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

AmpA_guess = np.random.uniform(Avals*.1, Avals*3, Namp)**2
AmpB_guess = np.random.uniform(Bvals*.1, Bvals*3, Namp)**2
#AmpA_guess = Avals**2
#AmpB_guess = Bvals**2

#AmpA_guess = np.random.uniform(min(Avals),max(Avals), Namp)
#AmpB_guess = np.random.uniform(min(Bvals),max(Bvals), Namp)
#Gain_guess = np.random.uniform(gains.min(), gains.max(), Ngain)  # going in blind here on the gain

# Parameters array, structured as amplitudes then gains:[AmplitudesA --- AmplitudesB --- Gains]
PRM = np.hstack((AmpA_guess, AmpB_guess))

#PRM = np.load('_PRM_iter7.npy')

#init_amp_sums = np.array(zip(AmpA_guess[adata]*LAdata, AmpB_guess[adata]*(1-fdata))).sum(axis=1)
#init_yfit = init_amp_sums*np.exp(-xdata**2) * Gain_guess[gdata]

#embed()

solver = TestSolver(PRM, ydata, adata,LAdata, LBdata, PAdata, PBdata, Namp, K=K)
import time
print "Solverize"

tstart = time.time()
try:
    for i in range(Niter):
        solver.iterate()
except KeyboardInterrupt:
    print "Breaking solver"
    pass

time_solve = time.time() - tstart

AmpA_final = solver.PRM[:Namp]
AmpB_final = solver.PRM[Namp:2*Namp]
#Gain_final = solver.PRM[2*Namp:]


#final_amp_sums = np.array(zip(AmpA_final[adata]*fdata, AmpB_final[adata]*(1-fdata))).sum(axis=1)
#final_yfit = final_amp_sums*np.exp(-xdata**2)*Gain_final[gdata]

# plot
import pylab as plt

density = float(solver.BS.count_nonzero()) / np.product(solver.BS.get_shape())
sparsity = 1 - density

plt.suptitle("Fit analysis; %d Amplitudes (x2 channels) ; %d measurements\n Sparsity=%.5g (%.2g %% occupancy in Jacobian); %d iters; time per iter: %.2f sec" \
             % (Namp, Nmeas, sparsity, density*100, solver.niters, time_solve/solver.niters))

plt.plot(FAdat**2, AmpA_guess[adata],'s', ms=.5)
plt.plot(FAdat**2, AmpA_final[adata],'o', ms=.5)
plt.xlabel("data")
plt.ylabel("fit")
plt.legend(("init guess", "final fit"), markerscale=10)

plt.figure()
lhs = ydata
IA = FAdat**2
IB = FBdat**2
rhs_final = (AmpA_final[adata]*LAdata*(PAdata/K) + AmpB_final[adata]*LBdata*(PBdata/K))
rhs_guess = (AmpA_guess[adata]*LAdata*(PAdata/K) + AmpB_guess[adata]*LBdata*(PBdata/K))

plt.plot( lhs, rhs_guess, '.', ms=.5)
plt.plot( lhs, rhs_final, '.', ms=.5)
plt.xlabel("y_obs")
plt.ylabel("y_calc")
plt.legend(("init guess", "final fit"), markerscale=10)

plt.figure()
plt.suptitle("B amps")

plt.plot(FBdat**2, AmpB_guess[adata],'s', ms=.5)
plt.plot(FBdat**2, AmpB_final[adata],'o', ms=.5)
plt.xlabel("data")
plt.ylabel("fit")
plt.legend(("init guess", "final fit"), markerscale=10)

plt.show()

