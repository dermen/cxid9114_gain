"""
Each measurement samples 1 of many Amplitudes
with 2 colors, and a variable gain
"""
import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix
import sys

np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

build_data = True
save_data = False

Niter = 100
method = "lsmr"

if build_data:
    Nshot = 50000  # total number of shots, each will get unique gain value
    max_meas_per_shot = 70 # e.g. how many Bragg reflections per shot
    min_meas_per_shot = 10
    xmin = -.1   # Gaussian amplitudes centered at x=0, so we will sample near peak amplitude
    xmax = .1
    Namp = 8000
    gains = np.random.uniform(1,3,Nshot)  # gains per shot
    AmpsA = np.random.uniform( 10,100,Namp)  # channel A amplitude
    AmpsB = np.random.uniform(AmpsA*.8,AmpsA*1.2)  # channel B amplitude offset slightly from channel A (e.g anomalous)
    Ngain = Nshot

    # parameters are the amplitudes and the gains

    # for each shot, pick some amplitudes and a gain
    gdata = []
    adata = []
    fdata = []
    Nmeas_per = []
    Nmeas_per_shot = np.random.randint(min_meas_per_shot, max_meas_per_shot, Nshot)  # measured amplitudes per shot
    chan_frac_per_shot = np.random.uniform(0, 1, Nshot)  # color channel per shot, e.g. FEE spec
    for i_shot in range( Nshot):
        if i_shot % 100 ==0:
            print "\rBuilding data %d / %d shots" % (i_shot+1, Nshot),
            sys.stdout.flush()  # no new line

        # choose which amplitudes are illuminated on each shot, each amp is measured 0 or 1 times per shot
        amps = np.random.choice( range(Namp), size=Nmeas_per_shot[i_shot], replace=False)
        adata.append( amps)

        gdata.append( [i_shot] * Nmeas_per_shot[i_shot])

        # its crucial that each color channel get a unique per shot adjustment
        # else the partial derivitives for either channel will be identical and the minimizer
        # wont be able to disentangle the two amplitudes...
        fdata.append( [chan_frac_per_shot[i_shot]] * Nmeas_per_shot[i_shot])

    print "\rBuilding data %d / %d shots" % (Nshot, Nshot)

    print "Combining data"
    Nmeas = np.sum( Nmeas_per_shot)  # total measurements
    gdata = np.hstack( gdata)
    adata = np.hstack( adata)
    fdata = np.hstack(fdata)
    xdata = np.random.uniform(xmin,xmax,Nmeas)

    # sum the two channel amplitudes for each measurement
    amp_sums = np.array(zip(AmpsA[adata]*fdata, AmpsB[adata]*(1-fdata))).sum(axis=1)

    # this is the measured intensity for each overlapping amplitude measurement
    print "adding gaussian noise to measurements"
    ydata = np.random.normal( amp_sums*np.exp(-xdata**2)*gains[gdata],0.5)

    if save_data:
        print("Saving...")
        np.savez_compressed(
            "_data_nlsq_data5", ydata=ydata,xdata=xdata,gdata=gdata,
            adata=adata,fdata=fdata,Namp=Namp,Ngain=Ngain,amp_sums=amp_sums,
            AmpsA=AmpsA, AmpsB=AmpsB, gains=gains)

else:
    data=np.load("_data_nlsq_data5.npz")
    ydata=data['ydata']
    xdata=data['xdata']
    adata=data['adata']
    gdata=data['gdata']
    fdata=data['fdata']
    Namp=data['Namp']
    Ngain=data['Ngain']
    amp_sums=data['amp_sums']
    AmpsA=data['AmpsA']
    AmpsB=data['AmpsB']
    gains=data['gains']
    Nmeas=len(ydata)

class TestSolver:
    def __init__(self, guess, xvals, yvals, gvals, avals, fvals, Namp, Ngain, save_iters=True):
        self.PRM = np.array(guess, float)
        self.xvals = xvals
        self.yvals = yvals
        self.gvals = gvals
        self.avals = avals
        self.fvals = fvals
        self.Namp = Namp
        self.Ngain = Ngain

        self.exp_factor = np.exp(-xdata**2)  # this is constant throughout

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.niters = 0
        self.residuals = []
        self.save_iters = save_iters

    def iterate(self, **kwargs):

        BIG_row = []
        BIG_col =  []
        BIG_data = []
        for i_meas, (xdata, ydata, gdata, adata, fdata) in enumerate(
                zip(self.xvals, self.yvals, self.gvals, self.avals, self.fvals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA --- AmplitudesB --- Gains]
            ampA_guess = self.PRM[adata]
            ampB_guess = self.PRM[self.Namp + adata]
            gain_guess = self.PRM[2*self.Namp + gdata]

            # residual between data and guess
            fA = fdata
            fB = 1-fdata
            self.Beta[i_meas] = ydata - (ampA_guess*fA + ampB_guess*fB) * gain_guess * self.exp_factor[i_meas]

            # partial derivitives
            dA = gain_guess * fA*self.exp_factor[i_meas]
            dB = gain_guess * fB*self.exp_factor[i_meas]
            dG = (ampA_guess*fA + ampB_guess*fB) * self.exp_factor[i_meas]

            # store the data in coordinate format for making sparse array
            BIG_col.extend([adata, self.Namp+adata, 2*self.Namp + gdata])
            BIG_row.extend([i_meas] * 3)
            BIG_data.extend([dA, dB, dG])

        # make the big sparse array
        BS = coo_matrix((BIG_data, (BIG_row,BIG_col)),
                        shape=(self.Nmeas, self.Nprm))
        BS = BS.tocsr()  # convert to csr for gains?

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

AmpA_guess = np.random.uniform(AmpsA*.1, AmpsA*3)  # assume we have a loose idea on the structure factors goig in
AmpB_guess = np.random.uniform(AmpsB*.1, AmpsB*3)
Gain_guess = np.random.uniform(1, 2, Ngain)  # going in blind here on the gain

# Parameters array, structured as amplitudes then gains:[AmplitudesA --- AmplitudesB --- Gains]
PRM = np.hstack((AmpA_guess, AmpB_guess, Gain_guess))
#PRM = np.load('_PRM_iter16.npy')

init_amp_sums = np.array(zip(AmpA_guess[adata]*fdata, AmpB_guess[adata]*(1-fdata))).sum(axis=1)
init_yfit = init_amp_sums*np.exp(-xdata**2) * Gain_guess[gdata]

solver = TestSolver(PRM, xdata, ydata, gdata, adata,fdata, Namp, Ngain)
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
Gain_final = solver.PRM[2*Namp:]
final_amp_sums = np.array(zip(AmpA_final[adata]*fdata, AmpB_final[adata]*(1-fdata))).sum(axis=1)
final_yfit = final_amp_sums*np.exp(-xdata**2)*Gain_final[gdata]

print "\n\n++++++++++++++"
#print "AmpSum*Gain truth: " ,np.round (np.sum(Amps) * gains,2)
#print "AmpSum*gain fit: " ,np.round( np.sum(final_amp) * final_gains,2)
print "AmpsA truth: ", AmpsA
print "AmpsA fit: ", AmpA_final
print "AmpsB truth: ", AmpsB
print "AmpsB fit: ", AmpB_final
print "Did I uncouple the gain and amplitude?"
print

# plot
import pylab as plt

density = float(solver.BS.count_nonzero()) / np.product(solver.BS.get_shape())
sparsity = 1 - density

fig,axs = plt.subplots(1,3, figsize=(10,4)) #plt.figure()
plt.suptitle("Fit analysis; %d Amplitudes (x2 channels) and %d gains (shots); %d measurements\n Sparsity=%.5g (%.2g %% occupancy in Jacobian); %d iters; time per iter: %.2f sec" \
             % (Namp, Ngain, Nmeas, sparsity, density*100, solver.niters, time_solve/solver.niters))

axs[0].set_title("$(A_if_i + B_i(1-f_i)) G_i $")
axs[0].plot(amp_sums*gains[gdata],
            init_amp_sums*Gain_guess[gdata],'.', ms=.5)
axs[0].plot(amp_sums*gains[gdata],
            final_amp_sums*Gain_final[gdata],'.', ms=.5)
axs[0].set_xlabel("data")
axs[0].set_ylabel("fit")
axs[0].legend(("init guess", "final fit"), markerscale=10)


axs[1].set_title("$A_i$")
axs[1].plot(AmpsA, AmpA_guess,'s', ms=.5)
axs[1].plot(AmpsA, AmpA_final,'o', ms=.5)
axs[1].set_xlabel("data")
axs[1].set_ylabel("fit")
axs[1].legend(("init guess", "final fit"), markerscale=10)

axs[2].set_title("$B_i$")
axs[2].plot(AmpsB, AmpB_guess,'s', ms=.5)
axs[2].plot(AmpsB, AmpB_final,'o', ms=.5)
axs[2].set_xlabel("data")
axs[2].set_ylabel("fit")
axs[2].legend(("init guess", "final fit"), markerscale=10)

plt.subplots_adjust(left=.07, right=.95, top=.8, wspace=.3)
plt.show()

