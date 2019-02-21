"""
Each measurement samples 1 of many Amplitudes
with 2 colors, and a variable gain
"""
import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix
import sys
from scitbx import lbfgs
from scitbx.array_family import flex

np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )


Nshot = 100  # total number of shots, each will get unique gain value
max_meas_per_shot = 8  # e.g. how many Bragg reflections per shot
min_meas_per_shot = 1
xmin = -.1   # Gaussian amplitudes centered at x=0, so we will sample near peak amplitude
xmax = .1
Namp = 10
gains = np.random.uniform(1,3,Nshot)
AmpsA = np.random.randint( 10,50,Namp)  # channel A amplitude
AmpsB = np.random.uniform(AmpsA*.8,AmpsA*1.2)  # channel B amplitude
Ngain = Nshot

# for each shot, pick some amplitudes and a gain
gdata = []
adata = []
Nmeas_per = []
for i_shot in range( Nshot):
    if i_shot % 50 ==0:
        print "\rBuilding data %d / %d shots" % (i_shot+1, Nshot),
        sys.stdout.flush()  # no new line

    Nmeas_this_shot = np.random.randint(min_meas_per_shot, max_meas_per_shot+1)

    amps = np.random.choice( range(Namp), size=Nmeas_this_shot, replace=False)
    adata.append( amps)

    gain = np.random.choice(range(Ngain))
    gdata.append( [gain] * Nmeas_this_shot)

    Nmeas_per.append( Nmeas_this_shot)
print "\rBuilding data %d / %d shots" % (Nshot, Nshot)

Nmeas = np.sum( Nmeas_per)  # total measurements
gdata = np.hstack( gdata)
adata = np.hstack( adata)
xdata = np.random.uniform(xmin,xmax,Nmeas)

# sum the two channel amplitudes for each measurement
amp_sums = np.array(zip(AmpsA[adata], AmpsB[adata])).sum(axis=1)

# this is the measured intensity for each overlapping amplitude measurement
ydata = np.random.normal( amp_sums*np.exp(-xdata**2)*gains[gdata],0.5)

class LimSolver:
    def __init__(self, guess, x_obs, y_obs, adata, gdata, Namp, Ngain):
        assert x_obs.size == y_obs.size == adata.size == gdata.size
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.adata = adata.astype(int)  # amp parameter ID, e.g. either 0,1,.. Namp, i.e. tells you which Fhkl
        self.gdata = gdata.astype(int)  # gain parameter ID, either 0,1,.. Ngain, i.e. tells you which shot
        self.Namp = Namp  # number of unique Amplitudes (Fhkls)
        self.Ngain = Ngain  # number of unique gain values (shots)

        self.y_diff = np.zeros_like(self.x_obs)  # yobs - ycalc
        self.amps_0_and_1 = np.zeros( (self.Namp, 2))   # two Amplitudes for each observation
        self.exp_factor = np.exp(-xdata**2)  # this is constant throughout
        self.Gvec = flex.double(len(guess))  # for the gradients

        # For inheritance sake
        # To make me a scitbx solver, give me a compute_functional_and_gradients method, that returns a tuple (float, flex.double)
        self.n = len(guess)  # also give me an attribute n,
        self.x = flex.double(np.ascontiguousarray(guess, np.float64))   #  also give me an attribute x, and make it a flex.double
        self.minimizer = lbfgs.run(target_evaluator=self)

    def print_step(pfh,message,target):
        print "%s %10.4f"%(message,target),
        print "["," ".join(["%10.4f" % a for a in pfh.x]),"]"

    def compute_functional_and_gradients(self):
        """returns 2-tuple, functional, gradient"""
        self.set_ydiff()  # NOTE sets attributes y_diff, AmpSum_factor, and G_factor

        f = np.sum(self.y_diff**2)  # single number, value of loss based on self.x
        self.print_step("LBFGS stp",f)

        Grad_gain = -2*np.sum(self.y_diff*self.exp_factor*self.AmpSum_factor)
        Grad_A0 = Grad_A1 = -2*np.sum(self.y_diff*self.exp_factor*self.G_factor)

        for i_a in range(self.Namp):
            self.Gvec[i_a] = Grad_A0
            self.Gvec[self.Namp + i_a] = Grad_A1
        for i_g in range( self.Ngain):
            self.Gvec[2*self.Namp+i_g] = Grad_gain
        return f, self.Gvec

    def set_ydiff(self):

        # first 2*Namp parameters are the amplitudes A,B
        self.amps_0_and_1[:,0] = self.x[:self.Namp].as_numpy_array()
        self.amps_0_and_1[:,1] = self.x[self.Namp:2*self.Namp].as_numpy_array()

        # reverse indexing in numpy to use IDs to construct per observation factors
        self.AmpSum_factor = self.amps_0_and_1.sum(axis=1)[self.adata]

        # the remaining parameters are the gains
        self.G_factor = self.x[2*self.Namp:].as_numpy_array()[self.gdata]

        self.y_calc = self.AmpSum_factor* self.exp_factor * self.G_factor

        self.y_diff = self.y_obs - self.y_calc



AmpA_guess = np.random.uniform(AmpsA*.1, AmpsA*3)  # assume we have a loose idea on the structure factors goig in
AmpB_guess = np.random.uniform(AmpsB*.1, AmpsB*3)
Gain_guess = np.random.uniform(1, 2, Ngain)  # going in blind here on the gain
PRM = np.hstack((AmpA_guess, AmpB_guess, Gain_guess))

init_amp_sums = np.array(zip(AmpA_guess[adata], AmpB_guess[adata])).sum(axis=1)
init_yfit = init_amp_sums*np.exp(-xdata**2) * Gain_guess[gdata]

import time
tstart = time.time()
solver = LimSolver(guess=PRM, x_obs=xdata, y_obs=ydata,
                   gdata=gdata, adata=adata, Namp=Namp, Ngain=Ngain)
time_solve = time.time() - tstart


AmpA_final = solver.x[:Namp].as_numpy_array()
AmpB_final = solver.x[Namp:2*Namp].as_numpy_array()
Gain_final = solver.x[2*Namp:].as_numpy_array()
final_amp_sums = np.array(zip(AmpA_final[adata], AmpB_final[adata])).sum(axis=1)
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


fig,axs = plt.subplots(1,3, figsize=(10,4)) #plt.figure()
plt.suptitle("Fit analysis; %d Amplitudes (x2 channels) and %d gains (shots); %d measurements\n time to solve: %.2f sec" \
             % (Namp, Ngain, Nmeas, time_solve))

axs[0].set_title("$(A_A + A_B) * G $")
axs[0].plot(amp_sums*gains[gdata],
            init_amp_sums*Gain_guess[gdata],'.', ms=.5)
axs[0].plot(amp_sums*gains[gdata],
            final_amp_sums*Gain_final[gdata],'.', ms=.5)
axs[0].set_xlabel("data")
axs[0].set_ylabel("fit")
axs[0].legend(("init guess", "final fit"), markerscale=10)


axs[1].set_title("$(A_A)$")
axs[1].plot(AmpsA, AmpA_guess,'s', ms=4)
axs[1].plot(AmpsA, AmpA_final,'o', ms=4)
axs[1].set_xlabel("data")
axs[1].set_ylabel("fit")
axs[1].legend(("init guess", "final fit"))

axs[2].set_title("$(A_B)$")
axs[2].plot(AmpsB, AmpB_guess,'s', ms=4)
axs[2].plot(AmpsB, AmpB_final,'o', ms=4)
axs[2].set_xlabel("data")
axs[2].set_ylabel("fit")
axs[2].legend(("init guess", "final fit"))

plt.subplots_adjust(left=.07, right=.95, top=.8, wspace=.3)
plt.show()

