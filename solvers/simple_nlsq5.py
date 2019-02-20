"""
Each measurement samples 1 of many Amplitudes
with 2 colors, and a variable gain
"""
import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix

np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

Nshot = 1000
max_meas_per_shot = 40
min_meas_per_shot = 10
xmin = -.1
xmax = .1
Namp = 100
gains = np.random.uniform(1,3,Nshot)
AmpsA = np.random.randint( 10,50,Namp)  # channel A amplitude
AmpsB = np.random.uniform(AmpsA*.8,AmpsA*1.2)  # channel B amplitude
Ngain = Nshot


Niter = 10
method = "lsmr"
# parameters are the amplitudes and the gains

# for each shot, pick some amplitudes and a gain
gdata = []
adata = []
Nmeas = 0
for i_shot in range( Nshot):
    if i_shot % 10 ==0:
        print i_shot, Nshot
    Nmeas_this_shot = np.random.randint(min_meas_per_shot, max_meas_per_shot+1)
    amps = np.random.choice( range(Namp), size=Nmeas_this_shot, replace=False)
    adata.append( amps)

    gain = np.random.choice(range(Ngain))
    gdata.append( [gain] * Nmeas_this_shot)
    Nmeas += Nmeas_this_shot

gdata = np.hstack( gdata)
adata = np.hstack( adata)
#gdata = np.random.randint(0,Ngain,Nmeas)
#adata = np.random.randint( 0, Namp, Nmeas)
xdata = np.random.uniform(xmin,xmax,Nmeas)

# sum the two channel amplitudes for each measurement
amp_sums = np.array(zip(AmpsA[adata], AmpsB[adata])).sum(axis=1)

# this is the measured intensity for each overlapping amplitude measurement
ydata = np.random.normal( amp_sums*np.exp(-xdata**2)*gains[gdata],0.5)

class TestSolver:
    def __init__(self, guess, xvals, yvals, gvals, avals, Namp, Ngain):
        self.PRM = np.array(guess, float)
        self.xvals = xvals
        self.yvals = yvals
        self.gvals = gvals
        self.avals = avals

        self.Namp = Namp
        self.Ngain = Ngain

        self.exp_factor = np.exp(-xdata**2)  # this is constant throughout

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.niters = 0
        self.residuals = []

    def iterate(self, **kwargs):

        BIG_row = []
        BIG_col =  []
        BIG_data = []
        for i_meas, (xdata,ydata, gdata, adata) in enumerate(
                zip(self.xvals, self.yvals, self.gvals, self.avals)):

            # get the parameters from the pre-structured parameters array
            # which in this case is [AmplitudesA, AmplitudesB, Gains]
            ampA_guess = self.PRM[adata]
            ampB_guess = self.PRM[self.Namp + adata]
            gain_guess = self.PRM[2*self.Namp + gdata]

            # residual between data and guess
            self.Beta[i_meas] = ydata - (ampA_guess+ampB_guess) * gain_guess * self.exp_factor[i_meas]

            # partial derivitives
            dA = gain_guess * self.exp_factor[i_meas]
            dB = gain_guess * self.exp_factor[i_meas]
            dG = (ampA_guess + ampB_guess) * self.exp_factor[i_meas]

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
        print "Iter %d ; Residual: %e" % (self.niters, resid)
        self.BS = BS  # store for looking

AmpA_guess = np.random.uniform(AmpsA*.1, AmpsA*3)  # assume we have a loose idea on the structure factors goig in
AmpB_guess = np.random.uniform(AmpsB*.1, AmpsB*3)
Gain_guess = np.random.uniform(1, 2, Ngain)  # going in blind here on the gain
PRM = np.hstack((AmpA_guess, AmpB_guess, Gain_guess))

init_amp_sums = np.array(zip(AmpA_guess[adata], AmpB_guess[adata])).sum(axis=1)
init_yfit = init_amp_sums*np.exp(-xdata**2) * Gain_guess[gdata]

solver = TestSolver(PRM, xdata, ydata, gdata, adata, Namp, Ngain)
for i in range(Niter):
    solver.iterate()

AmpA_final = solver.PRM[:Namp]
AmpB_final = solver.PRM[Namp:2*Namp]
Gain_final = solver.PRM[2*Namp:]
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
plt.suptitle("Fit analysis; %d Amplitudes (x2 channels) and %d gains (shots); %d measurements" \
             % (Namp, Ngain, Nmeas))

axs[0].set_title("$(A_A + A_B) * G $")
axs[0].plot(amp_sums*gains[gdata],
            init_amp_sums*Gain_guess[gdata],'.', ms=.9)
axs[0].plot(amp_sums*gains[gdata],
            final_amp_sums*Gain_final[gdata],'.', ms=.9)
axs[0].set_xlabel("data")
axs[0].set_ylabel("fit")
axs[0].legend(("init guess", "final fit"), markerscale=10)


axs[1].set_title("$(A_A)$")
axs[1].plot(AmpsA, AmpA_guess,'s', )
axs[1].plot(AmpsA, AmpA_final,'o', )
axs[1].set_xlabel("data")
axs[1].set_ylabel("fit")
axs[1].legend(("init guess", "final fit"))

axs[2].set_title("$(A_B)$")
axs[2].plot(AmpsB, AmpB_guess,'s', )
axs[2].plot(AmpsB, AmpB_final,'o', )
axs[2].set_xlabel("data")
axs[2].set_ylabel("fit")
axs[2].legend(("init guess", "final fit"))

plt.subplots_adjust(left=.07, right=.95, top=.85, wspace=.3)
plt.show()

