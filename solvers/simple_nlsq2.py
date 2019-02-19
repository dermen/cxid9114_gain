"""
Same as simple_nlsq.py, however we
have 3 Gaussians that can be sampled
"""
import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix

np.random.seed(hash("no problem is insoluble in all conceivable circumstances")&((1<<32)-1) )

Nmeas = 500
xmin = -2
xmax = 2
gains = np.array([1.,1.25, 1.5])
Amps = np.array([10,25,30])
Ngain = gains.shape[0]
Namp = Amps.shape[0]
Niter = 10
method = "lsmr"
# parameters are the amplitude and the gains

adata = np.random.randint( 0, Namp, Nmeas)
gdata = np.random.randint(0,Ngain,Nmeas)
xdata = np.random.uniform(xmin,xmax,Nmeas)
ydata = np.random.normal(Amps[adata] * np.exp(-xdata**2)*gains[gdata],0.5)


class TestSolver:
    def __init__(self, guess, xvals, yvals, gvals, avals, Namp, Ngauss):
        self.PRM = np.array(guess, float)
        self.xvals = xvals
        self.yvals = yvals
        self.gvals = gvals
        self.avals = avals

        self.exp_factor = np.exp(-xdata**2)  # this is constant throughout

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.BIG = np.zeros((self.Nmeas, self.Nprm))

        self.Namp = Namp
        self.Gauss = Ngauss

    def iterate(self, method='lsmr', **kwargs):
        for i_meas, (xdata,ydata, gdata, adata) in enumerate(
                zip(self.xvals, self.yvals, self.gvals, self.avals)):

            # adata is either 0, 1 or 2
            amp_guess = self.PRM[adata]

            # gdata is either 0, 1 or 2
            gain_guess = self.PRM[self.Namp + gdata]

            self.Beta[i_meas] = ydata - amp_guess * gain_guess * self.exp_factor[i_meas]

            self.BIG[i_meas,adata] = gain_guess * self.exp_factor[i_meas]
            self.BIG[i_meas,self.Namp+gdata] = amp_guess* self.exp_factor[i_meas]

        # continue with Wolfram notation
        # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html
        b = np.dot(self.BIG.T, self.Beta)
        A = np.dot(self.BIG.T, self.BIG)
        if method == 'lsmr':
            A = coo_matrix(A)  # this will eventually be built instead of self.BIG
            a = lsmr(A,b, **kwargs)[0]

        elif method == 'direct':
            a = np.dot(np.linalg.inv(A), b)

        print self.PRM,
        self.PRM += a  # update
        print "Residual: %.2e" % np.dot(self.Beta, self.Beta)

PRM = np.array([10.5, 15,20,1., 1.1, 1.2])  # initial guess: Amp0,Amp1, Amp2, gain0, gain1, gain2

init_amp = PRM[:Namp]
init_gains = PRM[Namp:]
init_yfit = init_amp[adata]*np.exp(-xdata**2) * init_gains[gdata]

solver = TestSolver(PRM, xdata, ydata, gdata, adata, Namp, Ngain)
for i in range(Niter):
    try:
        solver.iterate(method)
    except (np.linalg.LinAlgError):
        break

final_amp = solver.PRM[:Namp]
final_gains = solver.PRM[Namp:]
final_yfit = final_amp[adata]*np.exp(-xdata**2)*final_gains[gdata]

print "\n\n++++++++++++++"
print "Amp*Gain truth: " ,np.round (Amps * gains,2)
print "Amp*gain fit: " ,np.round( final_amp * final_gains,2)
print "Amps truth: ", Amps
print "Amps fit: ", final_amp
print "Did I uncouple the gain and amplitude?"
print

# plot
colors = ('salmon', 'darkblue', 'forestgreen')
import pylab as plt
for i_g in range(Ngain):
    plt.figure(i_g)
    plt.title("shot %d" % i_g)
    for i_a in range( Namp):
        C = colors[i_a]
        sel = np.logical_and(gdata==i_g, adata==i_a)
        O = np.argsort( xdata[sel])  # just for line plotting
        plt.plot( xdata[sel][O], init_yfit[sel][O], color=C, label='ini%d' % i_a )
        plt.plot( xdata[sel][O], final_yfit[sel][O],ls='--',color=C, label='fin%d' % i_a )
        plt.plot( xdata[sel], ydata[sel], '.', color=C, label='dat%d' % i_a )
    plt.legend()

plt.show()





