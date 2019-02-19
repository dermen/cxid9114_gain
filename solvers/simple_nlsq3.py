"""
Sample sum of 2 Gaussians
 with variable gain
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
Amps = np.array([10, 25])
Ngain = gains.shape[0]
Namp = Amps.shape[0]
Niter = 1000
method = "lsmr"
# parameters are the amplitudes and the gains

gdata = np.random.randint(0,Ngain,Nmeas)
xdata = np.random.uniform(xmin,xmax,Nmeas)

# numpy voodoo, just evaluates ydata for each measurement, given gain and xdata
ydata = np.random.normal( np.sum(Amps)*np.exp(-xdata**2)*gains[gdata],0.5)

class TestSolver:
    def __init__(self, guess, xvals, yvals, gvals):
        self.PRM = np.array(guess, float)
        self.xvals = xvals
        self.yvals = yvals
        self.gvals = gvals

        self.exp_factor = np.exp(-xdata**2)  # this is constant throughout

        self.Nmeas = len(yvals)
        self.Nprm = len( guess)

        self.Beta = np.zeros_like(yvals)
        self.BIG = np.zeros((self.Nmeas, self.Nprm))


    def iterate(self, method='lsmr', **kwargs):

        for i_meas, (xdata,ydata, gdata) in enumerate(
                zip(self.xvals, self.yvals, self.gvals)):

            amp0_guess, amp1_guess =  self.PRM[:2]

            # gdata is either 0, 1 or 2
            gain_guess = self.PRM[2 + gdata]

            self.Beta[i_meas] = ydata - (amp0_guess+amp1_guess) * gain_guess * self.exp_factor[i_meas]

            self.BIG[i_meas,0] = amp1_guess*gain_guess * self.exp_factor[i_meas]
            self.BIG[i_meas,1] = amp0_guess*gain_guess * self.exp_factor[i_meas]
            self.BIG[i_meas,2+gdata] = (amp0_guess + amp1_guess) * self.exp_factor[i_meas]


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

PRM = np.array([8,15, .9, 1.1, 2])  # initial guess: Amp0,Amp1, gain0, gain1, gain2

init_amps = PRM[:2]
init_gains = PRM[2:]
init_yfit = np.sum(init_amps)*np.exp(-xdata**2) * init_gains[gdata]

solver = TestSolver(PRM, xdata, ydata, gdata)
for i in range(Niter):
    try:
        solver.iterate(method)
    except (np.linalg.LinAlgError):
        break

final_amp = solver.PRM[:2]
final_gains = solver.PRM[2:]
final_yfit = np.sum(final_amp)*np.exp(-xdata**2)*final_gains[gdata]

print "\n\n++++++++++++++"
print "AmpSum*Gain truth: " ,np.round (np.sum(Amps) * gains,2)
print "AmpSum*gain fit: " ,np.round( np.sum(final_amp) * final_gains,2)
print "Amps truth: ", Amps
print "Amps fit: ", final_amp
print "Did I uncouple the gain and amplitude?"
print


# plot
import pylab as plt
for i_g in range(Ngain):
    plt.figure(i_g)
    plt.title("shot %d" % i_g)
    sel = gdata==i_g

    O = np.argsort( xdata[sel])  # just for line plotting
    plt.plot( xdata[sel][O], init_yfit[sel][O] ,c='k')
    plt.plot( xdata[sel][O], final_yfit[sel][O], 'k--')
    plt.plot( xdata[sel], ydata[sel], 'k.')

    plt.legend(("init","final","data"))

plt.show()

