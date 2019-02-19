import numpy as np
from IPython import embed
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix

Nmeas = 200
xmin = -2
xmax = 2
gains = np.array([1.,1.25, 1.5])
Amp = 10
Ngain = gains.shape[0]
Niter = 10
method = "lsmr"
# parameters are the amplitude and the gains

# gain assingments: tells us which gain level to assign xdata,ydata
gdata = np.random.randint(0,Ngain,Nmeas)  # takes on values 0, 1, or 2
xdata = np.random.uniform(xmin,xmax,Nmeas)
ydata = np.random.normal(Amp*np.exp(-xdata**2)*gains[gdata],0.5)


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

            # gdata is either 0, 1, or 2
            amp_guess = self.PRM[0]
            gain_guess = self.PRM[1 + gdata]
            #gain_guess=1

            self.Beta[i_meas] = ydata - amp_guess * gain_guess * self.exp_factor[i_meas]

            self.BIG[i_meas,0] = gain_guess * self.exp_factor[i_meas]
            self.BIG[i_meas,1+gdata] = amp_guess* self.exp_factor[i_meas]

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

PRM = np.array([10.5, 1., 1, 1])  # initial guess: Amplitude, gain0, gain1, gain2

init_gains = PRM[1:]
init_amp = PRM[0]
init_yfit = init_amp*np.exp(-xdata**2) * init_gains[gdata]

solver = TestSolver(PRM, xdata, ydata, gdata)
for i in range(Niter):
    try:
        solver.iterate(method)
    except (np.linalg.LinAlgError):
        break

final_gains = solver.PRM[1:]
final_amp = solver.PRM[0]
final_yfit = final_amp*np.exp(-xdata**2)*final_gains[gdata]

print "\n\n++++++++++++++"
print "Amp*Gain truth: " ,np.round (Amp * gains,2)
print "Amp*gain fit: " ,np.round( final_amp * final_gains,2)
print "Amp: ", Amp
print "Amp fit: ", final_amp
print "Did I uncouple the gain and amplitude?"
print
# plot
import pylab as plt
for i in range(Ngain):
    plt.figure(i)
    sel = gdata==i
    plt.title("shot %d" % i)
    plt.plot( xdata[sel], init_yfit[sel],'.' )
    plt.plot( xdata[sel], final_yfit[sel],'d' )
    plt.plot( xdata[sel], ydata[sel],'.' )
    plt.legend(("init","final","data"))

plt.show()





