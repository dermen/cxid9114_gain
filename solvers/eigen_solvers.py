import gen_data
import numpy as np
from cxid9114.solvers import solvers
import cxid9114
from scitbx.lstbx import normal_eqns_solving
from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common
import time
import pylab as plt
from IPython import embed


class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):
  def __init__(self, initial_estimates, Nhkl=None, plot=False, truth=None):
    super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
    self.Nhkl = Nhkl
    self.initialize(initial_estimates)
    self.stored_functional = []
    self.truth = truth
    self.plot = plot
    if plot:

      self.fig, (self.ax1,self.ax2,self.ax3) = plt.subplots(
          nrows=1,
          ncols=3,
          figsize=(10, 4))
      if self.truth is not None:
          self.fig2, self.f2_ax = plt.subplots(1,1)


  def build_up(self, objective_only=False):
    if not objective_only:
      self.counter+=1

    self.reset()
    if not objective_only:
      functional = self.functional(self.x)
      print("\n\t<><><><><><>")
      print("Functional value: %.4e" % functional)
      if self.truth is not None:
        functional_ideal = self.functional(self.truth)
        print("\tIdeal Functional value: %.4e" % functional_ideal)
      print("\t<><><><><><>\n")

      if self.Nhkl is not None:
        FA = self.x[:self.Nhkl]
        FB = self.x[self.Nhkl:2*self.Nhkl]
        G = self.x[2*self.Nhkl:]
        print("Max FA: %.4e   Min FA: %.4e " % (max(FA), min(FA)))
        print("Max FB: %.4e   Min FB: %.4e " % (max(FB), min(FB)))
        print("Max ScaleFactor: %.2e   Min ScaleFactor: %.2e " % (max(G), min(G)))
        if self.plot:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax1.set_xlabel("log |FA|^2")
            self.ax2.set_xlabel("log |FB|^2")
            self.ax3.set_xlabel("log |Scale factor|")
            FAmax = max(FA)
            FAmin = min(FA)
            FBmax = max(FB)
            FBmin = min(FB)
            Gmax = max(G)
            Gmin = min(G)
            FArng = np.linspace(FAmin, FAmax, 150)
            FBrng = np.linspace(FBmin, FBmax, 150)
            Grng = np.linspace(Gmin, Gmax, 150)
            self.ax1.hist(FA,bins=FArng, log=True)
            self.ax2.hist(FB, bins=FBrng, log=True)
            self.ax3.hist(G, bins=Grng, log=True)
            self.fig.canvas.draw()
            if self.truth is not None:
                self.f2_ax.clear()
                self.f2_ax.plot(self.truth[:len(FA)], FA, '.', ms=0.2)
                self.fig2.canvas.draw()
            plt.pause(0.3)


      self.stored_functional.append(functional)
    self.access_cpp_build_up_directly_eigen_eqn(objective_only, current_values = self.x)


class eigen_solver(solvers.LBFGSsolver):

  def __init__(self, conj_grad=True, plot=False, plot_truth=True, *args, **kwargs):
    solvers.LBFGSsolver.__init__(self, *args, **kwargs)  # NOTE: do it with lbfgs=False
    # ^ brings in Yobs, LA, LB, PA, PB, Nhkl, Ns, Nmeas,   Aidx, Gidx

    # correct because working with logs
    if self.IAprm_truth is not None:
        self.IAprm_truth = flex.log(self.IAprm_truth)
        self.IBprm_truth = flex.log(self.IBprm_truth)
        self.Gprm_truth = flex.log(self.Gprm_truth)
        self.x_truth = (self.IAprm_truth.concatenate(self.IBprm_truth)).concatenate(self.Gprm_truth)

    self.x_init = flex.double(np.ascontiguousarray(self.guess["IAprm"])).concatenate(
        flex.double(np.ascontiguousarray(self.guess["IBprm"]))).concatenate(flex.double(np.ascontiguousarray(self.guess["Gprm"])))
    assert (len(self.x_init) == self.Nhkl * 2 + self.Ns)

    IAx = flex.log(self.x_init[:self.Nhkl])
    IBx = flex.log(self.x_init[self.Nhkl:2 * self.Nhkl])
    Gx = flex.log(self.x_init[2 * self.Nhkl:]) - 1

    self.x_init = IAx.concatenate(IBx)
    self.x_init = self.x_init.concatenate(Gx)

    self.counter = 0

    # set dummie weights for now
    self.Wobs = flex.double(np.ones(len(self.Yobs)))
    if plot_truth:
        try:
            truth = self.x_truth
        except AttributeError as error:
            print(error)
            truth = None

    self.helper = eigen_helper(initial_estimates=self.x_init, Nhkl=self.Nhkl, plot=plot, truth=truth)
    self.helper.eigen_wrapper.conj_grad = conj_grad
    self.helper.set_cpp_data(
      self.Yobs, self.Wobs, self.Aidx, self.Gidx, self.PA, self.PB, self.LA, self.LB, self.Nhkl, self.Ns)

    self.helper.restart()
    _ = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
               non_linear_ls=self.helper,
               n_max_iterations=200,
               track_all=True,
               step_threshold=0.0001)
    print "End of minimization: Converged", self.helper.counter, "cycles"
    print self.helper.get_eigen_summary()
    print "Converged functional: ", self.helper.functional(self.helper.x)




def simdata_pipeline():
    data = gen_data.gen_data(load_hkl=False)
    guesses = gen_data.guess_data(data, perturbate=True, set_model4=False, set_model5=False, perturbate_factor=1)
    truth = gen_data.guess_data(data, perturbate=False)
    print("Loaded")

    # t1_lb = time.time()
    # lbfgs_solver = solvers.LogIsolverCurve(
    #    use_curvatures=True,
    #    data=data,
    #    guess=guesses, truth=truth)
    # t2_lb = time.time()
    # embed()

    t1_eig = time.time()
    ES = eigen_solver(data=data, guess=guesses, truth=truth, lbfgs=False, conj_grad=True, plot=True)
    t2_eig = time.time()

    embed()
    for i in range(10):
        print ES.IAprm_truth[i], np.log(ES.guess['IAprm'][i]), ES.helper.x[i]
        print ES.IBprm_truth[i], np.log(ES.guess['IBprm'][i]), ES.helper.x[ES.Nhkl+i]
        print ES.Gprm_truth[i], ES.guess["Gprm"][i], ES.helper.x[2*ES.Nhkl+i]
        print


def realdata_pipeline():
    GEN = gen_data.gen_real_data_and_guess(gain=28)
    data = GEN['data']
    guess = GEN['guess']

    t1_eig = time.time()
    ES = eigen_solver(data=data, guess=guess, lbfgs=False, conj_grad=True, plot=True)
    t2_eig = time.time()

    np.savez("_eig_dat_res",
                G=ES.x[:ES.Nhkl],
                A=ES.x[ES.Nhkl:ES.Nhkl*2],
                B=ES.x[ES.Nhkl*2:])
    embed()

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "sim":
        simdata_pipeline()
    elif sys.argv[1] == "real":
        realdata_pipeline()

