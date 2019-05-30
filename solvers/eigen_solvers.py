import gen_data
import numpy as np
from cxid9114.solvers import solvers
import cxid9114
from scitbx.lstbx import normal_eqns_solving
from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common
import time

from IPython import embed


data = gen_data.gen_data(load_hkl=False)
guesses = gen_data.guess_data(data, perturbate=True, set_model4=False, set_model5=False, perturbate_factor=1)
truth = gen_data.guess_data(data, perturbate=False)
print("Loaded")


#t1_lb = time.time()
#lbfgs_solver = solvers.LogIsolverCurve(
#    use_curvatures=True,
#    data=data,
#    guess=guesses, truth=truth)
#t2_lb = time.time()
#embed()


class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):
  def __init__(self, initial_estimates):
    super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
    self.initialize(initial_estimates)
    self.stored_functional = []

  def build_up(self, objective_only=False):
    if not objective_only:
      self.counter+=1

    self.reset()
    if not objective_only:
      functional = self.functional(self.x)
      print functional
      self.stored_functional.append( functional)
    self.access_cpp_build_up_directly_eigen_eqn(objective_only, current_values = self.x)


class eigen_solver(solvers.LBFGSsolver):

  def __init__(self, conj_grad=True, *args, **kwargs):
    solvers.LBFGSsolver.__init__(self, *args, **kwargs)  # NOTE: do it with lbfgs=False
    # ^ brings in Yobs, LA, LB, PA, PB, Nhkl, Ns, Nmeas,   Aidx, Gidx

    # correct because working with logs
    self.IAprm_truth = flex.log(self.IAprm_truth)
    self.IBprm_truth = flex.log(self.IBprm_truth)

    self.x_init = flex.double(np.ascontiguousarray(self.guess["IAprm"])).concatenate(
      flex.double(np.ascontiguousarray(self.guess["IBprm"]))).concatenate(flex.double(np.ascontiguousarray(self.guess["Gprm"])))
    assert (len(self.x_init) == self.Nhkl * 2 + self.Ns)

    IAx = flex.log(self.x_init[:self.Nhkl])
    IBx = flex.log(self.x_init[self.Nhkl:2 * self.Nhkl])
    Gx = self.x_init[2 * self.Nhkl:]

    self.x_init = IAx.concatenate(IBx)
    self.x_init = self.x_init.concatenate(Gx)

    self.counter = 0

    # set dummie weights for now
    self.Wobs = flex.double(np.ones(len(self.Yobs)))

    self.helper = eigen_helper(initial_estimates=self.x_init)
    self.helper.eigen_wrapper.conj_grad = conj_grad
    self.helper.set_cpp_data(
      self.Yobs, self.Wobs, self.Aidx, self.Gidx, self.PA, self.PB, self.LA, self.LB, self.Nhkl, self.Ns)

    self.helper.restart()
    _ = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
               non_linear_ls=self.helper,
               n_max_iterations=5000,
               track_all=True,
               step_threshold=0.0001)
    print "End of minimization: Converged", self.helper.counter, "cycles"
    print self.helper.get_eigen_summary()
    print "Converged functional: ", self.helper.functional(self.helper.x)

t1_eig = time.time()
ES = eigen_solver(data=data, guess=guesses, truth=truth, lbfgs=False, conj_grad=True)
t2_eig = time.time()

embed()
for i in range(10):
    print ES.IAprm_truth[i], np.log(ES.guess['IAprm'][i]), ES.helper.x[i]
    print ES.IBprm_truth[i], np.log(ES.guess['IBprm'][i]), ES.helper.x[ES.Nhkl+i]
    print ES.Gprm_truth[i], ES.guess["Gprm"][i], ES.helper.x[2*ES.Nhkl+i]
    print

