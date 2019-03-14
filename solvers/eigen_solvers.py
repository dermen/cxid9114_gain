import gen_data
import numpy as np
from cxid9114.solvers import solvers
from IPython import embed
import cxid9114
from itertools import izip
from scipy.sparse import coo_matrix
from libtbx.test_utils import approx_equal
from IPython import embed
data = gen_data.gen_data()
guesses = gen_data.guess_data(data, perturbate=True, set_model4=False, set_model5=False)
truth = gen_data.guess_data(data, perturbate=False)
print("Loaded")
import time
## working LBFGS example
#lbfgs_solver = solvers.LogIsolverCurve(
#    use_curvatures=False,
#    data=data,
#    guess=guesses, truth=truth)
#
#embed()
#
##lbfgs_solver.minimize()
#IA,IB,G = np.exp(lbfgs_solver.x[:lbfgs_solver.Nhkl]), np.exp(lbfgs_solver.x[lbfgs_solver.Nhkl: 2*lbfgs_solver.Nhkl]), lbfgs_solver.x[2*lbfgs_solver.Nhkl:]
#
## iterate for 5 minutes then save the model.. that would be model 5
#
# # Note curvatures work with model 4 or with perturbate_factor=0.1, i.e. close to minimum
t1_lb = time.time()
lbfgs_solver = solvers.LogIsolverCurve(
    use_curvatures=True,
    data=data,
    guess=guesses, truth=truth)
t2_lb = time.time()

embed()

from scitbx.array_family import flex
from scitbx.lstbx import normal_eqns
from scitbx.examples.bevington.silver import levenberg_common

class eigen_helper(cxid9114.log_sparse_jac_base,levenberg_common,normal_eqns.non_linear_ls_mixin):
  def __init__(self, initial_estimates):
    super(eigen_helper, self).__init__(n_parameters=len(initial_estimates))
    self.initialize(initial_estimates)
    self.stored_functional = []


  def build_up(self, objective_only=False):
    if not objective_only:
      self.counter+=1

    self.reset()
    #print list(self.x),objective_only
    if not objective_only:
      functional = self.functional(self.x)
      print functional
      self.stored_functional.append( functional)
      #self.print_step("LM sparse",functional = functional)
    self.access_cpp_build_up_directly_eigen_eqn(objective_only, current_values = self.x)

  #def functional(self):
  #  resid = self.fvec_callable(self.x)
  #  return flex.sum(resid*resid)

from scitbx.lstbx import normal_eqns_solving
class eigen_solver(solvers.LBFGSsolver):

  def __init__(self, *args, **kwargs):
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

    #self.Yobs  = flex.double(self.Yobs.as_numpy_array() / self.Gprm_truth.as_numpy_array()[self.Gidx.as_numpy_array()] )

    self.counter = 0

    # set dummie weights for now
    self.Wobs = flex.double(np.ones(len(self.Yobs)))

    #self.x = self.x_init.deep_copy()  # NOTE: put this in because it was set in bevington example
    self.helper = eigen_helper(initial_estimates = self.x_init)
    self.helper.set_cpp_data(
      self.Yobs, self.Wobs, self.Aidx, self.Gidx, self.PA, self.PB, self.LA, self.LB, self.Nhkl, self.Ns)

    self.helper.restart()
    #print self.helper.get_eigen_summary()
    iterations = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
               non_linear_ls = self.helper,
               n_max_iterations = 5000,
               track_all=True,
               step_threshold = 0.0001)
    print "End of minimization: Converged", self.helper.counter,"cycles"
    print self.helper.get_eigen_summary()

    print "Converged functional: ", self.helper.functional(self.helper.x)


# NOTE: setting the gain to be fixed

t1_eig = time.time()
ES = eigen_solver(data=data,guess=guesses,truth=truth, lbfgs=False)
t2_eig = time.time()

embed()

for i in range(10):
    print ES.IAprm_truth[i], np.log(ES.guess['IAprm'][i]), ES.helper.x[i]
    print ES.IBprm_truth[i], np.log(ES.guess['IBprm'][i]), ES.helper.x[ES.Nhkl+i]
    #print ES.Gprm_truth[i], ES.guess["Gprm"][i], ES.helper.x[2*ES.Nhkl+i]
    print

