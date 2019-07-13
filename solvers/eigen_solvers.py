#!/usr/bin/env libtbx.python

from argparse import ArgumentParser
parser = ArgumentParser("eigen solvers")
parser.add_argument("-i", type=str, help='input_file')
parser.add_argument("-o", type=str, help='output mtz')
parser.add_argument("-sim", action='store_true')
parser.add_argument("-plot", action='store_true')
parser.add_argument("-resoshuff", action='store_true')
parser.add_argument("-solverize", action='store_true')
parser.add_argument("-weights", action='store_true')
parser.add_argument("-N", type=int, default=None, help="Nshots max")
args = parser.parse_args()

import time
import numpy as np
from IPython import embed
from cxid9114.solvers import solvers, gen_data
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
          self.fig3,self.f3_ax = plt.subplots(1,1)

  def build_up(self, objective_only=False):
    if not objective_only:
      self.counter+=1

    self.reset()
    if not objective_only:
      functional = self.functional(self.x)
      print("\n\t<><><><><><>")
      print("Functional value: %.10e" % functional)
      if self.truth is not None:
        functional_ideal = self.functional(self.truth)
        print("\tIdeal Functional value: %.10e" % functional_ideal)
      print("\t<><><><><><>\n")

      if self.Nhkl is not None:
        FA = self.x[:self.Nhkl]
        FB = self.x[self.Nhkl:2*self.Nhkl]
        G = self.x[2*self.Nhkl:]
        print("Max FA: %.10e   Min FA: %.4e " % (max(FA), min(FA)))
        print("Max FB: %.10e   Min FB: %.4e " % (max(FB), min(FB)))
        print("Max ScaleFactor: %.10e   Min ScaleFactor: %.2e " % (max(G), min(G)))
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

                self.f3_ax.clear()
                self.f3_ax.plot(self.truth[2*len(FA):], G, '.', ms=0.2)
                self.fig3.canvas.draw()

            plt.pause(0.3)


      self.stored_functional.append(functional)
    self.access_cpp_build_up_directly_eigen_eqn(objective_only, current_values = self.x)


class eigen_solver(solvers.LBFGSsolver):

  def __init__(self, conj_grad=True, weights=None, plot_truth=False,
               plot=False, sovlerization_maximus=True, *args, **kwargs):
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
    Gx = self.x_init[2 * self.Nhkl:]

    self.x_init = IAx.concatenate(IBx)
    self.x_init = self.x_init.concatenate(Gx)

    self.counter = 0

    # set dummie weights for now
    if weights is None:
        self.Wobs = flex.double(np.ones(len(self.Yobs)))
    else:
        self.Wobs = weights

    if plot_truth:
        try:
            truth = self.x_truth
        except AttributeError as error:
            print(error)
            truth = None
    else:
        truth=None

    self.helper = eigen_helper(initial_estimates=self.x_init, Nhkl=self.Nhkl, plot=plot, truth=truth)
    self.helper.eigen_wrapper.conj_grad = conj_grad
    self.helper.set_cpp_data(
      self.Yobs, self.Wobs, self.Aidx, self.Gidx, self.PA, self.PB, self.LA, self.LB, self.Nhkl, self.Ns)

    self.helper.restart()

    if sovlerization_maximus:
        try:
          _ = normal_eqns_solving.levenberg_marquardt_iterations_encapsulated_eqns(
                   non_linear_ls=self.helper,
                   n_max_iterations=200,
                   track_all=True,
                   step_threshold=0.0001)
        except (KeyboardInterrupt, AssertionError):
          pass
        print "End of minimization: Converged", self.helper.counter, "cycles"
        print self.helper.get_eigen_summary()
        print "Converged functional: ", self.helper.functional(self.helper.x)



def simdata_pipeline():
    data = gen_data.gen_data(args.i, load_hkl=False, Nshot_max=args.N)

    if args.resoshuff:
        hmap = np.load(args.i)["hkl_map"][()]
        #reso_bins = [24, 4.5] + list(np.linspace(2, 4.3, 10)[::-1])i
        reso_bins = [24.9840,4.3025,
                     4.3025,3.4178,
                     3.4178, 2.9866,
                     2.9866, 2.7139,
                     2.7139, 2.5195,
                     2.5195, 2.3711,
                     2.3711, 2.2524,
                     2.2524, 2.1545,
                     2.1545, 2.0716,
                     2.0716, 2.0001]

    guesses = gen_data.guess_data(data, use_Iguess=False,
                                  perturbate=True, perturbate_factor=2.5,
                                  hmap=hmap, reso_bins=reso_bins)
    truth = gen_data.guess_data(data, perturbate=False)
    print("Loaded")


    # t1_lb = time.time()
    # lbfgs_solver = solvers.LogIsolverCurve(
    #    use_curvatures=True,
    #    data=data,
    #    guess=guesses, truth=truth)
    # t2_lb = time.time()
    # embed()

    if args.weights:
        weights = data['weights']
    else:
        weights=None
    t1_eig = time.time()
    ES = eigen_solver(data=data, guess=guesses, truth=truth,
                      lbfgs=False, conj_grad=True,
                        plot=False, weights=weights,
                      sovlerization_maximus=args.solverize)
    t2_eig = time.time()

    IB = np.exp(ES.helper.x[ES.Nhkl:2*ES.Nhkl].as_numpy_array())
    IA = np.exp(ES.helper.x[:ES.Nhkl].as_numpy_array())

    IAtru = np.exp(ES.IAprm_truth.as_numpy_array())
    IBtru = np.exp(ES.IBprm_truth.as_numpy_array())
    if args.plot:
        import pylab as plt
        plt.plot(IA, IAtru, '.')
        plt.plot(IB, IBtru, '.')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim(1e-1, 1e7)
        plt.ylim(1e-1, 1e7)
        plt.plot(IAtru, IBtru, '.' )
        plt.show()

    from cctbx import sgtbx, crystal
    from cctbx.array_family import flex
    from cctbx import miller
    from cxid9114 import utils

    Nh = ES.Nhkl
    IA = ES.helper.x[:Nh].as_numpy_array()
    IB = ES.helper.x[Nh:2*Nh].as_numpy_array()

    #IA = ES.x_init[:Nh].as_numpy_array()
    #IB = ES.x_init[Nh:2*Nh].as_numpy_array()

    dataA = np.load(args.i)
    hkl_map = dataA["hkl_map"][()]
    hkl_map2 = {v:k for k,v in hkl_map.iteritems()}
    Nhkl = len(hkl_map)
    assert(Nh == Nhkl)
    hout, IAout, IBout = [], [], []
    for i in range(Nhkl):
        h = hkl_map2[i]
        valA = IA[i]
        valB = IB[i]
        hout.append(h)
        IAout.append(np.exp(valA))
        IBout.append(np.exp(valB))

    sg = sgtbx.space_group(" P 4nw 2abw")
    Symm = crystal.symmetry(unit_cell=(79, 79, 38, 90, 90, 90), space_group=sg)
    hout = tuple(hout)
    mil_idx = flex.miller_index(hout)
    mil_set = miller.set(crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)
    IAout_flex = flex.double(np.ascontiguousarray(IAout))
    IBout_flex = flex.double(np.ascontiguousarray(IBout))
    mil_arA = miller.array(mil_set, data=IAout_flex).set_observation_type_xray_intensity()
    mil_arB = miller.array(mil_set, data=IBout_flex).set_observation_type_xray_intensity()

    from cxid9114.parameters import ENERGY_CONV
    waveA = ENERGY_CONV/8944.
    waveB = ENERGY_CONV/9034.
    ucell = mil_arA.unit_cell()
    sgi = mil_arA.space_group_info()
    from iotbx import mtz
    mtz_handle = mtz.object()
    mtz_handle.set_title(title="MAD_MTZ")
    mtz_handle.set_space_group_info(space_group_info=sgi)
    mtz_handle.set_hkl_base(unit_cell=ucell)
    mtz_cr = mtz_handle.crystals()[0] #mtz_handle.add_crystal(name="Crystal",
        #project_name="project", unit_cell=ucell)

    dsetA = mtz_cr.add_dataset(name="datasetA", wavelength=waveA)
    _=dsetA.add_miller_array(miller_array=mil_arA, column_root_label="IAobs")

    dsetB = mtz_cr.add_dataset(name="datasetB", wavelength=waveB)
    _=dsetB.add_miller_array(miller_array=mil_arB, column_root_label="IBobs")

    mtz_handle.show_summary()
    mtz_handle.write(args.o)

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
    if args.sim:
        simdata_pipeline()
    else:
        realdata_pipeline()

