from dials.algorithms.indexing.nave_parameters \
    import nave_parameters
from dials.algorithms.refinement import RefinerFactory
from dxtbx.model.experiment_list import ExperimentListDumper
from dials.algorithms.refinement.weighting_strategies \
    import StillsWeightingStrategy
from dials.algorithms.refinement.reflection_manager \
    import ReflectionManager
from dials.algorithms.refinement.prediction.managed_predictors \
    import ExperimentsPredictorFactory
from dials.algorithms.refinement.parameterisation import \
    build_prediction_parameterisation
from dials.algorithms.refinement.parameterisation.parameter_report \
    import ParameterReporter
from dials.algorithms.refinement.constraints import ConstraintManagerFactory
from dials.algorithms.refinement.target_stills \
    import LeastSquaresStillsResidualWithRmsdCutoff
from dials.algorithms.refinement.engine import Refinery

def get_refman( reflections, experiments, delpsi_constant=1000000):
    """
    delpsi_constant = 1000000
        .help = "used by the stills strategy to choose absolute weight value"
                "for the angular distance from Ewald sphere term of the target"
                "function, whilst the X and Y parts use statistical weights"
        .type = float(value_min=0, allow_none=True)
    """
    outlier_detector = None  # turn this off and see what happens
    min_refl = 0  #  params.minimum_sample_size,
    max_refl = None #  params.maximum_sample_size,
    verbosity = 10

    weight_strat = StillsWeightingStrategy(delpsi_constant)

    refman = ReflectionManager(
            reflections = reflections,
            experiments = experiments,
            nref_per_degree = None,
            max_sample_size=max_refl,
            min_sample_size=min_refl,
            close_to_spindle_cutoff=0.0,
            trim_scan_edges=0.0,
            outlier_detector=outlier_detector,
            weighting_strategy_override=weight_strat,
            verbosity = verbosity)

    return refman

def get_predictor(experiments):
    predictor = ExperimentsPredictorFactory.from_experiments(
        experiments,
        force_stills=True,
        spherical_relp=True)

    return predictor

def update_observation_table(refman, ref_predictor):
    obs_refls = refman.get_obs()

    _ = ref_predictor(obs_refls) # I believe this sets resolution of each reflection

    x,y,z = obs_refls['xyzobs.mm.value'].parts()  # .parts() must return the transpose zip(*array)
    x_calc, y_calc, z_calc = obs_refls['xyzcal.mm'].parts()
    obs_refls['x_resid'] = x_calc - x
    obs_refls['y_resid'] = y_calc - y
    obs_refls['phi_resid'] = z_calc - z


def finalize_refname(refman):
    #ca = refman.get_centroid_analyser()
    #analysis = ca(calc_average_residuals=False, calc_periodograms=False)
    refman.finalise(analysis=None)


def get_pred_params_and_reporter(params, experiments, refman):
    weird_pred_param_class_options = params.refinement.parameterisation
    weird_pred_param_class_options.sparse = False
    weird_pred_param_class_options.scan_varying = False
    pred_param = build_prediction_parameterisation(
                    options = weird_pred_param_class_options,
                    experiments = experiments,
                    reflection_manager = refman,
                    do_stills = True)

    # Parameter reporting
    param_reporter = ParameterReporter(
        pred_param.get_detector_parameterisations(),
        pred_param.get_beam_parameterisations(),
        pred_param.get_crystal_orientation_parameterisations(),
        pred_param.get_crystal_unit_cell_parameterisations(),
        pred_param.get_goniometer_parameterisations())

    return pred_param, param_reporter


def get_restr_param(params, pred_param):
    options = params.refinement.parameterisation
    options.crystal.unit_cell.restraints.tie_to_target.values = (79.1, 79.1, 38.4, 90., 90., 90.)
    options.crystal.unit_cell.restraints.tie_to_target.sigmas = (3, 3, 3, 0.2, 0.2, 0.2)
    restr_param = RefinerFactory.config_restraints(
        options, pred_param)
    print("asd")
    return restr_param, options


def get_constrain_man(params, pred_param, verbosity=10):
    cmf = ConstraintManagerFactory(params, pred_param, verbosity)
    return cmf()


def get_target(experiments, predictor, refl_man, pred_param,
               restr_param=None):

    target =  LeastSquaresStillsResidualWithRmsdCutoff(
                experiments=experiments,
                predictor=predictor,
                reflection_manager=refl_man,
                prediction_parameterisation=pred_param,
                restraints_parameterisation=restr_param,
                frac_binsize_cutoff=0.5,
                absolute_cutoffs=None,
                gradient_calculation_blocksize=None)

    return target


def get_refinery(target, pred_param, constr_man=None):
    refinery = Refinery(target=target,
                        prediction_parameterisation=pred_param,
                        constraints_manager=constr_man)
    return refinery


def make_refiner(params, experiments, reflections, re_estimate_mosaic=False, prefix=None):

    refiner = RefinerFactory.\
        from_parameters_data_experiments(params, reflections, experiments)
    refiner.run()

    experiments = refiner.get_experiments()
    predicted = refiner.predict_for_indexed()
    reflections['xyzcal.mm'] = predicted['xyzcal.mm']
    reflections['entering'] = predicted['entering']
    reflections = reflections.select(refiner.selection_used_for_refinement())

    # Re-estimate mosaic estimates
    if re_estimate_mosaic:
        nv = nave_parameters(params=params,
                             experiments=experiments,
                             reflections=reflections,
                             refinery=refiner,
                             graph_verbose=False)
        nv()
        acceptance_flags_nv = nv.nv_acceptance_flags
        reflections = reflections.select(acceptance_flags_nv)

    if prefix is not None:
        dump = ExperimentListDumper(experiments)
        dump.as_json( prefix + "_exper.json")
        reflections.as_pickle(reflections, prefix + "_refl.pickle")

    return experiments, reflections