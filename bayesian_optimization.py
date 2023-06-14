import datetime
import os
from typing import Dict, List, Tuple, Union
import GPy

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.loop import UserFunctionWrapper
from emukit.core.loop.stopping_conditions import FixedIterationsStoppingCondition
from emukit.model_wrappers import GPyModelWrapper

from tuning import *

class Evaluator:
    """Class to evaluate the simulator for a given set of parameters.

    :param replications: number of replications
    """

    def __init__(self, replications: int = 10):
        self.replications = replications

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, 1)
        """
        fitness = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            alloimmunisations = tuning(weights=x, replications=self.replications)
            fitness[i] = alloimmunisations.sum()
        return fitness


def bayes_opt_tuning(init_points_count=20, num_iterations=50, X_init=None, Y_init=None, replications=10):
    evaluator = Evaluator(replications)
    var_obj_names = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood', 'alloimmunisations']
    var_names = var_obj_names[:-1]
    space = ParameterSpace([ContinuousParameter(w,0,1) for w in var_names])
    
    lhs_design = LatinDesign(space)
    # init_points_count = 25
    if X_init is None:
        X_init = lhs_design.get_samples(init_points_count)
        Y_init = evaluator.evaluate(X_init)
    func = UserFunctionWrapper(evaluator.evaluate)
    
    gpy_model = GPy.models.GPRegression(
        X_init, Y_init, GPy.kern.Matern52(space.dimensionality, variance=1.0, ARD=True),
        mean_function=GPy.mappings.Constant(space.dimensionality, 1, 10))
    gpy_model.optimize()
    model = GPyModelWrapper(gpy_model)
    
    bo_loop = BayesianOptimizationLoop(model=model, space=space)
    # num_iterations = 50
    stopping_condition = FixedIterationsStoppingCondition(num_iterations)
    
    bo_loop.run_loop(func, stopping_condition)
    
    results = bo_loop.get_results()
    
    results_df = pd.DataFrame(np.atleast_2d(
        np.hstack((results.minimum_location.flatten(), [results.minimum_value]))), columns=var_obj_names)

    points = np.hstack((bo_loop.loop_state.X, bo_loop.loop_state.Y))
    points_df = pd.DataFrame(points, columns=var_obj_names)

    now = datetime.datetime.now()
    folder = os.path.realpath('out/experiments/exp3/tuning/' + now.strftime('%Y%m%d'))
    os.makedirs(folder, exist_ok=True)
    results_df.to_csv(os.path.join(folder, now.strftime('%H-%M') + '_tuning_results.tsv'), index=False, sep='\t')
    points_df.to_csv(os.path.join(folder, now.strftime('%H-%M') + '_tuning_points.tsv'), index=False, sep='\t')


def unpack_previous_evaluations(
        files: Union[List, str, Tuple],
        var_obj_names: List = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood',
                               'alloimmunisations']) -> Dict:
    """Unpack the previous evaluations from the tuning points file.
    
    :param files: path to the tuning points file or list of paths
    :param list var_obj_names: list of variable and objective names
    :return dict: dict with keys 'X_init' and 'Y_init'
    """
    if isinstance(files, str):
        results_df = pd.read_csv(files, sep='\t')
        X_init = results_df[var_obj_names[:-1]].to_numpy()
        Y_init = results_df[var_obj_names[-1:]].to_numpy()
    elif isinstance(files, (list, tuple)):
        all_X_init = []
        all_Y_init = []
        for file in files:
            results_df = pd.read_csv(file, sep='\t')
            all_X_init.append(results_df[var_obj_names[:-1]].to_numpy())
            all_Y_init.append(results_df[var_obj_names[-1:]].to_numpy())
        X_init = np.vstack(all_X_init)
        Y_init = np.vstack(all_Y_init)
    result = {'X_init': X_init, 'Y_init': Y_init}
    return result


