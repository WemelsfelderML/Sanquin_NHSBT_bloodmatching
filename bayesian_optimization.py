import numpy as np
import datetime
import GPy
import pickle
import os

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop import UserFunctionWrapper
from emukit.core.loop.stopping_conditions import FixedIterationsStoppingCondition
from emukit.model_wrappers import GPyModelWrapper
from collections import defaultdict
from itertools import chain

from simulation import *


class Evaluator:

    def __init__(self, SETTINGS, PARAMS, scenario):
        self.SETTINGS = SETTINGS
        self.PARAMS = PARAMS
        self.scenario = scenario

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, 1)
        """
        fitness = np.zeros((X.shape[0], 1))
        for i, weights in enumerate(X):
            alloimmunisations = tuning(self.SETTINGS, self.PARAMS, self.scenario, weights)
            fitness[i] = sum(alloimmunisations)
        return fitness


def find_init_points(SETTINGS, PARAMS, scenario):

    num_init_points = SETTINGS.num_init_points

    X_init = PARAMS.LHD[:num_init_points, 1:]

    Y_init = []
    for p in range(num_init_points):
        Y_init.append(len(get_antibodies(SETTINGS, PARAMS, scenario, num_init_points=num_init_points, p=p)))
    
    return X_init, np.array(Y_init).reshape(-1, 1)


def bayes_opt_tuning(SETTINGS, PARAMS):

    if sum(SETTINGS.n_hospitals.values()) == 1:
        scenario = "single"
    else:
        scenario = "multi"
    
    X_init, Y_init = find_init_points(SETTINGS, PARAMS, scenario)

    evaluator = Evaluator(SETTINGS, PARAMS, scenario)
    var_obj_names = ["mismatches", "youngblood", "FIFO", "usability", "substitution", "today", "antibodies"]
    var_names = var_obj_names[:-1]
    space = ParameterSpace([ContinuousParameter(w,0,1) for w in var_names])
    
    func = UserFunctionWrapper(evaluator.evaluate)
    
    gpy_model = GPy.models.GPRegression(
        X_init, Y_init, GPy.kern.Matern52(space.dimensionality, variance=1.0, ARD=True),
        mean_function=GPy.mappings.Constant(space.dimensionality, 1, 10))                       # TODO
    gpy_model.optimize()
    model = GPyModelWrapper(gpy_model)
    
    bo_loop = BayesianOptimizationLoop(model=model, space=space)
    stopping_condition = FixedIterationsStoppingCondition(SETTINGS.num_iterations)
    
    bo_loop.run_loop(func, stopping_condition)
    
    results = bo_loop.get_results()
    
    results_df = pd.DataFrame(np.atleast_2d(
        np.hstack((results.minimum_location.flatten(), [results.minimum_value]))), columns=var_obj_names)

    points = np.hstack((bo_loop.loop_state.X, bo_loop.loop_state.Y))
    points_df = pd.DataFrame(points, columns=var_obj_names)

    now = datetime.datetime.now()
    results_df.to_csv(SETTINGS.generate_filename(output_type="params", scenario=scenario, name=f"tuning_results_{now.strftime('%m%d%H%M')}")+".csv", index=False, sep=",")
    points_df.to_csv(SETTINGS.generate_filename(output_type="params", scenario=scenario, name=f"tuning_points_{now.strftime('%m%d%H%M')}")+".csv", index=False, sep=",")


def tuning(SETTINGS, PARAMS, scenario, weights):

    # Add objective value parameters to be tested to the parameters class.
    PARAMS.BO_params = np.concatenate([[10], weights])     # 10 = weight for shortages

    # Find already existing results and continue from the lowest episode number without results.
    e = 0
    while os.path.exists(SETTINGS.generate_filename(output_type="results", scenario=scenario, name='-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0]), e=e)+".csv"):
        e += 1
    episodes = (e, e + SETTINGS.replications)

    # Execute simulations for the selected episodes.
    SETTINGS.episodes = episodes
    simulation(SETTINGS, PARAMS)

    return get_antibodies(SETTINGS, PARAMS, scenario, e)


def get_antibodies(SETTINGS, PARAMS, scenario, episode_start=0, num_init_points=1, p=0):
    
    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):

        for htype in SETTINGS.n_hospitals.keys():
            n = SETTINGS.n_hospitals[htype]
            for i in range(n):

                e = (((r * num_init_points) + p) * n) + i

                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):
                    data = unpickle(SETTINGS.generate_filename(output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[18:35]
                        rq_index = f"{e}_{rq[52]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return list(chain.from_iterable(antibodies_per_patient.values()))
        

def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)