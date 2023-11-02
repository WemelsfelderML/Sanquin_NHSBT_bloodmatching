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
from Folarin.bayesopt import MultiObjectiveBayesianOptimizationLoop, TargetExtractorFunction


class Evaluator:

    def __init__(self, SETTINGS, PARAMS, scenario, objs):
        self.SETTINGS = SETTINGS
        self.PARAMS = PARAMS
        self.scenario = scenario
        self.objs = objs

    def evaluate_singleobj(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, 1)
        """
        fitness = np.zeros((X.shape[0], 1))
        for i, weights in enumerate(X):
            fitness[i] = tuning(self.SETTINGS, self.PARAMS, self.scenario, self.objs, weights)[0]
        return fitness

    def evaluate_multiobj(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, n_objs)
        """
        fitness = np.zeros((X.shape[0], len(self.objs)))
        for i, weights in enumerate(X):
            fitness[i,:] = tuning(self.SETTINGS, self.PARAMS, self.scenario, self.objs, weights)
        return fitness


def find_init_points(SETTINGS, PARAMS, scenario, objs):

    num_init_points = SETTINGS.num_init_points
    X_init = PARAMS.LHD[:num_init_points, 1:]
    Y_init = np.zeros([num_init_points,len(objs)])

    for o in range(len(objs)):

        if objs[o] == "total_antibodies":
            Y_init[:,o] = [get_total_antibodies(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "total_shortages":
            Y_init[:,o] = [get_shortages(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "total_outdates":
            Y_init[:,o] = [get_outdates(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "alloimm_patients":
            Y_init[:,o] = [get_patients_with_antibodies(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "max_antibodies_pp":
            Y_init[:,o] = [get_max_antibodies_per_patients(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "total_alloimm_risk":
            Y_init[:,o] = [get_total_alloimmunization_risk(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

        elif objs[o] == "issuing_age_SCD":
            Y_init[:,o] = [get_issued_products_nonoptimal_age_SCD(SETTINGS=SETTINGS, PARAMS=PARAMS, method="LP", scenario=scenario, num_init_points=num_init_points, p=p) for p in range(num_init_points)]

    return X_init, Y_init


def bayesian_optimization_singleobj(SETTINGS, PARAMS):

    if sum(SETTINGS.n_hospitals.values()) == 1:
        scenario = "single"
    else:
        scenario = "multi"

    obj = max(SETTINGS.n_obj, key = lambda i: SETTINGS.n_obj[i])
    
    if SETTINGS.num_init_points > 0:
        X_init, Y_init = find_init_points(SETTINGS, PARAMS, scenario, [obj])
    else:
        X_init = [[],[]]
        Y_init = [[],[]]
        

    evaluator = Evaluator(SETTINGS, PARAMS, scenario, [obj])
    var_names = ["mismatches", "youngblood", "FIFO", "usability", "substitution", "today"]
    var_obj_names = var_names + [obj]
    space = ParameterSpace([ContinuousParameter(w,PARAMS.BO_param_ranges[w][0],PARAMS.BO_param_ranges[w][1]) for w in var_names])
    
    func = UserFunctionWrapper(evaluator.evaluate_singleobj)
    
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
    # results_df.to_csv(SETTINGS.generate_filename(output_type="params", scenario=scenario, objectives=[obj], name=f"tuning_results_{now.strftime('%m%d%H%M')}")+".csv", index=False, sep=",")
    points_df.to_csv(SETTINGS.generate_filename(method="BO", output_type="params", scenario=scenario, name=f"tuning_points_{now.strftime('%m%d%H%M')}")+".csv", index=False, sep=",")



def bayesian_optimization_multiobj(SETTINGS, PARAMS):

    if sum(SETTINGS.n_hospitals.values()) == 1:
        scenario = "single"
    else:
        scenario = "multi"

    objs = [obj for obj in SETTINGS.n_obj.keys() if SETTINGS.n_obj[obj] > 0]
    
    X_init, Y_init = find_init_points(SETTINGS, PARAMS, scenario, objs)

    evaluator = Evaluator(SETTINGS, PARAMS, scenario, objs)
    var_names = ["mismatches", "youngblood", "FIFO", "usability", "substitution", "today"]
    var_obj_names = var_names + objs
    space = ParameterSpace([ContinuousParameter(w,PARAMS.BO_param_ranges[w][0],PARAMS.BO_param_ranges[w][1]) for w in var_names])
    
    func = UserFunctionWrapper(evaluator.evaluate_multiobj)
    Y_init_gp = Y_init[:, :1]

    gpy_model = GPy.models.GPRegression(
        X_init, Y_init_gp, GPy.kern.Matern52(space.dimensionality, variance=1.0, ARD=False))
    model = GPyModelWrapper(gpy_model)
    
    bo_loop = MultiObjectiveBayesianOptimizationLoop(model=model, space=space, X_init=X_init, Y_init=Y_init, targets_extractor=TargetExtractorFunction(num_objectives=len([obj for obj in SETTINGS.n_obj.keys() if SETTINGS.n_obj[obj] > 0])))
    stopping_condition = FixedIterationsStoppingCondition(SETTINGS.num_iterations)

    bo_loop.run_loop(func, stopping_condition)
    results = bo_loop.get_results()
    pareto_front = pd.DataFrame(np.hstack([results.pareto_front_X, results.pareto_front_Y]), columns=var_obj_names)

    now = datetime.datetime.now()
    pareto_front.to_csv(SETTINGS.generate_filename(method="BO", output_type="params", scenario=scenario, name=f"tuning_results_{now.strftime('%m%d%H%M')}")+".csv", index=False, sep=",")


def tuning(SETTINGS, PARAMS, scenario, objs, weights):

    # Add objective value parameters to be tested to the parameters class.
    PARAMS.BO_params = np.concatenate([[10], weights])     # 10 = weight for shortages
    # print(PARAMS.BO_params)

    # Find already existing results and continue from the lowest episode number without results.
    e = SETTINGS.num_init_points * SETTINGS.replications
    while os.path.exists(SETTINGS.generate_filename(method="BO", output_type="results", scenario=scenario, name='-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0]), e=e)+".csv"):
        e += 1
    episodes = (e, e + SETTINGS.replications)

    # Execute simulations for the selected episodes.
    SETTINGS.episodes = episodes
    simulation(SETTINGS, PARAMS)

    obj_vals = []
    for obj in objs:
        if obj == "total_antibodies":
            obj_vals.append(get_total_antibodies(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "total_shortages":
            obj_vals.append(get_shortages(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "total_outdates":
            obj_vals.append(get_outdates(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "alloimm_patients":
            obj_vals.append(get_patients_with_antibodies(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "max_antibodies_pp":
            obj_vals.append(get_max_antibodies_per_patients(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "total_alloimm_risk":
            obj_vals.append(get_total_alloimmunization_risk(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

        elif obj == "issuing_age_SCD":
            obj_vals.append(get_issued_products_nonoptimal_age_SCD(SETTINGS=SETTINGS, PARAMS=PARAMS, method="BO", scenario=scenario, episode_start=e))

    return obj_vals


def get_total_antibodies(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):
    
    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[16:31]
                        rq_index = f"{e}_{rq[49]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return len(list(chain.from_iterable(antibodies_per_patient.values())))


def get_max_antibodies_per_patients(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[16:31]
                        rq_index = f"{e}_{rq[49]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return max([len(antibodies) for antibodies in antibodies_per_patient.values()])


def get_patients_with_antibodies(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[16:31]
                        rq_index = f"{e}_{rq[49]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return len([a for a in antibodies_per_patient.values() if len(a) > 0])


def get_shortages(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    shortages = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                
                scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                shortages += data["num shortages"].sum()

    return shortages


def get_outdates(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    outdates = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                
                scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                outdates += data["num outdates"].sum()

    return outdates


def get_total_alloimmunization_risk(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens
    total_alloimm_risk = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            P = [pg for p, pg in PARAMS.patgroups.items() if PARAMS.weekly_demand[htype][p] > 0]
            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i

                scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                for k in antigens.keys():
                    total_alloimm_risk += PARAMS.alloimmunization_risks[2,k] * data[[f"num mismatched patients {pg} {antigens[k]}" for pg in P]].sum().sum()
                        
    return total_alloimm_risk


def get_issued_products_nonoptimal_age_SCD(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    total_products = 0
    nonoptimal_age = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i

                scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="issuing_age", scenario=scenario, name=scenario_name, e=e))

                total_products += data[1,:].sum()
                nonoptimal_age += data[1,:7].sum() + data[1,11:].sum()

    return nonoptimal_age / total_products
        

def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)