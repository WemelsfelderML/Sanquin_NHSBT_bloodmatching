import multiprocessing

from settings import *
from params import *
from demand import *
from supply import *
from simulation import *

from bayesian_optimization import *

def main():
    
    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)

    if SETTINGS.method == "LP":
        episodes_min = SETTINGS.episodes[0]
        episodes_max = SETTINGS.episodes[1]
    elif SETTINGS.method == "BO":
        episodes_min = 0
        episodes_max = (SETTINGS.num_init_points * SETTINGS.replications) + (SETTINGS.num_iterations * SETTINGS.replications)
    else:
        print("Unknown method selected, check self.method in settings.py.")

    if sum(SETTINGS.n_hospitals.values()) == 1:
        scenario = "single"
    else:
        scenario = "multi"

    # If a directory to store log files or results does not yet exist, make one.
    paths =  ["optimize_params", f"optimize_params/{SETTINGS.model_name}_{scenario}"]
    paths += ["wip", f"wip/{SETTINGS.model_name}_{scenario}"] + ["results", f"results/{SETTINGS.model_name}_{scenario}"]
    paths += [f"wip/{SETTINGS.model_name}_{scenario}/{SETTINGS.strategy}_{htype}_{e}" for e in range(episodes_min, episodes_max) for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype] > 0]
    paths += [f"results/{SETTINGS.model_name}_{scenario}/patients_{SETTINGS.strategy}_{htype}_{e}" for e in range(episodes_min, episodes_max) for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype] > 0]
    
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    # Sample demand for each day in the simulation, and write to a csv file.
    if SETTINGS.mode == "demand":
        for htype in SETTINGS.n_hospitals.keys():
            for _ in range(SETTINGS.n_hospitals[htype]):
                generate_demand(SETTINGS, PARAMS, htype)

    # Sample RBC units to be used as supply in the simulation, and write to csv file.
    elif SETTINGS.mode == "supply":
        generate_supply(SETTINGS, PARAMS)

    # Run the simulation, using either linear programming or reinforcement learning to determine the matching strategy.
    elif SETTINGS.mode == "optimize":

        print(f"Starting {SETTINGS.method} optimization ({SETTINGS.line}line).")
        print(f"Results will be written to {SETTINGS.model_name} folder.")
        print(f"Simulating {SETTINGS.init_days + SETTINGS.test_days} days ({SETTINGS.init_days} init, {SETTINGS.test_days} test).")
        if sum(SETTINGS.n_hospitals.values()) > 1:
            print(f"Multi-hospital scenario with {', '.join([f'{SETTINGS.n_hospitals[ds]} {ds}' for ds in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[ds] > 0])} hospitals.")
        else:
            print(f"Single-hospital scenario, {max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])} hospital.")
        print(f"Using {SETTINGS.strategy} strategy for matching, with patgroup_musts = {SETTINGS.patgroup_musts}.")

        if SETTINGS.method == "LP":
            simulation(SETTINGS, PARAMS)
        elif SETTINGS.method == "BO":
            bayes_opt_tuning(SETTINGS, PARAMS)
        else:
            print("Parameter 'mode' is set to 'optimize', but no existing method for optimization is given. Try 'BO' or 'LP'.")
    else:
        print("No mode for running the code is given. Please change the 'mode' parameter in 'settings.py' to one of the following values:")
        print("'demand': generate demand scenarios")
        print("'supply': generate supply scenarios")
        print("'optimize': for optimizing RBC matching, either using LP or RL method")


if __name__ == "__main__":
    main()