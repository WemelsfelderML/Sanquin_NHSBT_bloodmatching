from settings import *
from params import *
from demand import *
from supply import *
from simulation import *

def main():
    
    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)

    # If a directory to store log files or results does not yet exist, make one.
    paths = ["results", "results/"+SETTINGS.model_name]
    paths += ["wip", f"wip/{SETTINGS.model_name}"] + [f"wip/{SETTINGS.model_name}/{e}" for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1])]
    paths += [f"results/{SETTINGS.model_name}/{e}/patients_{SETTINGS.strategy}_{htype}" for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]) for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype] > 0]
    # paths += ["NN_training_data"] + [f"NN_training_data/{htype}_{''.join(PARAMS.antigens.values())}" for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype] > 0]
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
            print(f"Multi-hospital scenario with {SETTINGS.n_hospitals['regional']} regional and {SETTINGS.n_hospitals['university']} university hospitals.")
        else:
            print(f"Single-hospital scenario, {max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])} hospital.")
        print(f"Using {SETTINGS.strategy} strategy for matching, with patgroup_musts = {SETTINGS.patgroup_musts}.")

        if SETTINGS.method == "LP":
            simulation(SETTINGS, PARAMS)
        elif SETTINGS.method == "RL":
            reinforcement_learning(SETTINGS, PARAMS)
        else:
            print("Parameter 'mode' is set to 'optimize', but no existing method for optimization is given. Try 'RL' or 'LP'.")
    else:
        print("No mode for running the code is given. Please change the 'mode' parameter in 'settings.py' to one of the following values:")
        print("'demand': generate demand scenarios")
        print("'supply': generate supply scenarios")
        print("'optimize': for optimizing RBC matching, either using LP or RL method")


if __name__ == "__main__":
    main()