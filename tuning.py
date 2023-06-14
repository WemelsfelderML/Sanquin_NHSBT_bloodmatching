import numpy as np
import pandas as pd

from settings import *
from params import *
from simulation import *


def tuning(weights, replications):

    # Check episodes already saved -> change SETTINGS.episodes to next available episode.
    # Find already existing supply files of the chosen size and duration, and make sure not to overwrite them.
    e = 0
    while os.path.exists(SETTINGS.generate_filename("results") + f"{SETTINGS.strategy}_{htype}_{e}.csv"):
        e += 1

    
    simulation(SETTINGS, PARAMS)
    

    # Read SCD logging files from executed episode -> calculate and return alloimmunizations.

    unpickle(SETTINGS.home_dir + f"results/{SETTINGS.model_name}/{e}/patients_{SETTINGS.strategy}_{hospital.htype}/{day}")


def initialize_setup(weights, e):

    SETTINGS = Settings()
    PARAMS = Params(SETTINGS, weights)

    # If a directory to store log files or results does not yet exist, make one.
    paths =  ["wip", f"wip/{SETTINGS.model_name}", f"wip/{SETTINGS.model_name}/{e}"]
    paths += ["results", "results/"+SETTINGS.model_name, f"results/{SETTINGS.model_name}/{e}"]
    paths += [f"results/{SETTINGS.model_name}/{e}/patients_{SETTINGS.strategy}_{htype}" for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype] > 0]
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    print(f"Starting {SETTINGS.method} optimization ({SETTINGS.line}line).")
    print(f"Results will be written to {SETTINGS.model_name} folder.")
    print(f"Simulating {SETTINGS.init_days + SETTINGS.test_days} days ({SETTINGS.init_days} init, {SETTINGS.test_days} test).")
    print(f"Single-hospital scenario, {max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])} hospital.")
    print(f"Using {SETTINGS.strategy} strategy for matching, with patgroup_musts = {SETTINGS.patgroup_musts}.")


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)