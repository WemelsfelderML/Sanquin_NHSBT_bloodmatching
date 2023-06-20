import numpy as np
import pandas as pd

from settings import *
from params import *
from simulation import *

def tuning(weights, num_init_points, replications):

    
    PARAMS.BO_params = np.array(weights)

    SETTINGS
    
    for r in range(replications):
        simulation(SETTINGS, PARAMS)
    
    # Read SCD logging files from executed episode -> calculate and return alloimmunizations.
    unpickle(SETTINGS.home_dir + f"results/{SETTINGS.model_name}/{e}/patients_{SETTINGS.strategy}_{hospital.htype}/{day}")


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)