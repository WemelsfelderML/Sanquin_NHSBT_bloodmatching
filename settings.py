import numpy as np
from itertools import product
import os

class Settings():

    def __init__(self):

        self.home_dir = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Sanquin_NHSBT_bloodmatching/"
        # self.home_dir = "/home/mw922/Sanquin_NHSBT_bloodmatching/"

        # "demand": generate demand data
        # "supply": generate supply data
        # "optimize": run simulations and optimize matching
        self.mode = "optimize"

        # Output files will be stored in directory results/[model_name].
        self.model_name = "LHD testing"

        
        ##########
        # SUPPLY #
        ##########

        # Initialize hospital inventory with certain major antigen profile, [] for normal initialization.
        self.majors_init = []

        #########################
        # OPTIMIZATION SETTINGS #
        #########################

        # "LP": Use linear programming.
        # "RL": Use reinforcement learning.
        self.method = "LP"

        # "on": online optimization.
        # "off": offline optimization.
        self.line = "on"

        #########################
        # SIMULATION PARAMETERS #
        #########################

        # Only the results of test days will be logged.
        self.test_days = 26 * (7 * 6)   # Follow SCD patients over 26 transfusion episodes (~3 years)
        self.init_days = 2 * 35

        # (x,y): Episode numbers range(x,y) will be optimized.
        # The total number of simulations executed will thus be y - x.
        self.episodes = (0,20)

        # Number of hospitals considered. If more than 1 (regional and university combined), a distribution center is included.
        # "UCLH" : University College London Hospitals
        # "NMUH" : North Middlesex University Hospital
        # "WH" : Whittington Health
        self.n_hospitals = {
            "UCLH" : 1,
            "NMUH" : 0,
            "WH" : 0,
        }

        # Size factor for distribution center and hospitals.
        # Average daily demand × size factor = inventory size.
        self.inv_size_factor_dc = 6
        self.inv_size_factor_hosp = 3

        # "major": Only match on the major antigens.
        # "relimm": Use relative immunogenicity weights for mismatching.
        # "patgroups": Use patient group specific mismatching weights.
        self.strategy = "patgroups"
        self.patgroup_musts = True
        

        ####################
        # GUROBI OPTIMIZER #
        ####################

        self.show_gurobi_output = False     # True or False
        self.gurobi_threads = None          # Number of threads available, or None in case of no limit
        self.gurobi_timeout = 15*60       # Number of minutes allowed for optimization, None in case of no limit


    # Generate a file name for exporting log or result files.
    def generate_filename(self, output_type):

        return self.home_dir + f"{output_type}/{self.model_name}/{self.method.lower()}_"


    # Create the dataframe with all required columns, to store outputs during the simulations.
    def initialize_output_dataframe(self, PARAMS, hospitals, episode):

        ##########
        # PARAMS #
        ##########

        antigens = PARAMS.antigens.values()
        ABOD_names = PARAMS.ABOD
        patgroups = [PARAMS.patgroups[p] for p in PARAMS.patgroups.keys() if np.array([PARAMS.weekly_demand[hospital.htype] for hospital in hospitals]).sum(axis=0)[p] > 0]
        # ethnicities = ["Caucasian", "African", "Asian"]
        days = list(range(self.init_days + self.test_days))

    
        ##########
        # HEADER #
        ##########

        # General information.
        header = ["logged", "day", "location", "model name", "supply scenario", "avg daily demand", "inventory size", "test days", "init days"]

        # Gurobi optimizer info.
        header += ["gurobi status", "nvars", "calc time", "ncons"]
        # header += ["objval shortages", "objval mismatches", "objval FIFO", "objval usability", "objval substitution"]
        
        # Information about patients, donors, demand and supply.
        header += ["num patients"] + [f"num {p} patients" for p in patgroups]
        # header += [f"num {eth} patients" for eth in ethnicities]
        header += ["num units requested"] + [f"num units requested {p}" for p in patgroups]
        # header += [f"num requests {i+1} units" for i in range(12)]
        header += ["num supplied products"]
        # header += [f"num supplied {i}" for i in ABOD_names] + [f"num requests {i}" for i in ABOD_names]
        header += [f"num {i} in inventory" for i in ABOD_names]
        header += ["num Fya-Fyb- in inventory", "num Fya+Fyb- in inventory", "num Fya-Fyb+ in inventory", "num Fya+Fyb+ in inventory", "num R0 in inventory"]

        # Only if the offline model is used.
        # header += ["products available today", "products in inventory today"]

        # Which products were issued to which patiens.
        header += ["avg issuing age"]
        # header += [f"{i} to {j}" for i in ABOD_names for j in ABOD_names]
        # header += [f"{eth0} to {eth1}" for eth0 in ethnicities for eth1 in ethnicities]
        # header += [f"num allocated at dc {p}" for p in patgroups]

        # Matching performance.
        header += ["num outdates"] + [f"num outdates {i}" for i in ABOD_names]
        header += ["num shortages"]
        header += [f"num shortages {i}" for i in ABOD_names]
        header += [f"num shortages {p}" for p in patgroups]
        # header += [f"num {p} {i+1} units short" for p in patgroups for i in range(12)] + ["num unavoidable shortages"]
        header += [f"num mismatched patients {p} {k}" for p in patgroups for k in antigens] + [f"num mismatched units {p} {k}" for p in patgroups for k in antigens]
        # header += [f"num mismatches {eth} {k}" for eth in ethnicities for k in antigens]

        # Set the dataframe's index to each combination of day and location name.
        locations = [hospital.name for hospital in hospitals]
        if len(hospitals) > 1:
            locations.append(f"dc_{episode}")

        self.column_indices = {header[i] : i for i in range(len(header))}
        self.row_indices = {(day, location): i for i, (day, location) in enumerate(product(days, locations))}
        logs = np.zeros([len(self.row_indices),len(self.column_indices)])
        
        return logs


    # Check whether a given path exists, and create the path if it doesn't.
    def check_dir_existence(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)