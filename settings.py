import numpy as np
from itertools import product
import os

class Settings():

    # def __init__(self, model_name, LHD_configs, emin, emax, total_cores_max):
    def __init__(self):

        # self.home_dir = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Sanquin_NHSBT_bloodmatching/"
        self.home_dir = "/home/mw922/Sanquin_NHSBT_bloodmatching/"

        # "demand": generate demand data
        # "supply": generate supply data
        # "optimize": run simulations and optimize matching
        self.mode = "optimize"

        # Output files will be stored in directory results/[model_name].
        # self.model_name = model_name
        self.model_name = "newnew"

        
        ##########
        # SUPPLY #
        ##########

        # Initialize hospital inventory with certain major antigen profile, [] for normal initialization.
        self.majors_init = []

        #########################
        # OPTIMIZATION SETTINGS #
        #########################

        # "LP": Use linear programming.
        # "BO": Use bayesian optimization to tune objval parameters.
        self.method = "LP"
        # self.LHD_configs = LHD_configs
        self.LHD_configs = 500

        # "on": online optimization.
        # "off": offline optimization.
        self.line = "on"

        #########################
        # SIMULATION PARAMETERS #
        #########################

        # Only the results of test days will be logged.
        # self.test_days = 2 * (7 * 6)
        self.test_days = 26 * (7 * 6)     # Follow SCD patients over 26 transfusion episodes (~3 years)
        # self.test_days = 87 * (7 * 6)   # Follow SCD patients over 87 transfusion episodes (~10 years)
        self.init_days = 2 * 35

        # (x,y): Episode numbers range(x,y) will be optimized.
        # The total number of simulations executed will thus be y - x.
        # self.episodes = (emin, emax)
        self.episodes = (0, 500)
        # self.total_cores_max = total_cores_max    # Set the maximum number of cores to be used in total when executing episodes in parallel.
        self.total_cores_max = 16    # Set the maximum number of cores to be used in total when executing episodes in parallel.

        # Number of hospitals considered. If more than 1 (regional and university combined), a distribution center is included.
        # "UCLH" : University College London Hospitals
        # "NMUH" : North Middlesex University Hospital
        # "WH" : Whittington Health
        # "London" : Merged supply and demand of the three hospitals above.
        self.n_hospitals = {
            "UCLH"  : 0,
            "NMUH"  : 0,
            "WH"    : 0,
            "London": 1,
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

        #########################
        # BAYESIAN OPTIMIZATION #
        #########################

        # self.num_init_points = 3
        self.num_init_points = 500
        self.num_iterations = 100 
        self.replications = 1

        self.dir1 = f"{self.LHD_configs}x{self.replications}LHD"
        
        # Put 1 if the objective should be optimized in BO, 0 if not.
        self.n_obj = {
            "total_antibodies"   : 1,
            "total_shortages"    : 0,
            "total_outdates"     : 0,
            "alloimm_patients"   : 0,
            "max_antibodies_pp"  : 0,
            "total_alloimm_risk" : 0,
            "issuing_age_SCD"    : 0,
        }

        ####################
        # GUROBI OPTIMIZER #
        ####################

        self.show_gurobi_output = False   # True or False
        self.gurobi_threads = 1           # Number of threads available, or None in case of no limit
        self.gurobi_timeout = 15*60       # Number of seconds allowed for optimization, None in case of no limit



    # Generate a file name for exporting log or result files.
    def generate_filename(self, method="method", output_type="output_type", subtype="subtype", scenario="single", size="size", name="name", e="e", day="day"):

        path = self.home_dir

        # Generated demand data.
        if output_type == "demand":
            path += f"demand/{size}/{name}_{e}"
            return path

        # Generated supply data.
        if output_type == "supply":
            path += f"supply/{size}/{name}_{e}"
            return path

        dir0 = f"{scenario}_{self.test_days}"
        
        # dir2 = self.model_name+"_" if self.model_name != "" and method == "BO" else ""
        dir2 = self.model_name+"_" if self.model_name != "" else ""
        dir2 += "_".join(["".join([s[0] for s in obj_name.split("_")]) for obj_name in self.n_obj.keys() if self.n_obj[obj_name] > 0]) if method == "BO" else ""
        dir2 += "/" if dir2 != "" else ""

        # Simulation results.
        if output_type == "results":

            path += f"results/{dir0}/{self.dir1}/{dir2}"

            if subtype == "patients":
                # here 'name' also includes the episode number
                path += f"patients_{self.strategy}_{name}/{day}" 
            elif subtype == "issuing_age":
                path += f"age_{self.strategy}_{name}_{e}"
            else:
                path += f"{self.strategy}_{name}_{e}"

        if output_type == "wip":
            path += f"wip/{dir0}/{self.dir1}/{dir2}{self.strategy}_{name}_{e}/"

        if output_type == "params":
            path += f"optimize_params/{dir0}/{self.dir1}/{dir2}{name}_{e}"

        return path
         


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
        header = ["logged", "day", "location", "model name", "avg daily demand", "inventory size", "test days", "init days"]

        # Gurobi optimizer info.
        header += ["gurobi status", "nvars", "calc time", "ncons"]
        
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
        # header += ["avg issuing age"]
        # header += [f"{i} to {j}" for i in ABOD_names for j in ABOD_names]
        # header += [f"{eth0} to {eth1}" for eth0 in ethnicities for eth1 in ethnicities]
        # header += [f"num allocated at dc {p}" for p in patgroups]

        # Matching performance.
        header += ["num outdates"] + [f"num outdates {i}" for i in ABOD_names]
        header += ["num shortages"] + [f"num shortages {i}" for i in ABOD_names] + [f"num shortages {p}" for p in patgroups]
        # header += [f"num {p} {i+1} units short" for p in patgroups for i in range(12)] + ["num unavoidable shortages"]
        header += [f"num mismatched patients {p} {k}" for p in patgroups for k in antigens] + [f"num mismatched units {p} {k}" for p in patgroups for k in antigens]
        # header += [f"num mismatches {eth} {k}" for eth in ethnicities for k in antigens]

        # Set the dataframe's index to each combination of day and location name.
        locations = [hospital.name for hospital in hospitals]
        if len(hospitals) > 1:
            locations.append(f"dc_{episode}")
            header += ["num outdates dc"] + [f"num outdates dc {i}" for i in ABOD_names]

        self.column_indices = {header[i] : i for i in range(len(header))}
        self.row_indices = {(day, location): i for i, (day, location) in enumerate(product(days, locations))}
        logs = np.zeros([len(self.row_indices),len(self.column_indices)])
        
        return logs


    # Check whether a given path exists, and create the path if it doesn't.
    def check_dir_existence(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)