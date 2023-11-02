from collections import defaultdict
from itertools import chain
import pickle
import pandas as pd

from settings import *
from params import *


def get_total_antibodies(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):
    
    antibodies_per_patient = defaultdict(set)
    for day in range(init_days, init_days + test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[16:31]
            rq_index = f"{e}_{rq[49]}"
            antibodies_per_patient[rq_index].update(k for k in range(n_antigens) if rq_antibodies[k] > 0)

    return [list(chain.from_iterable(antibodies_per_patient.values())).count(k) for k in range(n_antigens)]


def get_max_antibodies_per_patients(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    antibodies_per_patient = defaultdict(set)
    for day in range(init_days, init_days + test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[16:31]
            rq_index = f"{e}_{rq[49]}"
            antibodies_per_patient[rq_index].update(k for k in range(n_antigens) if rq_antibodies[k] > 0)

    return max([len(antibodies) for antibodies in antibodies_per_patient.values()])


def get_patients_with_antibodies(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    antibodies_per_patient = defaultdict(set)
    for day in range(init_days, init_days + test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[16:31]
            rq_index = f"{e}_{rq[49]}"
            antibodies_per_patient[rq_index].update(k for k in range(n_antigens) if rq_antibodies[k] > 0)

    return len([a for a in antibodies_per_patient.values() if len(a) > 0])


def get_substitutions_SCD_major(HOME_DIR, init_days, test_days, folder="folder", n_antigens=3, e=0):
    
    total_substitutions = defaultdict(set)
    for day in range(init_days, init_days + test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_substitutions = rq[46:49]
            rq_index = f"{e}_{rq[49]}"
            total_substitutions[rq_index].update(k for k in range(n_antigens) if rq_substitutions[k] > 0)

    return [list(chain.from_iterable(total_substitutions.values())).count(k) for k in range(n_antigens)]


def get_shortages(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    # shortages = 0

    data = pd.read_csv(HOME_DIR + f"{folder}/patgroups_1London_{e}.csv")
    shortages = data["num shortages"].sum()

    return shortages


def get_outdates(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    # outdates = 0
    # for day in range(init_days, init_days + test_days):

    data = pd.read_csv(HOME_DIR + f"{folder}/patgroups_1London_{e}.csv")
    outdates = data["num outdates"].sum()

    return outdates


def get_total_alloimmunization_risk(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)

    patgroups = [PARAMS.patgroups[p] for p in PARAMS.patgroups.keys() if PARAMS.weekly_demand["London"][p] > 0]

    total_alloimm_risk = 0
    # for day in range(init_days, init_days + test_days):

    data = pd.read_csv(HOME_DIR + f"{folder}/patgroups_1London_{e}.csv")
    for k in PARAMS.antigens.keys():
        total_alloimm_risk += PARAMS.alloimmunization_risks[2,k] * data[[f"num mismatched patients {pg} {PARAMS.antigens[k]}" for pg in patgroups]].sum().sum()

    return total_alloimm_risk


def get_issued_products_nonoptimal_age_SCD(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    total_products = 0
    nonoptimal_age = 0

    data = unpickle(HOME_DIR + f"{folder}/age_patgroups_1London_{e}").astype(int)
    total_products += data[1,:].sum()
    nonoptimal_age += data[1,:7].sum() + data[1,11:].sum()

    return nonoptimal_age / total_products


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)