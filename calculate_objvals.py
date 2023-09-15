from collections import defaultdict
from itertools import chain
import pickle


def get_total_antibodies(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):
    
    antibodies_per_patient = defaultdict(set)
    for day in range(init_days, init_days + test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[18:35]
            rq_index = f"{e}_{rq[52]}"
            antibodies_per_patient[rq_index].update(k for k in range(n_antigens) if rq_antibodies[k] > 0)

    return len(list(chain.from_iterable(antibodies_per_patient.values())))


def get_patients_with_antibodies(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    antibodies_per_patient = defaultdict(set)
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[18:35]
            rq_index = f"{e}_{rq[52]}"
            antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return len(antibodies_per_patient.keys())


def get_max_antibodies_per_patients(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    antibodies_per_patient = defaultdict(set)
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)
        for rq in data:
            rq_antibodies = rq[18:35]
            rq_index = f"{e}_{rq[52]}"
            antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return max([len(antibodies) for antibodies in antibodies_per_patient.values()])


def get_shortages(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    shortages = 0
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
        data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

        shortages += data["num shortages"].sum()

    return shortages


def get_outdates(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    outdates = 0
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
        data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

        outdates += data["num outdates"].sum()

    return outdates


def get_total_alloimmunization_risk(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    antigens = PARAMS.antigens
    total_alloimm_risk = 0
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
        data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

        for k in antigens.keys():
            total_alloimm_risk += PARAMS.alloimmunization_risks[2,k] * data[[f"num mismatched patients {pg} {antigens[k]}" for pg in P]].sum().sum()
                        
    return total_alloimm_risk


def get_issued_products_nonoptimal_age_SCD(HOME_DIR, init_days, test_days, folder="folder", n_antigens=15, e=0):

    total_products = 0
    nonoptimal_age = 0
    for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

        scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
        data = unpickle(HOME_DIR + f"{folder}/patients_patgroups_London_{e}/{day}").astype(int)

        total_products += data[1,:].sum()
        nonoptimal_age += data[1,:7].sum() + data[1,11:].sum()

    return nonoptimal_age / total_products


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)