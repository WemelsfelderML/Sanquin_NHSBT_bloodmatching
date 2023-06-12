import numpy as np
import pandas as pd
import random
import math
import os
import multiprocessing

from blood import *


def generate_demand_for_week(args):

    SETTINGS, PARAMS, htype, first_weekday = args

    weekly_demand = PARAMS.weekly_demand[htype]
    data = []

    for patgroup in PARAMS.patgroups.keys():
        requests = sample_requests_for_week(SETTINGS, PARAMS, patgroup, weekly_demand[patgroup])

        daily_demand = np.random.rand(7)
        daily_demand *= weekly_demand[patgroup] / np.sum(daily_demand)
        ordered_days = np.argsort(daily_demand)[::-1]
        
        for day in ordered_days:

            num_units = 0
            while (num_units < daily_demand[day]) and (len(requests) > 0):

                rq = requests.pop()
                lead_time = random.choices(range(8), weights = PARAMS.request_lead_time_probabilities[patgroup], k=1)[0]
                data.append([first_weekday + day, max(0, first_weekday + day - lead_time), rq[1], patgroup, rq[0]] + rq[2:])
                num_units += rq[1]

    return data


def get_random_request(SETTINGS, PARAMS, patgroup):

    # Determine lead time, number of units and ethnicity of the patient request.
    num_units = random.choices(range(13), weights = PARAMS.request_num_units_probabilities[patgroup], k=1)[0]

    # The antigen phenotypes for patients with sickle cell disease are modelled according to prevales in the African population,
    # while patients of all other patient groups are modelled in accordance with the Caucasian population.
    if patgroup == 1:
        ethnicity = 1
    else:
        ethnicity = 0

    AB0 = list(random.choices(PARAMS.ABO_phenotypes, weights = PARAMS.ABO_prevalences[ethnicity], k=1)[0])
    Rhesus = list(random.choices(PARAMS.Rhesus_phenotypes, weights = PARAMS.Rhesus_prevalences[ethnicity], k=1)[0])
    Kell = list(random.choices(PARAMS.Kell_phenotypes, weights = PARAMS.Kell_prevalences[ethnicity], k=1)[0])
    Duffy = list(random.choices(PARAMS.Duffy_phenotypes, weights = PARAMS.Duffy_prevalences[ethnicity], k=1)[0])
    Kidd = list(random.choices(PARAMS.Kidd_phenotypes, weights = PARAMS.Kidd_prevalences[ethnicity], k=1)[0])
    MNS = list(random.choices(PARAMS.MNS_phenotypes, weights = PARAMS.MNS_prevalences[ethnicity], k=1)[0])

    # Create a Blood instance using the generated information.
    return [ethnicity, num_units] + AB0 + Rhesus + Kell + Duffy + Kidd + MNS


# Generate a list of random requests according to the given distribution.
def sample_requests_for_week(SETTINGS, PARAMS, patgroup, num_units_requested):

    requests = []
    num_units = 0
    while num_units <= (num_units_requested - (max(PARAMS.request_num_units_probabilities[patgroup])/2)):
        rq = get_random_request(SETTINGS, PARAMS, patgroup)
        requests.append(rq)
        num_units += rq[1]

    return requests


# Generate a given number of demand files, where each file contains all demand for one simulation episode.
def generate_demand(SETTINGS, PARAMS, htype):

    duration = SETTINGS.test_days + SETTINGS.init_days

    # Make sure all required folders exist.
    path = SETTINGS.home_dir + "demand"
    if os.path.exists(path) == False:
        os.mkdir(path)

    path = SETTINGS.home_dir + f"demand/{duration}"
    if os.path.exists(path) == False:
        os.mkdir(path)

    # Find already existing demand files of the chosen size and duration, and make sure not to overwrite them.
    i = 0
    while os.path.exists(SETTINGS.home_dir + f"demand/{duration}/{htype}_{i}.csv"):
        i += 1

    # For every episode in the given range, generate requests for all days of the simulation.
    for _ in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):

        print(f"Generating demand '{htype}_{i}'.")

        num_processes = multiprocessing.cpu_count()  # Get the number of CPU cores
        with multiprocessing.Pool(num_processes) as pool:
            all_weekly_demands = np.concatenate(pool.map(generate_demand_for_week, [(SETTINGS, PARAMS, htype, first_weekday) for first_weekday in range(0, duration, 7)]), axis=0)

        # Convert list of numpy arrays to pandas DataFrame
        df = pd.DataFrame(all_weekly_demands, columns = ["day issuing", "day available", "num units", "patgroup", "ethnicity"] + list(PARAMS.antigens.values()))

        df.to_csv(SETTINGS.home_dir + f"demand/{duration}/{htype}_{i}.csv", index=False)

        i += 1



