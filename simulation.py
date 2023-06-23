import numpy as np
import pandas as pd
import pickle
import multiprocessing

from blood import *
from hospital import *
from dc import *
from minrar_single import *
from minrar_multi import *
# from minrar_offline import *
from read_solution import *
from save_state import *
from alloimmunize import *

# Run the simulation.
def simulation(SETTINGS, PARAMS):

    # Get the hospital's type
    htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])

    processes = []
    for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
        p = multiprocessing.Process(target=simulate_episode, args=(SETTINGS, PARAMS, htype, e))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def simulate_episode(SETTINGS, PARAMS, htype, e):

    print(f"\nEpisode: {e}")

    obj_params = PARAMS.BO_params
    if len(obj_params) == 0:
        obj_params = PARAMS.LHD[e]
    with open(SETTINGS.home_dir + f"param_opt/params_{e}.pickle", 'wb') as f:
        pickle.dump(obj_params, f, pickle.HIGHEST_PROTOCOL)
        
    # Initialize the hospital. A distribution center is also initialized to provide the hospital with random supply.
    hospital = Hospital(SETTINGS, PARAMS, htype, e)
    dc = Distribution_center(SETTINGS, PARAMS, [hospital], e)

    # Fill the initial inventory with product only of age 0.
    hospital.inventory += dc.sample_supply_single_day(PARAMS, hospital.inventory_size, 0)

    # # Fill the initial inventory with products of uniformly distributed age.
    # n_products = round(hospital.inventory_size / PARAMS.max_age)
    # for age in range(PARAMS.max_age):
    #     hospital.inventory += dc.sample_supply_single_day(PARAMS, n_products, age)

    # Create a dataframe to be filled with output measures for every simulated day.
    logs = SETTINGS.initialize_output_dataframe(PARAMS, [hospital], e)

    days = range(SETTINGS.init_days + SETTINGS.test_days)

    logs, day, dc, hospitals = load_state(SETTINGS, PARAMS, e, logs, dc, [hospital])
    hospital = hospitals[0]
    days = [d for d in days if d >= day]

    # Run the simulation for the given number of days, and write outputs for all 'test days' to the dataframe.
    for day in days:
        print(f"\nDay {day}")
        logs = simulate_day(SETTINGS, PARAMS, obj_params, logs, dc, hospital, e, day)

        # if day % 5 == 0:
        save_state(SETTINGS, logs, e, day, dc, [hospital])

    # Write the created output dataframe to a csv file in the 'results' directory.
    df = pd.DataFrame(logs, columns = sorted(SETTINGS.column_indices, key=SETTINGS.column_indices.get))
    ci = SETTINGS.column_indices

    df["model name"] = SETTINGS.model_name
    df["test days"] += SETTINGS.test_days
    df["init days"] += SETTINGS.init_days
    df["supply scenario"] = '-'.join([str(SETTINGS.n_hospitals[ds])+ds[:3] for ds in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[ds] > 0]) + f"_{e}"
    
    df["location"] = hospital.name
    df["avg daily demand"] = hospital.avg_daily_demand
    df["inventory size"] = hospital.inventory_size

    df.to_csv(SETTINGS.generate_filename("results") + f"{SETTINGS.strategy}_{htype}_{e}.csv", sep=',', index=True)


# Single-hospital setup: perform matching within one hospital.
def simulate_day(SETTINGS, PARAMS, obj_params, logs, dc, hospital, e, day):

    # Update the set of available requests, by removing requests for previous days (regardless of 
    # whether they were satisfied or not) and sampling new requests that become known today.
    hospital.requests = [r for r in hospital.requests if r.day_issuing >= day]
    num_requests = hospital.sample_requests_single_day(SETTINGS, PARAMS, e, day=day)

    # Ii = {ip.index: idx for idx, ip in enumerate(I)}
    # Ri = {rq.index: idx for idx, rq in enumerate(R)}
    # heuristic = [(Ii[m[0]], Ri[m[1]]) for m in set(matches_indices) if m[0] in set(Ii.keys()) and m[1] in set(Ri.keys())] if len(matches_indices) > 0 else []

    if num_requests > 0:
        # Solve the MINRAR model, matching the hospital's inventory products to the available requests.
        gurobi_logs, x = minrar_single_hospital(SETTINGS, PARAMS, obj_params, hospital, day, e)
        alloimmunize(SETTINGS, PARAMS, hospital, e, day, x)
    else:
        gurobi_logs = [0, 2, 0, 0]
        x = np.zeros([len(hospital.inventory),1])
    
    # matches = np.where(x>0)
    # matches_indices = [(I[m[0]].index, R[m[1]].index) for m in zip(matches[0], matches[1])] if len(matches) > 0 else []

    logs = log_results(SETTINGS, PARAMS, logs, gurobi_logs, hospital, e, day, x=x)

    # Update the hospital's inventory, by removing issued or outdated products, increasing product age, and sampling new supply.
    supply_size = hospital.update_inventory(SETTINGS, PARAMS, x, day)
    hospital.inventory += dc.sample_supply_single_day(PARAMS, supply_size)

    return logs
