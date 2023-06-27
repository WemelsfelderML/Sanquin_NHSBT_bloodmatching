import numpy as np
import pandas as pd
import pickle
import multiprocessing
import itertools

from blood import *
from hospital import *
from dc import *
from minrar_single import *
from minrar_multi import *
from save_state import *
from alloimmunize import *

# Run the simulation.
def simulation(SETTINGS, PARAMS):

    # Multi-hospital setup
    if sum(SETTINGS.n_hospitals.values()) > 1:

        for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
            simulate_episode_multi(SETTINGS, PARAMS, e)

        # processes = []
        # for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
        #     p = multiprocessing.Process(target=simulate_episode_multi, args=(SETTINGS, PARAMS, e))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

    # Single-hospital setup
    else:

        # Get the hospital's type
        htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])

        for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
            simulate_episode_single(SETTINGS, PARAMS, htype, e)
        # processes = []
        # for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
        #     p = multiprocessing.Process(target=simulate_episode_single, args=(SETTINGS, PARAMS, htype, e))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()


def simulate_episode_single(SETTINGS, PARAMS, htype, e):

    print(f"\nEpisode: {e}")

    obj_params = PARAMS.BO_params
    if len(obj_params) == 0:
        obj_params = PARAMS.LHD[e]
    with open(SETTINGS.generate_filename(output_type="params", scenario="single", name="params", e=e)+".pickle", 'wb') as f:
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
    issuing_age = np.zeros([len(PARAMS.patgroups), PARAMS.max_age])

    days = range(SETTINGS.init_days + SETTINGS.test_days)

    wip_path = SETTINGS.generate_filename(output_type="wip", scenario="single", name=htype, e=e)
    logs, issuing_age, day, dc, hospitals = load_state(SETTINGS, PARAMS, wip_path, e, logs, issuing_age, dc, [hospital])
    hospital = hospitals[0]
    days = [d for d in days if d >= day]

    # Run the simulation for the given number of days, and write outputs for all 'test days' to the dataframe.
    x = {}
    for day in days:
        print(f"\nDay {day}")
        logs, issuing_age, x = simulate_day_single(SETTINGS, PARAMS, obj_params, logs, issuing_age, dc, hospital, e, day, x)

        # if day % 5 == 0:
        save_state(SETTINGS, wip_path, logs, issuing_age, e, day, dc, [hospital])

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

    df.to_csv(SETTINGS.generate_filename(output_type="results", scenario="single", name=hospital.htype, e=e)+".csv", sep=',', index=True)


def simulate_episode_multi(SETTINGS, PARAMS, e):

    print(f"\nEpisode: {e}")

    obj_params = PARAMS.BO_params
    if len(obj_params) == 0:
        obj_params = PARAMS.LHD[e]
    with open(SETTINGS.generate_filename(output_type="params", scenario="multi", name="params", e=e)+".pickle", 'wb') as f:
        pickle.dump(obj_params, f, pickle.HIGHEST_PROTOCOL)

    # Initialize all hospitals and the distribution center.
    hospitals = []
    for htype in SETTINGS.n_hospitals.keys():
        hospitals += [Hospital(SETTINGS, PARAMS, htype, (e*SETTINGS.n_hospitals[htype])+i) for i in range(SETTINGS.n_hospitals[htype])]
    dc = Distribution_center(SETTINGS, PARAMS, hospitals, e)

    # Initialize all hospital inventories with random supply, where the product's age is uniformly distributed between 0 and the maximum shelf life.
    for hospital in hospitals:
        # Fill the initial inventory with product only of age 0.
        hospital.inventory += dc.sample_supply_single_day(PARAMS, hospital.inventory_size, 0)

        # # Fill the initial inventory with products of uniformly distributed age.
        # n_products = round(hospital.inventory_size / PARAMS.max_age)
        # for age in range(PARAMS.max_age):
        #     hospital.inventory += dc.sample_supply_single_day(PARAMS, n_products, age)

    # Create a dataframe to be filled with output measures for every simulated day.
    logs = SETTINGS.initialize_output_dataframe(PARAMS, hospitals, e)
    # df_matches = pd.DataFrame(columns = ["day", "unit", "patient"])
    
    days = range(SETTINGS.init_days + SETTINGS.test_days)

    wip_path = SETTINGS.generate_filename(output_type="wip", scenario="multi", name='-'.join([h.name[:3] for h in hospitals]), e=e)
    logs, day, dc, hospitals = load_state(SETTINGS, PARAMS, wip_path, e, logs, dc, hospitals)
    days = [d for d in days if d >= day]
    
    # Run the simulation for the given number of days, and write outputs for all 'test days' to the dataframe.
    for day in days:
        print(f"\nDay {day}")
        logs = simulate_day_multi(SETTINGS, PARAMS, obj_params, logs, dc, hospitals, e, day)

        save_state(SETTINGS, wip_path, logs, e, day, dc, hospitals)

    # Write the created output dataframe to a csv file in the 'results' directory.
    df = pd.DataFrame(logs, columns = sorted(SETTINGS.column_indices, key=SETTINGS.column_indices.get))
    ci = SETTINGS.column_indices

    df["model name"] = SETTINGS.model_name
    df["test days"] += SETTINGS.test_days
    df["init days"] += SETTINGS.init_days
    df["supply scenario"] = '-'.join([str(SETTINGS.n_hospitals[ds])+ds[:3] for ds in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[ds] > 0]) + f"_{e}"
    
    for hospital in hospitals:
        indices = [SETTINGS.row_indices[(day,hospital.name)] for day in days]
        df.loc[indices,"location"] = hospital.name
        df.loc[indices,"avg daily demand"] = hospital.avg_daily_demand
        df.loc[indices,"inventory size"] = hospital.inventory_size

    df.to_csv(SETTINGS.generate_filename(output_type="results", scenario="multi", name='-'.join([h.name[:3] for h in hospitals]), e=e)+".csv", sep=',', index=True)


# Single-hospital setup: perform matching within one hospital.
def simulate_day_single(SETTINGS, PARAMS, obj_params, logs, issuing_age, dc, hospital, e, day, x):

    # Update the set of available requests, by removing requests for previous days (regardless of 
    # whether they were satisfied or not) and sampling new requests that become known today.
    hospital.requests = [r for r in hospital.requests if r.day_issuing >= day]
    num_requests = hospital.sample_requests_single_day(SETTINGS, PARAMS, e, day=day)

    # Sets of all inventory products and patient requests.
    I = hospital.inventory
    R = hospital.requests

    x_next = {}
    if num_requests > 0:
        # Solve the MINRAR model, matching the hospital's inventory products to the available requests.
        gurobi_logs, x = minrar_single_hospital(SETTINGS, PARAMS, obj_params, hospital, I, R, day, e, x)
        alloimmunize(SETTINGS, PARAMS, "single", hospital, day, x)

        for i, ip in enumerate(I):
            for r, rq in enumerate(R):
                ir = (ip.index, rq.index)
                x_next[ir] = x[i,r]
    else:
        gurobi_logs = [0, 2, 0, 0]
        x = np.zeros([len(hospital.inventory),1])
        with open(SETTINGS.generate_filename(output_type="results", subtype="patients", scenario="single", name=hospital.name, day=day)+".pickle", 'wb') as f:
            pickle.dump(np.array([]), f, pickle.HIGHEST_PROTOCOL)

    logs, issuing_age = log_results(SETTINGS, PARAMS, logs, issuing_age, gurobi_logs, dc, hospital, e, day, x=x)

    # Update the hospital's inventory, by removing issued or outdated products, increasing product age, and sampling new supply.
    supply_size = hospital.update_inventory(SETTINGS, PARAMS, x, day)
    hospital.inventory += dc.sample_supply_single_day(PARAMS, supply_size)

    return logs, issuing_age, x_next


# Multi-hospital setup: perform matching simultaniously for multiple hospitals, and strategically distribute new supply over all hospitals.
def simulate_day_multi(SETTINGS, PARAMS, obj_params, logs, issuing_age, dc, hospitals, e, day):

    # For each hospital, update the set of available requests, by removing requests for previous days 
    # (regardless of whether they were satisfied or not) and sampling new requests that become known today.
    num_requests = 0
    for hospital in hospitals:
        hospital.requests = [r for r in hospital.requests if r.day_issuing >= day]
        num_requests += hospital.sample_requests_single_day(SETTINGS, PARAMS, e, day=day)

        for rq in hospital.requests:
            rq.allocated_from_dc = 0

    if num_requests > 0:
        # Solve the MINRAR model, matching the inventories of all hospitals and the distribution center to all available requests.
        gurobi_logs, xh, xdc = minrar_multiple_hospitals(SETTINGS, PARAMS, obj_params, dc, hospitals, day, e)
        for h in range(len(hospitals)):
            alloimmunize(SETTINGS, PARAMS, "multi", hospitals[h], day, xh[h])

    # Get all distribution center products that were allocated to requests with tomorrow as their issuing date.
    allocations_from_dc = np.zeros([len(dc.inventory),len(hospitals)])
    for h in range(len(hospitals)):
        for r in range(len(hospitals[h].requests)):
            if hospitals[h].requests[r].day_issuing == (day + 1):
                
                issued = np.where(xdc[h][:,r]==1)[0]
                for i in issued:
                    allocations_from_dc[i,h] = 1
                    hospitals[h].requests[r].allocated_from_dc += 1     # total number of products allocated to this request from DC

        # Write the results to a csv file.
        logs, issuing_age = log_results(SETTINGS, PARAMS, logs, issuing_age, gurobi_logs, dc, hospitals[h], e, day, x=xh[h])

    # Update the hospital's inventory, by removing issued or outdated products and sampling new supply.
    supply_sizes = np.array([hospitals[h].update_inventory(SETTINGS, PARAMS, xh[h], day) for h in range(len(hospitals))])

    # Allocate products to each of the hospitals to restock them upto their maximum capacity.
    model, x = allocate_remaining_supply_from_dc(SETTINGS, PARAMS, day, dc.inventory, hospitals, supply_sizes, allocations_from_dc)
    
    # Abstract the remaining supply from the solved model, and ship all allocated products to the hospitals.
    for h in range(len(hospitals)):
        hospitals[h].inventory += [dc.inventory[i] for i in range(len(dc.inventory)) if x[i,h] >= 1]

    # Update the distribution centers's inventory, by removing shipped or outdated products, and increasing product age.
    dc.update_inventory(SETTINGS, PARAMS, x, day)

    return logs, issuing_age
