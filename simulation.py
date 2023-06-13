import numpy as np
import pandas as pd
import pickle
# import csv
# import os

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

    # Multi-hospital setup: perform matching simultaniously for multiple hospitals, and strategically distribute new supply over all hospitals.
    if sum(SETTINGS.n_hospitals.values()) > 1:

        print("Not yet implemented.")

        # # Run the simulation for the given range of episodes.
        # for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
        #     print(f"\nEpisode: {e}")

        #     # Initialize all hospitals and the distribution center.
        #     hospitals = []
        #     for htype in SETTINGS.n_hospitals.keys():
        #         hospitals += [Hospital(SETTINGS, PARAMS, htype, (e*SETTINGS.n_hospitals[htype])+i) for i in range(SETTINGS.n_hospitals[htype])]
        #     dc = Distribution_center(SETTINGS, PARAMS, hospitals, e)

        #     # Initialize all hospital inventories with random supply, where the product's age is uniformly distributed between 0 and the maximum shelf life.
        #     for hospital in hospitals:
        #         # Fill the initial inventory with product only of age 0.
        #         hospital.inventory += dc.sample_supply_single_day(PARAMS, hospital.inventory_size, 0)

        #         # # Fill the initial inventory with products of uniformly distributed age.
        #         # n_products = round(hospital.inventory_size / PARAMS.max_age)
        #         # for age in range(PARAMS.max_age):
        #         #     hospital.inventory += dc.sample_supply_single_day(PARAMS, n_products, age)

        #     # Create a dataframe to be filled with output measures for every simulated day.
        #     logs = SETTINGS.initialize_output_dataframe(PARAMS, hospitals, e)
        #     # df_matches = pd.DataFrame(columns = ["day", "unit", "patient"])
            
        #     days = range(SETTINGS.init_days + SETTINGS.test_days)

        #     logs, day, dc, hospitals = load_state(SETTINGS, PARAMS, e, logs, dc, hospitals)
        #     days = [d for d in days if d >= day]
            
        #     # Run the simulation for the given number of days, and write outputs for all 'test days' to the dataframe.
        #     for day in days:
        #         print(f"\nDay {day}")
        #         logs = simulate_multiple_hospitals(SETTINGS, PARAMS, logs, dc, hospitals, e, day)

        #         save_state(SETTINGS, logs, e, day, dc, hospitals)

        #     # Write the created output dataframe to a csv file in the 'results' directory.
        #     df = pd.DataFrame(logs, columns = sorted(SETTINGS.column_indices, key=SETTINGS.column_indices.get))
        #     ci = SETTINGS.column_indices

        #     df.iloc[:,ci["model name"]] = SETTINGS.model_name
        #     df.iloc[:,ci["test days"]] += SETTINGS.test_days
        #     df.iloc[:,ci["init days"]] += SETTINGS.init_days
        #     df.iloc[:,ci["supply scenario"]] = f"cau{round(SETTINGS.donor_eth_distr[0]*100)}_afr{round(SETTINGS.donor_eth_distr[1]*100)}_asi{round(SETTINGS.donor_eth_distr[2]*100)}_{e}"
            
        #     for hospital in hospitals:
        #         indices = [SETTINGS.row_indices[(day,hospital.name)] for day in days]
        #         df.iloc[indices,ci["demand scenario"]] = f"{hospital.htype}_{e}"
        #         df.iloc[indices,ci["avg daily demand"]] = hospital.avg_daily_demand
        #         df.iloc[indices,ci["inventory size"]] = hospital.inventory_size

        #     df.to_csv(SETTINGS.generate_filename("results") + f"{SETTINGS.strategy}_{'-'.join([str(SETTINGS.n_hospitals[ds]) + ds[:3] for ds in SETTINGS.n_hospitals.keys()])}_{e}.csv", sep=',', index=True)

         
    # Single-hospital setup: perform matching within one hospital.
    else:

        # Get the hospital's type ('regional' or 'university')
        htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")

            # Initialize the hospital. A distribution center is also initialized to provide the hospital with random supply.
            hospital = Hospital(SETTINGS, PARAMS, htype, e)
            dc = Distribution_center(SETTINGS, PARAMS, [hospital], e)

            # Online model: each day in the simulation is solved iteratively, without knowledge about future days.
            if SETTINGS.line == "on":

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
                matches_indices = []

                # Run the simulation for the given number of days, and write outputs for all 'test days' to the dataframe.
                for day in days:
                    print(f"\nDay {day}")
                    logs, matches_indices = simulate_single_hospital(SETTINGS, PARAMS, logs, dc, hospital, e, day, matches_indices)

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


            # Offline model: all days in the simulation are solved simultaniously, having full knowledge about all demand and supply involved.
            elif SETTINGS.line == "off":

                print("Offline simulation not yet implemented.")

                # # Get the full range of days to simulate.
                # days = range(SETTINGS.init_days + SETTINGS.test_days)

                # # Load all sampled supply products into the hospital's inventory. Note that the inventory is stored as an
                # # ordered list, and within the model it is ensured that products can only become available in the supplied order.
                # hospital.inventory = dc.sample_supply_single_day(PARAMS, SETTINGS.supply_size)
                
                # # Load all requests, containing both the day of becoming known and the day of issuing as properties.
                # for day in days:
                #     hospital.sample_requests_single_day(PARAMS, day=day)

                # # Create a dataframe to be filled with output measures for every simulated day.
                # df = SETTINGS.initialize_output_dataframe(PARAMS, [hospital], e)
                
                # # Run the simulation for the full range of days.
                # model = minrar_offline(SETTINGS, PARAMS, hospital, days)
                
                # # Abstract the optimal variable values from the solved model and write the corresponding results to a csv file.
                # df, x, y, z, a, b = read_minrar_solution(SETTINGS, PARAMS, df, model, [hospital], e)
                # for day in days:
                #     df = log_results(SETTINGS, PARAMS, df, hospital, e, day, x=x, y=y, z=z, a=a, b=b)

                # # Write the created output dataframe to a csv file in the 'results' directory.
                # df.to_csv(SETTINGS.generate_filename("results") + f"offline_{SETTINGS.strategy}_{htype}_{e}.csv", sep=',', index=True)

            else:
                print("Please set the 'line' variable in settings.py to a valid value (either 'off' or 'on').")


# Single-hospital setup: perform matching within one hospital.
def simulate_single_hospital(SETTINGS, PARAMS, logs, dc, hospital, e, day, matches_indices):

    # Update the set of available requests, by removing requests for previous days (regardless of 
    # whether they were satisfied or not) and sampling new requests that become known today.
    hospital.requests = [r for r in hospital.requests if r.day_issuing >= day]
    num_requests = hospital.sample_requests_single_day(SETTINGS, PARAMS, e, day=day)

    # Sets of all inventory products and patient requests.
    I = hospital.inventory
    R = hospital.requests

    # Ii = {ip.index: idx for idx, ip in enumerate(I)}
    # Ri = {rq.index: idx for idx, rq in enumerate(R)}
    # heuristic = [(Ii[m[0]], Ri[m[1]]) for m in set(matches_indices) if m[0] in set(Ii.keys()) and m[1] in set(Ri.keys())] if len(matches_indices) > 0 else []
    heuristic = []

    if num_requests > 0:
        # Solve the MINRAR model, matching the hospital's inventory products to the available requests.
        gurobi_logs, x = minrar_single_hospital(SETTINGS, PARAMS, hospital, I, R, day, e, heuristic)
        alloimmunize(SETTINGS, PARAMS, hospital, e, day, x)
    else:
        gurobi_logs = [0, 2, 0, 0]
        x = np.zeros([len(hospital.inventory),1])
    
    # matches = np.where(x>0)
    # matches_indices = [(I[m[0]].index, R[m[1]].index) for m in zip(matches[0], matches[1])] if len(matches) > 0 else []
    matches_indices = []

    logs = log_results(SETTINGS, PARAMS, logs, gurobi_logs, hospital, e, day, x=x)

    # Update the hospital's inventory, by removing issued or outdated products, increasing product age, and sampling new supply.
    supply_size = hospital.update_inventory(SETTINGS, PARAMS, x, day)
    hospital.inventory += dc.sample_supply_single_day(PARAMS, supply_size)

    return logs, matches_indices


# # Multi-hospital setup: perform matching simultaniously for multiple hospitals, and strategically distribute new supply over all hospitals.
# def simulate_multiple_hospitals(SETTINGS, PARAMS, logs, dc, hospitals, e, day):

#     # For each hospital, update the set of available requests, by removing requests for previous days 
#     # (regardless of whether they were satisfied or not) and sampling new requests that become known today.
#     for hospital in hospitals:
#         hospital.requests = [r for r in hospital.requests if r.day_issuing >= day]
#         num_requests = hospital.sample_requests_single_day(SETTINGS, PARAMS, e, day=day)

#         for rq in hospital.requests:
#             rq.allocated_from_dc = 0

#     # Solve the MINRAR model, matching the inventories of all hospitals and the distribution center to all available requests.
#     gurobi_logs, xh, xdc = minrar_multiple_hospitals(SETTINGS, PARAMS, dc, hospitals, day, e)

#     # Get all distribution center products that were allocated to requests with tomorrow as their issuing date.
#     allocations_from_dc = np.zeros([len(dc.inventory),len(hospitals)])
#     for h in range(len(hospitals)):
#         for r in range(len(hospitals[h].requests)):
#             if hospitals[h].requests[r].day_issuing == (day + 1):
                
#                 issued = np.where(xdc[h][:,r]==1)[0]
#                 for i in issued:
#                     allocations_from_dc[i,h] = 1
#                     hospitals[h].requests[r].allocated_from_dc += 1     # total number of products allocated to this request from DC

#         # Write the results to a csv file.
#         logs = log_results(SETTINGS, PARAMS, logs, gurobi_logs, hospitals[h], e, day, x=xh[h], y=y[h], z=z[h])

#     # Update the hospital's inventory, by removing issued or outdated products and sampling new supply.
#     supply_sizes = np.array([hospitals[h].update_inventory(SETTINGS, PARAMS, xh[h], day) for h in range(len(hospitals))])

#     # Allocate products to each of the hospitals to restock them upto their maximum capacity.
#     model = allocate_remaining_supply_from_dc(SETTINGS, PARAMS, day, dc.inventory, hospitals, supply_sizes, allocations_from_dc)
    
#     # Abstract the remaining supply from the solved model, and ship all allocated products to the hospitals.
#     x = read_transports_solution(model, len(dc.inventory), len(hospitals))
#     for h in range(len(hospitals)):
#         hospitals[h].inventory += [dc.inventory[i] for i in range(len(dc.inventory)) if x[i,h] >= 1]

#     # Update the distribution centers's inventory, by removing shipped or outdated products, and increasing product age.
#     dc.update_inventory(SETTINGS, PARAMS, x, day)

#     return logs
