import pandas as pd
import pickle
import sys

from blood import *

class Hospital():
    
    def __init__(self, SETTINGS, PARAMS, htype, e):

        self.htype = htype                                                              # Hospital type ("UCLH", "NMUH", "WH")
        self.name = f"{htype}_{e}"                                                      # Name for the hospital.
        self.avg_daily_demand = round(sum(PARAMS.weekly_demand[htype])/7)               # Average daily number of units requested within this hospital.
        self.inventory_size = SETTINGS.inv_size_factor_hosp * self.avg_daily_demand     # Size of the hospital's inventory.

        # Read the demand that was generated using SETTINGS.mode = "demand".
        data = pd.read_csv(SETTINGS.generate_filename(output_type="demand", size=SETTINGS.test_days + SETTINGS.init_days, name=htype, e=e)+".csv")
        data["index"] = range(len(data))

        # Only sample SCD patients for the first 6 weeks after the initialization period (they will return afterwards).
        data = data.loc[(data["day issuing"] < SETTINGS.init_days + 42) | (data["patgroup"] != 1)]

        self.demand_data = []
        for day in range(SETTINGS.init_days + SETTINGS.test_days):
            indices = data.loc[data["day available"] == day].index
            self.demand_data.append(np.array(data.loc[indices,["ethnicity", "patgroup"] + list(PARAMS.antigens.values()) + ["num units", "day issuing", "day available", "index"]]).astype(int))

        self.inventory = []
        self.requests = []


    # At the end of a day in the simulation, remove all issued or outdated products, and increase the age of remaining products.
    def update_inventory(self, SETTINGS, PARAMS, x, day):

        I = self.inventory
        remove = []

        # Remove all products form inventory that were issued to requests with today as their issuing date.
        for r in [r for r in range(len(self.requests)) if self.requests[r].day_issuing == day]:
            remove += list(np.where(x[:,r]>0)[0])

        # If a product will be outdated at the end of this day, remove from inventory, otherwise increase its age.
        for i in range(len(I)):
            if I[i].age >= (PARAMS.max_age-1):
                remove.append(i)
            else:
                I[i].age += 1
 
        self.inventory = [I[i] for i in range(len(I)) if i not in remove]

        # Return the number of products to be supplied, in order to fill the inventory upto its maximum capacity.
        return max(0, self.inventory_size - len(self.inventory))


    def sample_requests_single_day(self, SETTINGS, PARAMS, scenario, e, day = 0):

        # Select the part of the demand scenario belonging to the given day.
        data = self.demand_data[day]

        zeros_A = np.zeros(len(PARAMS.antigens))
        requests = [Blood(PARAMS, index=rq[20], ethnicity=rq[0], patgroup=rq[1], antigens=rq[2:17], num_units=rq[17], day_issuing=rq[18], day_available=rq[19], antibodies=zeros_A.copy(), mism_units=zeros_A.copy()) for rq in data]

        if day >= (SETTINGS.init_days + (5*7)):
            data_SCD = unpickle(SETTINGS.generate_filename(method=SETTINGS.method, output_type="results", subtype="patients", scenario=scenario, name=self.name, day=day-(5*7)))
            requests += [Blood(PARAMS, index=rq[49], ethnicity=1, patgroup=1, antigens=rq[1:16], num_units=rq[0], day_issuing=day+7, day_available=day, antibodies=rq[16:31], mism_units=rq[31:46]) for rq in data_SCD]

        self.requests += requests

        # In order to check if there are any requests to perform matching on.
        return len(self.requests)


    def pickle(self, path):
        with open(path+".pickle", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)