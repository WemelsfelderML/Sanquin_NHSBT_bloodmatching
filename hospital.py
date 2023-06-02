import pandas as pd
import pickle
import sys

from blood import *

class Hospital():
    
    def __init__(self, SETTINGS, PARAMS, htype, e):

        self.htype = htype                                                              # Hospital type ("regional" or "university")
        self.name = f"{htype[:3]}_{e}"                                                  # Name for the hospital.
        self.avg_daily_demand = SETTINGS.avg_daily_demand[htype]                        # Average daily number of units requested within this hospital.
        self.inventory_size = SETTINGS.inv_size_factor_hosp * self.avg_daily_demand     # Size of the hospital's inventory.

        # Read the demand that was generated using SETTINGS.mode = "demand".
        data = pd.read_csv(SETTINGS.home_dir + f"demand/{self.avg_daily_demand}/{SETTINGS.test_days + SETTINGS.init_days}/{htype}_{e}.csv")
        data = data.loc[(data["Day Needed"] < 42) | (data["Patient Type"] != 1)]  # Only sample SCD patients for the first 6 weeks (they will return afterwards).
        self.demand_data = []
        for day in range(SETTINGS.init_days + SETTINGS.test_days):
            indices = data.loc[data["Day Available"] == day].index
            self.demand_data.append(np.array(data.loc[indices,["Ethnicity", "Patient Type"] + list(PARAMS.antigens.values()) + ["Num Units", "Day Needed", "Day Available"]]).astype(int))

        # TODO maybe have inventory and requests already be a index-product dictionary as used in MINRAR?
        self.inventory = []
        self.requests = []


    # At the end of a day in the simulation, remove all issued or outdated products, and increase the age of remaining products.
    def update_inventory(self, SETTINGS, PARAMS, x, day):

        I = {i : self.inventory[i] for i in range(len(self.inventory))}
        remove = []

        # Remove all products form inventory that were issued to requests with today as their issuing date
        for r in [r for r in range(len(self.requests)) if self.requests[r].day_issuing == day]:
            remove += list(np.where(x[:,r]==1)[0])

        # If a product will be outdated at the end of this day, remove from inventory, otherwise increase its age.
        for i in I.keys():
            if I[i].age >= (PARAMS.max_age-1):
                remove.append(i)
            else:
                I[i].age += 1
 
        self.inventory = [I[i] for i in I.keys() if i not in remove]

        # Return the number of products to be supplied, in order to fill the inventory upto its maximum capacity.
        return max(0, self.inventory_size - len(self.inventory))


    def sample_requests_single_day(self, SETTINGS, PARAMS, e, day = 0):

        # Select the part of the demand scenario belonging to the given day.
        data = self.demand_data[day]

        zeros_A = np.zeros(len(PARAMS.antigens))
        requests = [Blood(PARAMS, ethnicity=rq[0], patgroup=rq[1], major=rq[2:5], minor=rq[5:19], num_units=rq[19], day_issuing=rq[20], day_available=rq[21], antibodies=zeros_A.copy(), mism_units=zeros_A.copy()) for rq in data]

        if day >= (5*7):
            data_SCD = unpickle(SETTINGS.home_dir + f"wip/{SETTINGS.model_name}/{e}/patients_{SETTINGS.strategy}_{self.htype[:3]}/{day-(5*7)}")
            requests += [Blood(PARAMS, ethnicity=1, patgroup=1, major=rq[1:4], minor=rq[4:18], num_units=rq[0], day_issuing=day+7, day_available=day, antibodies=rq[18:35], mism_units=rq[35:]) for rq in data_SCD]

        self.requests += requests

    def pickle(self, path):
        with open(path+".pickle", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)