import pandas as pd
import pickle
import sys

from blood import *

class Distribution_center():
    
    def __init__(self, SETTINGS, PARAMS, hospitals, e, supply_index = 0):

        # Name for the distribution center (currently not used)
        self.name = f"dc_{e}"

        # Read the supply that was generated using SETTINGS.mode = "supply"
        self.supply_data = unpickle(SETTINGS.home_dir + f"supply/{PARAMS.supply_size}/{'-'.join([str(SETTINGS.n_hospitals[ds])+ds for ds in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[ds] > 0])}_{e}")

        # Keep track of the supply index to know which item of the supply data to read next.
        self.supply_index = supply_index

        inventory = []
        # In the multi-hospital scenario, the distribution center also has its own inventory.
        if len(hospitals) > 1:

            self.inventory_size = SETTINGS.inv_size_factor_dc * sum([hospital.avg_daily_demand for hospital in hospitals])

            # Initialize the inventory with products from the supply data, where the product's age is uniformly distributed between 0 and the maximum shelf life.
            
            n_products = round(self.inventory_size / PARAMS.max_age)
            for age in range(PARAMS.max_age):
                inventory += self.sample_supply_single_day(PARAMS, n_products, age)

        self.inventory = inventory


    # Update the distribution centers's inventory at the end of a day in the simulation.
    def update_inventory(self, SETTINGS, PARAMS, x, day):

        I = {i : self.inventory[i] for i in range(len(self.inventory))}
        remove = []

        # Remove all products from inventory that will be shipped to a hospital.
        xi = x.sum(axis=1)
        remove += [i for i in I.keys() if xi[i] >= 1]

        # If a product will be outdated at the end of this day, remove from inventory, otherwise increase its age.
        for i in I.keys():
            if I[i].age >= (PARAMS.max_age-1):
                remove.append(i)
            else:
                I[i].age += 1

        self.inventory = [I[i] for i in I.keys() if i not in remove]
        
        # Supply the inventory upto its capacity with new products from the supply data.
        self.inventory += self.sample_supply_single_day(PARAMS, max(0, self.inventory_size - len(self.inventory)))


    # Read the required number of products from the supply data and add these products to the distribution center's inventory.
    def sample_supply_single_day(self, PARAMS, n_products, age = 0):

        # Select the next part of the supply scenario.
        data = self.supply_data[self.supply_index : self.supply_index + n_products, :]
        supply = [Blood(PARAMS, index=self.supply_index+i, antigens=data[i], age=age) for i in range(len(data))]

        self.supply_index += n_products

        return supply


    def pickle(self, path):
        with open(path+".pickle", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)