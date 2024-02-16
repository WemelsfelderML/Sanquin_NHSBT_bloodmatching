import math
import os
import pandas as pd
import pickle
import multiprocessing


from blood import *


# Generate a given number of supply files, where each file contains enough supply for one simulation episode.
def generate_supply(SETTINGS, PARAMS):

    size = PARAMS.supply_size           # Total number of products to be generated.
    name = '-'.join([str(SETTINGS.n_hospitals[ds])+ds for ds in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[ds] > 0])

    # Make sure all required folders exist.
    path = SETTINGS.home_dir + "supply"
    if os.path.exists(path) == False:
        os.mkdir(path)

    path = SETTINGS.home_dir + f"supply/{size}"
    if os.path.exists(path) == False:
        os.mkdir(path)



    # Find already existing supply files of the chosen size and duration, and make sure not to overwrite them.
    i = 0
    while os.path.exists(SETTINGS.generate_filename(output_type="supply", size=size, name=name, e=i)+".pickle"):
        i += 1

    print(f"Generating supply '{name}_{SETTINGS.episodes[0]+i}' to '{name}_{SETTINGS.episodes[1]+i}'.")

    processes = []
    for e in range(SETTINGS.episodes[0],SETTINGS.episodes[1]):
        p = multiprocessing.Process(target=generate_episode, args=(SETTINGS, PARAMS, name, size, e+i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def generate_episode(SETTINGS, PARAMS, name, size, episode):

    # Generate the required number of products.
    supply = generate_products(SETTINGS, PARAMS, size)

    if len(SETTINGS.majors_init) == 3:
        htype = [h for h in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[h] == 1][0]
        inventory_size = SETTINGS.inv_size_factor_hosp * round(sum(PARAMS.weekly_demand[htype])/7)
        supply[:inventory_size, :3] = np.tile(SETTINGS.majors_init, (inventory_size,1))
    
    # Write numpy array to pickle file.
    with open(SETTINGS.generate_filename(output_type="supply", size=size, name=name, e=episode)+".pickle", 'wb') as f:
        pickle.dump(supply, f, pickle.HIGHEST_PROTOCOL)


def generate_minor_antigens(PARAMS, ethnicity, size):

    Kell = np.array(random.choices(PARAMS.Kell_phenotypes, weights = PARAMS.Kell_prevalences[ethnicity], k=size))
    MNS = np.array(random.choices(PARAMS.MNS_phenotypes, weights = PARAMS.MNS_prevalences[ethnicity], k=size))
    Duffy = np.array(random.choices(PARAMS.Duffy_phenotypes, weights = PARAMS.Duffy_prevalences[ethnicity], k=size))
    Kidd = np.array(random.choices(PARAMS.Kidd_phenotypes, weights = PARAMS.Kidd_prevalences[ethnicity], k=size))
    
    return np.concatenate([Kell, Duffy, Kidd, MNS], axis=1)


# Generate a list of products with a specific ethnic distribution and a specific ABODistribution, in random order
def generate_products(SETTINGS, PARAMS, size):

    R0_size = int(size * 0.095)
    non_R0_size = size - R0_size


    #################
    # NON-R0 SUPPLY #
    #################

    # Draw randomly from donation distribution of A, B, and Rh antigens to get the first 90.5% of the RBC issues
    non_R0_phenotypes = PARAMS.donor_ABO_Rhesus_distr[:,:7]
    non_R0_prevalences = PARAMS.donor_ABO_Rhesus_distr[:,7]

    # non_R0_size × 7 array with ABO-Rhesus phenotypes for 90.5% of donations.
    non_R0_ABRh = np.array(random.choices(non_R0_phenotypes, weights = non_R0_prevalences, k=non_R0_size))

    # # Sample 1% of remaining antigens for non-R0 donations from African population.
    non_R0_part_African = int(round(non_R0_size*0.01))
    non_R0_minor_African = generate_minor_antigens(PARAMS, 1, non_R0_part_African)

    # # Sample 99% of remaining antigens for non-R0 donations from Caucasian population.
    non_R0_part_Caucasian = non_R0_size - non_R0_part_African
    non_R0_minor_Caucasian = generate_minor_antigens(PARAMS, 0, non_R0_part_Caucasian)
    
    # Put all sampled non-R0 demand together.
    non_R0_supply = np.concatenate([non_R0_ABRh, np.concatenate([non_R0_minor_African, non_R0_minor_Caucasian], axis=0)], axis=1)


    #############
    # R0 SUPPLY #
    #############

    # Fix R0 phenotype for all R0 supply.
    R0_Rh = np.tile([1,0,1,0,1], (R0_size, 1))

    # Sample antigens A and B according to their frequencies in national R0 donations.
    AB = np.array(random.choices(PARAMS.ABO_phenotypes, weights = PARAMS.ABO_prevalences[3], k=R0_size))
    
    # Sample 20.5% of remaining antigens for R0 donations from African population.
    R0_part_African = int(round(R0_size*0.205))
    R0_minor_African = generate_minor_antigens(PARAMS, 1, R0_part_African)

    # Sample 79.5% of remaining antigens for R0 donations from Caucasian population.
    R0_part_Caucasian = R0_size - R0_part_African
    R0_minor_Caucasian = generate_minor_antigens(PARAMS, 0, R0_part_Caucasian)
    
    # Put all sampled R0 demand together.
    R0_supply = np.concatenate([AB, R0_Rh, np.concatenate([R0_minor_African, R0_minor_Caucasian], axis=0)], axis=1)


    #####################
    # MERGE AND SHUFFLE #
    #####################

    supply = np.concatenate([non_R0_supply, R0_supply], axis=0)

    np.random.shuffle(supply)

    return supply

