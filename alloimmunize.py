import numpy as np
import pickle

def alloimmunize(SETTINGS, PARAMS, scenario, hospital, day, x):

    # Initialize inventory, requests and antigens.
    I = hospital.inventory
    R = hospital.requests
    A = PARAMS.antigens
    r_SCD = [r for r in range(len(R)) if (R[r].patgroup == 1) and (R[r].day_issuing == day)]

    Iv = np.array([ip.vector for ip in I])     # I × A matrix with a 1 if the inventory product is positive for some antigen, 0 if negative.
    Rv = np.array([rq.vector for rq in R])     # R × A matrix with a 1 if the request is positive for some antigen, 0 if negative.
    Rm = (np.ones(Rv.shape) - Rv)           # R × A matrix with a 1 if the request is negative for some antigen, 0 if positive.
    Rm[:,9] *= Rv[:,8]                        # Only count Fyb mismatches if patient is positive for Fya.

    for r in r_SCD:                                     # Loop over all requests for SCD patients.
        for i in range(len(I)):                         # Loop over all inventory products.
            for k in A.keys():                          # Loop over all antigens.
                if x[i,r] * Iv[i,k] * Rm[r,k] > 0:      # If there is a mismatch ...
                    R[r].mism_units[k] += 1             # Increase the number of mismatched units received by 1.
                    if np.random.rand() <= PARAMS.alloimmunization_risks[int(min(R[r].mism_units[k], 20)),k]:  
                        R[r].antibodies[k] = 1          # Alloimmunization happens with given probability.
                    
    requests_SCD = np.array([[R[r].num_units] + list(R[r].vector) + list(R[r].antibodies) + list(R[r].mism_units) + [R[r].index] for r in r_SCD])
    

    with open(SETTINGS.generate_filename(method=SETTINGS.method, output_type="results", subtype="patients", scenario=scenario, name=hospital.name, day=day)+".pickle", 'wb') as f:
        pickle.dump(requests_SCD, f, pickle.HIGHEST_PROTOCOL)
