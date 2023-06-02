import numpy as np
from bitstring import BitArray
import pickle
import os

# After obtaining the optimal variable values from the solved model, write corresponding results to a csv file.
def log_results(SETTINGS, PARAMS, logs, gurobi_logs, hospital, e, day, x=[]):

    # Name of the hospital (e.g. "reg_2" or "uni_0").
    name = hospital.name
    ri = SETTINGS.row_indices[(day, name)]
    ci = SETTINGS.column_indices

    logs[ri,[ci[c] for c in ["calc time", "gurobi status", "nvars", "ncons"]]] = gurobi_logs

    # Gather some parameters.
    ABOD_names = PARAMS.ABOD
    ethnicities = ["Caucasian", "African", "Asian"]
    A = PARAMS.antigens
    P = PARAMS.patgroups
    I = hospital.inventory
    R = hospital.requests

    # Most results will be calculated only considering the requests that are issued today.
    r_today = [r for r in range(len(R)) if R[r].day_issuing == day]
    r_today.sort(key=lambda r: BitArray(R[r].vector).uint)
    rq_today = [R[r] for r in r_today]

    logs[ri,ci["logged"]] = 1
    logs[ri,ci["num patients"]] = len(r_today)                                                                            # number of patients
    logs[ri,ci["num units requested"]] = sum([rq.num_units for rq in rq_today])                                           # number of units requested
    for eth in ethnicities:
        logs[ri,ci[f"num {eth} patients"]] = sum([1 for rq in rq_today if ethnicities[rq.ethnicity] == eth])              # number of patients per ethnicity
    for p in range(len(P)):
        logs[ri,ci[f"num {P[p]} patients"]] = sum([1 for rq in rq_today if rq.patgroup == p])                             # number of patients per patient group
        logs[ri,ci[f"num units requested {P[p]}"]] = sum([rq.num_units for rq in rq_today if rq.patgroup == p])           # number of units requested per patient group
        logs[ri,ci[f"num allocated at dc {P[p]}"]] = sum([rq.allocated_from_dc for rq in R if rq.patgroup == p])   # number of products allocated from the distribution center per patient group

    for u in range(1,5):
        logs[ri,ci[f"num requests {u} units"]] = sum([1 for rq in rq_today if rq.num_units == u])                         # number of requests asking for [1-4] units

    logs[ri,ci["num supplied products"]] = sum([1 for ip in I if ip.age == 0])                                   # number of products added to the inventory at the end of the previous day
    for major in ABOD_names:
        logs[ri,ci[f"num supplied {major}"]] = sum([1 for ip in I if ip.major == major and ip.age == 0])         # number of products per major blood group added to the inventory at the end of the previous day
        logs[ri,ci[f"num requests {major}"]] = sum([1 for rq in rq_today if rq.major == major])                           # number of patients per major blood group
        logs[ri,ci[f"num {major} in inventory"]] = sum([1 for ip in I if ip.major == major])                     # number of products in inventory per major blood group

    logs[ri,ci["num Fya-Fyb- in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 0) and (ip.vector[10] == 0)])
    logs[ri,ci["num Fya+Fyb- in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 1) and (ip.vector[10] == 0)])
    logs[ri,ci["num Fya-Fyb+ in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 0) and (ip.vector[10] == 1)])
    logs[ri,ci["num Fya+Fyb+ in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 1) and (ip.vector[10] == 1)])
    logs[ri,ci["num R0 in inventory"]] = sum([ip.R0 for ip in I])                                                # number of R0 products in inventory

    # num_units = np.array([rq.num_units for rq in R])
    # Iv = np.array([ip.vector for ip in I])
    # Rv = np.array([rq.vector for rq in R])
    # bi = np.array([ip.get_usability(PARAMS, [hospital]) for ip in I])
    # br = np.array([rq.get_usability(PARAMS, [hospital]) for rq in R])
    # t = [1 - min(1, rq.day_issuing - day) for rq in R]
    # LHD = PARAMS.LHD[e]
    # LHD_today = (np.full(len(R),LHD[5]) * t) + np.ones(len(R))                  # length R
    # FIFO_penalties = np.array([-0.5 ** ((35 - ip.age - 1) / 5) for ip in I])    # length I
    # Is = (np.ones(Iv.shape) - Iv) # substitution
    # w = PARAMS.mismatch_weights
    # w_usab = w.copy()
    # w_usab[:5,:] = 0
    # w_usab[:,:3] = 0
    # onesR = np.ones(len(R)).T
    # Rp = np.zeros([len(R),len(P)])
    # for r in range(len(R)):
    #     Rp[r][R[r].patgroup] = 1
    # logs[ri,ci["objval shortages"] = onesR @ ((np.full(len(R),LHD[0]) * y) * LHD_today)
    # logs[ri,ci["objval mismatches"] = onesR @ ((((Rp @ w) * z) @ np.ones(len(A))) * LHD_today)
    # logs[ri,ci["objval FIFO"] = onesR @ ((x.T @ FIFO_penalties) * LHD_today)
    # logs[ri,ci["objval usability"] = onesR @ ((x.T @ (bi - (x @ br))) * LHD_today)
    # logs[ri,ci["objval substitution"] = onesR @ (((((x.T @ Is) * Rv) @ w_usab.T) @ np.ones(len(P))) * LHD_today)

    xi = x.sum(axis=1)  # For each inventory product i∈I, xi[i] = 1 if the product is issued, 0 otherwise.
    xr = x.sum(axis=0)  # For each request r∈R, xr[r] = the number of products issued to this request.

    # num_bloodgroups = 2**len(PARAMS.antigens)
    # state = np.zeros([num_bloodgroups, PARAMS.max_age + PARAMS.max_lead_time + 1])
    # for ip in I:
    #     state[BitArray(ip.vector).uint, ip.age] += 1
    # for rq in R:
    #     state[BitArray(rq.vector).uint, -2 -(day - rq.day_issuing)] += rq.num_units

    # states = []
    # Q_matrices = []
    
    age_sum = 0
    issued_sum = 0
    for r in r_today:
        # Get all products from inventory that were issued to request r.
        issued = np.where(x[:,r]==1)[0]
        rq = R[r]

        # state[:,-1] = np.zeros([len(state)])
        # state[BitArray(rq.vector).uint,-1] = 1

        mismatch = np.zeros(len(A))
        for ip in [I[i] for i in issued]:

            # df_matches.loc[len(df_matches),:] = [day, BitArray(ip.vector).uint, BitArray(rq.vector).uint]

            # Q_matrix = np.zeros([num_bloodgroups])
            # Q_matrix[BitArray(ip.vector).uint] = 1

            # states.append(np.ravel(state).reshape(1,-1))
            # Q_matrices.append(np.ravel(Q_matrix).reshape(1,-1))

            # state[BitArray(rq.vector).uint, -2] -= 1
            # state[BitArray(ip.vector).uint, ip.age] -= 1

            age_sum += ip.age
            issued_sum += 1
            logs[ri,ci[f"{ip.major} to {rq.major}"]] += 1                                 # number of products per major blood group issued to requests per major blood group
            logs[ri,ci[f"{ethnicities[ip.ethnicity]} to {ethnicities[rq.ethnicity]}"]] += 1                         # number of products per ethnicity issued to requests per ethnicity
            
            # Get all antigens k on which product ip and request rq are mismatched.
            for k in [k for k in range(len(A)) if ip.vector[k] > rq.vector[k]]:
                # Fy(a-b-) should only be matched on Fy(a), not on Fy(b). -> Fy(b-) only mismatch when Fy(a+)
                if (k != 10) or (rq.vector[9] == 1):
                    mismatch[k] = 1
                    logs[ri,ci[f"num mismatched units {P[rq.patgroup]} {A[k]}"]] += 1        # number of mismatched units per patient group and antigen
                     
        for k in range(len(A)):
            logs[ri,ci[f"num mismatches {P[rq.patgroup]} {A[k]}"]] += mismatch[k]           # number of mismatched patients per patient group and antigen
            logs[ri,ci[f"num mismatches {ethnicities[rq.ethnicity]} {A[k]}"]] += mismatch[k]          # number of mismatched patients per patient ethnicity and antigen

    logs[ri,ci[f"avg issuing age"]] = age_sum / max(1, issued_sum)                        # average age of all issued products

    for ip in [I[i] for i in range(len(I)) if (xi[i] == 0) and (I[i].age >= (PARAMS.max_age-1))]:
        logs[ri,ci["num outdates"]] += 1                                                  # number of outdated inventory products
        logs[ri,ci[f"num outdates {ip.major}"]] += 1                                      # number of outdated inventory products per major blood group

    logs[ri,ci["num unavoidable shortages"]] = max(0, sum([R[r].num_units for r in r_today]) - len(I))                # difference between the number of requested units and number of products in inventory, in case the former is larger
    for r in [r for r in r_today if sum(x[:,r]) < R[r].num_units]:
        rq = R[r]
        logs[ri,ci["num shortages"]] += 1                                                 # number of today's requests that were left unsatisfied
        logs[ri,ci[f"num shortages {rq.major}"]] += 1                                     # number of unsatisfied requests per major blood group
        logs[ri,ci[f"num shortages {P[rq.patgroup]}"]] += 1                                  # number of unsatisfied requests per patient group
        logs[ri,ci[f"num {P[rq.patgroup]} {int(rq.num_units - xr[r])} units short"]] += 1    # difference between the number units requested and issued


    # if (day >= SETTINGS.init_days) and (len(states) >= 1):

    #     part = int(np.floor((day-SETTINGS.init_days)/73))

    #     htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])[:3]
    #     states_path = SETTINGS.home_dir + f"NN_training_data/{htype}_{''.join(PARAMS.antigens.values())}/states_{e}_{part}.pickle"
    #     q_path = SETTINGS.home_dir + f"NN_training_data/{htype}_{''.join(PARAMS.antigens.values())}/Q_matrices_{e}_{part}.pickle"

    #     # states = states[0]
    #     # Q_matrices = Q_matrices[0]
    #     states = np.concatenate(states, axis=0)
    #     Q_matrices = np.concatenate(Q_matrices, axis=0)

    #     # if (day-SETTINGS.init_days) % 73 != 0:
    #     if os.path.exists(states_path):
    #         states = np.concatenate((unpickle(states_path), states), axis=0)
    #         Q_matrices = np.concatenate((unpickle(q_path), Q_matrices), axis=0)

    #     with open(states_path, 'wb') as f:
    #         pickle.dump(states, f, pickle.HIGHEST_PROTOCOL)
    #     with open(q_path, 'wb') as f:
    #         pickle.dump(Q_matrices, f, pickle.HIGHEST_PROTOCOL)
    
    return logs


def unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)