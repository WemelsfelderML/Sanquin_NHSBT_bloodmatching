import numpy as np
from bitstring import BitArray
import pickle
import os

# After obtaining the optimal variable values from the solved model, write corresponding results to a csv file.
def log_results(SETTINGS, PARAMS, logs, issuing_age, gurobi_logs, dc, hospital, e, day, x=[]):

    # Name of the hospital
    ri = SETTINGS.row_indices[(day, hospital.name)]
    ci = SETTINGS.column_indices

    logs[ri,[ci[c] for c in ["calc time", "gurobi status", "nvars", "ncons"]]] = gurobi_logs

    # Gather some parameters.
    ABOD_names = PARAMS.ABOD
    # ethnicities = ["Caucasian", "African", "Asian"]
    A = PARAMS.antigens
    P = {p : pg for p, pg in PARAMS.patgroups.items() if PARAMS.weekly_demand[hospital.htype][p] > 0}
    I = hospital.inventory
    R = hospital.requests

    # Most results will be calculated only considering the requests that are issued today.
    r_today = [r for r in range(len(R)) if R[r].day_issuing == day]
    r_today.sort(key=lambda r: BitArray(R[r].vector).uint)
    rq_today = [R[r] for r in r_today]

    logs[ri,ci["logged"]] = 1
    logs[ri,ci["day"]] = day
    logs[ri,ci["num patients"]] = len(r_today)                                                                            # number of patients
    logs[ri,ci["num units requested"]] = sum([rq.num_units for rq in rq_today])                                           # number of units requested
    # for eth in ethnicities:
    #     logs[ri,ci[f"num {eth} patients"]] = sum([1 for rq in rq_today if ethnicities[rq.ethnicity] == eth])              # number of patients per ethnicity
    for p in P.keys():
        logs[ri,ci[f"num {P[p]} patients"]] = sum([1 for rq in rq_today if rq.patgroup == p])                             # number of patients per patient group
        logs[ri,ci[f"num units requested {P[p]}"]] = sum([rq.num_units for rq in rq_today if rq.patgroup == p])           # number of units requested per patient group
        # logs[ri,ci[f"num allocated at dc {P[p]}"]] = sum([rq.allocated_from_dc for rq in R if rq.patgroup == p])   # number of products allocated from the distribution center per patient group

    # for u in range(1,13):
    #     logs[ri,ci[f"num requests {u} units"]] = sum([1 for rq in rq_today if rq.num_units == u])                         # number of requests asking for [1-4] units

    logs[ri,ci["num supplied products"]] = sum([1 for ip in I if ip.age == 0])                                   # number of products added to the inventory at the end of the previous day
    for major in ABOD_names:
        # logs[ri,ci[f"num supplied {major}"]] = sum([1 for ip in I if ip.major == major and ip.age == 0])         # number of products per major blood group added to the inventory at the end of the previous day
        # logs[ri,ci[f"num requests {major}"]] = sum([1 for rq in rq_today if rq.major == major])                           # number of patients per major blood group
        logs[ri,ci[f"num {major} in inventory"]] = sum([1 for ip in I if ip.major == major])                     # number of products in inventory per major blood group

    logs[ri,ci["num Fya-Fyb- in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 0) and (ip.vector[10] == 0)])
    logs[ri,ci["num Fya+Fyb- in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 1) and (ip.vector[10] == 0)])
    logs[ri,ci["num Fya-Fyb+ in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 0) and (ip.vector[10] == 1)])
    logs[ri,ci["num Fya+Fyb+ in inventory"]] = sum([1 for ip in I if (ip.vector[9] == 1) and (ip.vector[10] == 1)])
    logs[ri,ci["num R0 in inventory"]] = sum([ip.R0 for ip in I])                                                # number of R0 products in inventory

    xi = x.sum(axis=1)  # For each inventory product i∈I, xi[i] = 1 if the product is issued, 0 otherwise.
    # xr = x.sum(axis=0)  # For each request r∈R, xr[r] = the number of products issued to this request.
    
    # age_sum = 0
    issued_sum = 0
    for r in r_today:
        # Get all products from inventory that were issued to request r.
        issued = np.where(x[:,r]>0)[0]
        rq = R[r]

        mismatch = np.zeros(len(A))
        for ip in [I[i] for i in issued]:

            issuing_age[rq.patgroup, ip.age] += 1
            # age_sum += ip.age
            issued_sum += 1
            # logs[ri,ci[f"{ip.major} to {rq.major}"]] += 1                                 # number of products per major blood group issued to requests per major blood group
            # logs[ri,ci[f"{ethnicities[ip.ethnicity]} to {ethnicities[rq.ethnicity]}"]] += 1                         # number of products per ethnicity issued to requests per ethnicity
            
            # Get all antigens k on which product ip and request rq are mismatched.
            for k in [k for k in range(len(A)) if ip.vector[k] > rq.vector[k]]:
                # Fy(a-b-) should only be matched on Fy(a), not on Fy(b). -> Fy(b-) only mismatch when Fy(a+)
                if (k != 10) or (rq.vector[9] == 1):
                    mismatch[k] = 1
                    logs[ri,ci[f"num mismatched units {P[rq.patgroup]} {A[k]}"]] += 1        # number of mismatched units per patient group and antigen
                     
        for k in range(len(A)):
            logs[ri,ci[f"num mismatched patients {P[rq.patgroup]} {A[k]}"]] += mismatch[k]           # number of mismatched patients per patient group and antigen
            # logs[ri,ci[f"num mismatches {ethnicities[rq.ethnicity]} {A[k]}"]] += mismatch[k]          # number of mismatched patients per patient ethnicity and antigen

    # logs[ri,ci[f"avg issuing age"]] = age_sum / max(1, issued_sum)                        # average age of all issued products

    for ip in [ip for ip in dc.inventory if ip.age >= (PARAMS.max_age-1)]:
        logs[ri,ci["num outdates dc"]] += 1
        logs[ri,ci[f"num outdates dc {ip.major}"]] += 1

    for ip in [I[i] for i in range(len(I)) if (xi[i] == 0) and (I[i].age >= (PARAMS.max_age-1))]:
        logs[ri,ci["num outdates"]] += 1                                                  # number of outdated inventory products
        logs[ri,ci[f"num outdates {ip.major}"]] += 1                                      # number of outdated inventory products per major blood group

    # logs[ri,ci["num unavoidable shortages"]] = max(0, sum([R[r].num_units for r in r_today]) - len(I))                # difference between the number of requested units and number of products in inventory, in case the former is larger
    for r in [r for r in r_today if sum(x[:,r]) < R[r].num_units]:
        rq = R[r]
        logs[ri,ci["num shortages"]] += 1                                                 # number of today's requests that were left unsatisfied
        logs[ri,ci[f"num shortages {rq.major}"]] += 1                                     # number of unsatisfied requests per major blood group
        logs[ri,ci[f"num shortages {P[rq.patgroup]}"]] += 1                                  # number of unsatisfied requests per patient group
        # logs[ri,ci[f"num {P[rq.patgroup]} {int(rq.num_units - xr[r])} units short"]] += 1    # difference between the number units requested and issued


    return logs
