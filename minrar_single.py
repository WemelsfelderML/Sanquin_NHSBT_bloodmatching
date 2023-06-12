import gurobipy as grb
import numpy as np
import time
import re
import math

from blood import *
from log import *

# Single-hospital setup: MINRAR model for matching within a single hospital.
def minrar_single_hospital(SETTINGS, PARAMS, hospital, day, e):

    start = time.perf_counter()

    ################
    ## PARAMETERS ##
    ################

    # The set of antigens is reshaped to several formats, for different uses within this class.
    A = range(len(PARAMS.antigens))

    # Mapping of the patient groups to their index in the list of patient groups.
    P = PARAMS.patgroups.keys()

    # Sets of all inventory products and patient requests.
    I = hospital.inventory
    R = hospital.requests
    
    num_units = np.array([rq.num_units for rq in R])
    Iv = np.array([ip.vector for ip in I])      # I × A matrix, antigens for each inventory product
    Rv = np.array([rq.vector for rq in R])      # R × A matrix, antigens for each request
    Rb = np.array([rq.antibodies for rq in R])  # R × A matrix, antibodies for each request

    Is = (np.ones(Iv.shape) - Iv)       # invert 0s and 1s in Iv for substitution penalty
    Rp = np.zeros([len(R),len(P)])      # one-hot encoded whether request is of some patient group
    for r in range(len(R)):
        Rp[r][R[r].patgroup] = 1

    Rm = (np.ones(Rv.shape) - Rv)       # Count mismatches if inventory product is positive for some antigen, and request is negative.
    Rm[:,10] *= Rv[:,9]                 # Only count Fyb mismatches if patient is positive for Fya.

    IR_SCD = np.zeros([len(I),len(R)])
    for r in [r for r in range(len(R)) if R[r].patgroup == 0]:
        IR_SCD[:,r] = np.ones(len(I))
    IR_nonSCD = np.ones([len(I),len(R)]) - IR_SCD

    # Get the usability of all inventory products and patient requests with respect to the
    # distribution of major blood types in the patient population.
    bi = np.tile(np.array([ip.get_usability(PARAMS, [hospital]) for ip in I]), (len(R), 1)).T
    br = np.tile(np.array([rq.get_usability(PARAMS, [hospital]) for rq in R]), (len(I), 1))

    # Matrices containing a 1 if product i∈I is compatible with request r∈R, 0 otherwise.
    C = precompute_compatibility(SETTINGS, PARAMS, R, Iv, Rv, Rb)   # The product is compatible with the request on major and manditory antigens
    T = timewise_possible(SETTINGS, PARAMS, I, R, day)          # The product is not outdated before issuing date of request.

    # For each request r∈R, t[r] = 1 if the issuing day is today, 0 if it lies in the future.
    t = [1 - min(1, rq.day_issuing - day) for rq in R]
    
    # Retrieve the antigen (and patient group) weights.
    if "patgroups" in SETTINGS.strategy:
        w = PARAMS.patgroup_weights
    else:
        if "relimm" in SETTINGS.strategy:
            w = PARAMS.relimm_weights
        elif "major" in SETTINGS.strategy:
            w = PARAMS.major_weights
        w = np.tile(w, (len(P), 1))

    w_subst = w.copy()
    w_subst[:5,:] = 0
    w_subst[:,:3] = 0

    # Latin hypercube designs for parameter testing.
    LHD = PARAMS.LHD[e]
    LHD_today = (np.full(len(R),LHD[6]) * t) + np.ones(len(R))  # length R
    

    ############
    ## GUROBI ##
    ############

    model = grb.Model(name="model")
    if SETTINGS.show_gurobi_output == False:
        model.Params.LogToConsole = 0
    if SETTINGS.gurobi_threads != None:
        model.setParam('Threads', SETTINGS.gurobi_threads)
    if SETTINGS.gurobi_timeout != None:
        model.setParam('TimeLimit', SETTINGS.gurobi_timeout)

    model.Params.PoolSearchMode = 1
    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 500
    # model.Params.PoolGap = 0


    ###############
    ## VARIABLES ##
    ###############

    # x: For each request r∈R and inventory product i∈I, x[i,r] = 1 if r is satisfied by i, 0 otherwise.
    x = model.addMVar((len(I), len(R)), name='x', vtype=grb.GRB.BINARY, lb=0, ub=1)

    model.update()
    model.ModelSense = grb.GRB.MINIMIZE

    # Remove variable x[i,r] if the match is not timewise or antigen compatible.
    for r in range(len(R)):
        for i in range(len(I)):
            if (C[i,r] == 0) or (T[i,r] == 0):
                model.remove(x[i,r])
            if (I[i].age > 14) and (R[r].patgroup == 1):
                model.remove(x[i,r])

    #################
    ## CONSTRAINTS ##
    #################

    # Each request can not receive more products than requested.
    # model.addConstrs(quicksum(x[i,r] for i in I.keys()) <= R[r].num_units for r in R.keys())
    print(num_units)
    print(np.zeros([len(I), 1]).T @ np.ones(len(I)))
    model.addMConstr((x.T @ np.ones(len(I))) <= num_units)

    # For each inventory product i∈I, ensure that i can not be issued more than once.
    # model.addConstrs(quicksum(x[i,r] for r in R.keys()) <= 1 for i in I.keys())
    model.addMConstr(x.sum(axis=1) <= 1)


    ################
    ## OBJECTIVES ##
    ################

    # These are all I×R matrices.
    short = np.full([len(I),len(R)], LHD[0] * -1)
    mism = LHD[1] * (Iv @ ((Rp @ w) * Rm).T)
    youngblood = LHD[2] * IR_SCD * np.tile(np.array([(math.exp(ip.age - 8.5) - ip.age) / 238.085 for ip in I]), (len(R), 1)).T    # /238.085 is for normalization
    FIFO = LHD[3] * IR_nonSCD * np.tile(np.array([-0.5 ** ((35 - ip.age - 1) / 5) for ip in I]), (len(R), 1)).T
    usab = LHD[4] * (bi - br)
    subst = LHD[5] * (Is @ ((Rp @ w_subst) * Rv).T) 

    x_penalties = (short + mism + youngblood + FIFO + usab + subst) * np.tile(LHD_today,(len(I),1))
    
    model.setObjective(expr = grb.quicksum(grb.quicksum(x_penalties * x)))

    stop = time.perf_counter()
    print(f"model initialization: {(stop - start):0.4f} seconds")

    start = time.perf_counter()
    model.optimize()
    stop = time.perf_counter()
    print(f"Optimize: {(stop - start):0.4f} seconds")

    x = np.zeros([len(I), len(R)])
    for var in model.getVars():
        var_name = re.split(r'\W+', var.varName)[0]
        if var_name == "x":
            index0 = int(re.split(r'\W+', var.varName)[1])
            index1 = int(re.split(r'\W+', var.varName)[2])
            x[index0, index1] = var.X

    gurobi_logs = [stop - start, model.status, len(model.getVars()), len(model.getConstrs())]

    return gurobi_logs, x