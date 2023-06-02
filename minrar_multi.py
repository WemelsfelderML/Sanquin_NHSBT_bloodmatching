import gurobipy as grb
import numpy as np
import time
import math
import re

from blood import *
from log import *


# Multi-hospital setup: MINRAR model for matching simultaniously in multiple hospitals.
def minrar_multiple_hospitals(SETTINGS, PARAMS, dc, hospitals, day, e):

    start = time.perf_counter()

    ################
    ## PARAMETERS ##
    ################

    # The set of antigens is reshaped to several formats, for different uses within this class.
    A = range(len(PARAMS.antigens))
    # "Fya" = 9
    # "Fyb" = 10

    # Mapping of the patient groups to their index in the list of patient groups.
    P = range(len(SETTINGS.patgroups))

    # Sets of all inventory products and patient requests.
    # Sets of all hospitals, their inventory products and patient requests.
    H = range(len(hospitals))
    R = [hospital.requests for hospital in hospitals]
    Ih = [hospital.inventory for hospital in hospitals]
    Idc = dc.inventory          # Products in the distribution center's inventory.
    
    num_units = [np.array([rq.num_units for rq in R[h]]) for h in H]
    Ihv = [np.array([ip.vector for ip in Ih[h]]) for h in H]  # I × A matrix
    Idcv = np.array([ip.vector for ip in Idc])  # I × A matrix
    Rv = [np.array([rq.vector for rq in R[h]]) for h in H]  # R × A matrix

    Ihs = [(np.ones(Ihv[h].shape) - Ihv[h]) for h in H]     # invert 0s and 1s in Iv for substitution penalty
    Idcs = (np.ones(Idcv.shape) - Idcv)                     # invert 0s and 1s in Iv for substitution penalty
    Rp = [np.zeros([len(R[h]),len(P)]) for h in H]          # one-hot encoded whether request is of some patient group
    for h in H:
        for r in range(len(R[h])):
            Rp[h][r,R[h][r].patgroup] = 1

    Rm = [(np.ones(Rv[h].shape) - Rv[h]) for h in H]      # Count mismatches if inventory product is positive for some antigen, and request is negative.
    for h in H:
        Rm[h][:,10] *= Rv[h][:,9]                   # Only count Fyb mismatches if patient is positive for Fya.

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

    # Get the usability of all inventory products, in both the hospitals and the distribution center, and for all
    # patient requests with respect to the distribution of major blood types in the patient population.
    bih = [np.tile(np.array([ip.get_usability(PARAMS, [hospitals[h]]) for ip in Ih[h]]), (len(R[h]), 1)).T for h in H]
    bidc = [np.tile(np.array([ip.get_usability(PARAMS, [hospitals[h]]) for ip in Idc]), (len(R[h]), 1)).T for h in H]
    brh = [np.tile(np.array([rq.get_usability(PARAMS, [hospitals[h]]) for rq in R[h]]), (len(Ih[h]), 1)) for h in H]
    brdc = [np.tile(np.array([rq.get_usability(PARAMS, [hospitals[h]]) for rq in R[h]]), (len(Idc), 1)) for h in H]


    # Matrices containing a 1 if product i∈I is compatible with request r∈R, 0 otherwise.
    Ch = [precompute_compatibility(SETTINGS, PARAMS, Ihv[h], Rv[h], R[h]) for h in H]    # The product in the hospital's inventory is compatible on major and manditory antigens.
    Cdc = [precompute_compatibility(SETTINGS, PARAMS, Idcv, Rv[h], R[h]) for h in H]     # The product in the distribution center's inventory is compatible on major and manditory antigens.
    Th = [timewise_possible(SETTINGS, PARAMS, Ih[h], R[h], day) for h in H]      # The product in the hospital's inventory is not outdated before issuing date of request.
    Tdc = [timewise_possible(SETTINGS, PARAMS, Idc, R[h], day) for h in H]       # The product in the distribution center's inventory is not outdated before issuing date of request.

    # t[r] = 2 if issuing day of r is today, t[r] = 1 if it is tomorrow, and t[r] = 0 if it is more than one day in the future.
    t = [[1 - min(1, rq.day_issuing - (day+1)) for rq in R[h]] for h in H]

    # Latin hypercube designs for parameter testing.
    LHD = PARAMS.LHD[e]
    LHD_today = [(np.full(len(R[h]),LHD[5]) * t[h]) + np.ones(len(R[h])) for h in H]  # length R
    
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
    # model.Params.PoolSolutions = 50
    # model.Params.PoolGap = 0

    ###############
    ## VARIABLES ##
    ###############

    # For each hospital h∈H:
    # xh: For eachrequest r∈R[h] and product i∈Ih[h] (hospital's inventory), xh[h][i,r] = 1 if r is satisfied by i, 0 otherwise.
    # xdc: For each request r∈R[h] and i∈Idc (distribution center's inventory), xdc[h][i,r] = 1 if r is satisfied by i, 0 otherwise.
    # y: For each request r∈R[h], y[h][r] = 1 if request r can not be fully satisfied (shortage), 0 otherwise.
    # z: For each request r∈R[h] and antigen k∈A, z[h][r,k] = 1 if request r is mismatched on antigen k, 0 otherwise.
    xh = [model.addMVar((len(Ih[h]), len(R[h])), name=f"xh{h}", vtype=grb.GRB.BINARY, lb=0, ub=1) for h in H]
    xdc = [model.addMVar((len(Idc), len(R[h])), name=f"xdc{h}", vtype=grb.GRB.BINARY, lb=0, ub=1) for h in H]

    model.update()
    model.ModelSense = grb.GRB.MINIMIZE

    for h in H:
        for r in range(len(R[h])):

            # Remove variable xh[h][i,r] if the match is not timewise or antigen compatible.
            for i in range(len(Ih[h])):
                if (Ch[h][i,r] == 0) or (Th[h][i,r] == 0):
                    model.remove(xh[h][i,r])
            
            # Remove variable xdc[h][i,r] if the match is not timewise or antigen compatible.
            for i in range(len(Idc)):
                if (Cdc[h][i,r] == 0) or (Tdc[h][i,r] == 0):
                    model.remove(xdc[h][i,r])

            # Remove variable xdc[h][i,r] if the issuing date of request r is today.
            if t[h][r] == 2:
                for i in range(len(Idc)):
                    model.remove(xdc[h][i,r])

    #################
    ## CONSTRAINTS ##
    #################

    for h in H:

        # Force y[r] to 1 if not all requested units are satisfied (either from the hospital's own inventory or from the dc's inventory).
        model.addConstr((y[h] * num_units[h]) + (xh[h].T @ np.ones(len(Ih[h]))) + (xdc[h].T @ np.ones(len(Idc))) >= num_units[h])

        # Force z[r,k] to 1 if at least one of the products i∈I that are issued to request r∈R mismatches on antigen k∈A.
        model.addConstr(((xh[h].T @ Ihv[h]) * Rm[h]) <= z[h] * np.tile(num_units[h], (len(A),1)).T)
        model.addConstr(((xdc[h].T @ Idcv) * Rm[h]) <= z[h] * np.tile(num_units[h], (len(A),1)).T)

        # For each request, the number of products allocated by the hospital and DC together should not exceed the number of units requested.
        model.addConstr((xh[h].T @ np.ones(len(Ih[h]))) + (xdc[h].T @ np.ones(len(Idc))) <= num_units[h])

        # For each inventory product i∈I, ensure that i can not be issued more than once.
        model.addConstr(xh[h].sum(axis=1) <= np.ones(len(Ih[h])))
    model.addConstr(grb.quicksum(xdc[h].sum(axis=1) for h in H) <= np.ones(len(Idc)))
    
    ################
    ## OBJECTIVES ##
    ################

    FIFO_h = [LHD[2] * np.tile(np.array([-0.5 ** ((35 - ip.age - 1) / 5) for ip in Ih[h]]), (len(R[h]), 1)).T for h in H]
    usab_h = [LHD[3] * (bih[h] - brh[h]) for h in H]
    subst_h = [LHD[4] * (Ihs[h] @ ((Rp[h] @ w_subst) * Rv[h]).T) for h in H]

    FIFO_dc = [LHD[2] * np.tile(np.array([-0.5 ** ((35 - ip.age - 1) / 5) for ip in Idc]), (len(R[h]), 1)).T for h in H]
    usab_dc = [LHD[3] * (bidc[h] - brdc[h]) for h in H]
    subst_dc = [LHD[4] * (Idcs @ ((Rp[h] @ w_subst) * Rv[h]).T) for h in H]
    
    xh_penalties = [(FIFO_h[h] + usab_h[h] + subst_h[h]) * np.tile(LHD_today[h],(len(Ih[h]),1)) for h in H]
    xdc_penalties = [(FIFO_dc[h] + usab_dc[h] + subst_dc[h]) * np.tile(LHD_today[h],(len(Idc),1)) for h in H]
    y_penalties = [np.full(len(R[h]),LHD[0]) * LHD_today[h] for h in H]
    z_penalties = [LHD[1] * (Rp[h] @ w) * np.tile(LHD_today[h],(len(A),1)).T for h in H]

    
    model.setObjective(expr = grb.quicksum(grb.quicksum(y_penalties[h] * y[h])
                            + grb.quicksum(grb.quicksum(z_penalties[h] * z[h]))
                            + grb.quicksum(grb.quicksum(xh_penalties[h] * xh[h]))
                            + grb.quicksum(grb.quicksum(xdc_penalties[h] * xdc[h])) for h in H))

    stop = time.perf_counter()
    print(f"model initialization: {(stop - start):0.4f} seconds")

    start = time.perf_counter()
    model.optimize()
    stop = time.perf_counter()
    print(f"Optimize: {(stop - start):0.4f} seconds")

    # sc = model.SolCount
    # print(f"Solutions found: {sc}")

    # Create numpy arrays filled with zeros.
    xh = [np.zeros([len(Ih[h]), len(R[h])]) for h in H]
    xdc = [np.zeros([len(Idc), len(R[h])]) for h in H]
    y = [np.zeros([len(R[h])]) for h in H]
    z = [np.zeros([len(R[h]), len(A)]) for h in H]
    for h in range(len(hospitals)):
        for var in model.getVars():
            var_name = re.split(r'\W+', var.varName)[0]
            if var_name == f"xh{h}":
                index0 = int(re.split(r'\W+', var.varName)[1])
                index1 = int(re.split(r'\W+', var.varName)[2])
                xh[h][index0, index1] = var.X
            if var_name == f"xdc{h}":
                index0 = int(re.split(r'\W+', var.varName)[1])
                index1 = int(re.split(r'\W+', var.varName)[2])
                xdc[h][index0, index1] = var.X
            if var_name == f"y{h}":
                index0 = int(re.split(r'\W+', var.varName)[1])
                y[h][index0] = var.X
            if var_name == f"z{h}":
                index0 = int(re.split(r'\W+', var.varName)[1])
                index1 = int(re.split(r'\W+', var.varName)[2])
                z[h][index0, index1] = var.X

    gurobi_logs = [stop - start, model.status, len(model.getVars()), len(model.getConstrs())]

    # df.loc[(day,dc.name),"calc time"] = stop - start
    # df.loc[(day,dc.name),"gurobi status"] = model.status
    # df.loc[(day,dc.name),"nvars"] = len(model.getVars())
    # df.loc[(day,dc.name),"ncons"] = len(model.getConstrs())


    return gurobi_logs, xh, xdc, y, z


# Multi-hospital setup: allocate products to each of the hospitals to restock them upto their maximum capacity.
def allocate_remaining_supply_from_dc(SETTINGS, PARAMS, day, inventory, hospitals, supply_sizes, allocations_from_dc):

    start = time.perf_counter()

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

    ################
    ## PARAMETERS ##
    ################

    # Sets of all hospitals and of the distribution center's inventory products.
    H = range(len(hospitals))
    I = inventory
    
    # Get the usability of all inventory products with respect to the distribution of major blood types in the patient population.
    bi = np.tile(np.array([ip.get_usability(PARAMS, hospitals, antigens=list(range(3,17))) for ip in I]), (len(H), 1)).T

    ###############
    ## VARIABLES ##
    ###############

    # For each inventory product i∈I, x[i,h] = 1 if product i will be shipped to hospital h, 0 otherwise.
    x = model.addMVar((len(I), len(H)), name='x', vtype=grb.GRB.BINARY, lb=0, ub=1)
    
    model.update()
    model.ModelSense = grb.GRB.MINIMIZE


    #################
    ## CONSTRAINTS ##
    #################

    # Force x[i,h] to 1 if product i∈I was already allocated to hospital h∈H in the previous optimization.
    model.addConstr(x >= allocations_from_dc)

    # Make sure the number of supplied products is at least the necessary amount to restock each hospital completely.
    model.addConstr(x.sum(axis=0) >= supply_sizes)

    # For each inventory product i∈I, ensure that i can not be allocated more than once.
    model.addConstr(x.sum(axis=1) <= 1)


    ################
    ## OBJECTIVES ##
    ################

    # These are all I×R matrices.
    FIFO = np.tile(np.array([-0.5 ** ((35 - ip.age - 1) / 5) for ip in I]), (len(H), 1)).T
    x_penalties = FIFO + bi
    
    model.setObjective(expr = grb.quicksum(grb.quicksum(x_penalties * x)))
    # model.setObjective(expr = quicksum(0.5 ** ((PARAMS.max_age - I[i].age - 1) / 5) * x[i,h] for i in I.keys() for h in H.keys())   # FIFO penalties.
    #                         + quicksum(bi[i] * x[i,h] for i in I.keys() for h in H.keys()))                                         # Product usability on major antigens.


    ##############
    ## OPTIMIZE ##
    ##############

    model.optimize()
    stop = time.perf_counter()
    print(f"allocation from dc: {(stop - start):0.4f} seconds")

    print(PARAMS.status_code[model.status])

    return model


    
