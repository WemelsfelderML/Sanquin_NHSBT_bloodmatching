import gurobipy as grb
import numpy as np
import time
from scipy.spatial import distance


# Single-hospital setup: MINRAR model for matching within a single hospital.
def create_LHD(dim, n_points):

    start = time.perf_counter()
    
    ############
    ## GUROBI ##
    ############

    model = grb.Model(name="model")
    model.Params.LogToConsole = 1
    model.setParam('Threads', 8)
    # model.setParam('TimeLimit', ...)
    model.Params.PoolSearchMode = 1

    ###############
    ## VARIABLES ##
    ###############

    # x: For each request r∈R and inventory product i∈I, x[i,r] = 1 if r is satisfied by i, 0 otherwise.
    x = model.addMVar((n_points, dim), name="x", vtype=grb.GRB.INTEGER, lb=1, ub=n_points)
    dist = model.addVar(name="dist", vtype=grb.GRB.CONTINUOUS, lb=0)

    model.update()
    model.ModelSense = grb.GRB.MAXIMIZE

    #################
    ## CONSTRAINTS ##
    #################

    for d in range(dim):
        model.addConstr(len(np.unique(x[:,d])) >= n_points)

    for i in range(n_points-1):
        for j in range(i+1, n_points):
            model.addConstr(((x[i,:] - x[j,:])*(x[i,:] - x[j,:])).sum() >= dist)

    ################
    ## OBJECTIVES ##
    ################
    
    model.setObjective(expr = z)

    stop = time.perf_counter()
    print(f"model initialization: {(stop - start):0.4f} seconds")

    start = time.perf_counter()
    model.optimize()
    stop = time.perf_counter()
    print(f"Optimize: {(stop - start):0.4f} seconds")

    x_solved = x.X

    return x_solved


if __name__ == "__main__":

    HOME_DIR = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Sanquin_NHSBT_bloodmatching/"
    dim = 6
    n_points = 500

    LHD = create_LHD(dim, n_points)
    
    with open(HOME_DIR + f"LHD/{dim}_{n_points}.pickle", "wb") as f:
        pickle.dump(LHD, f, pickle.HIGHEST_PROTOCOL)

