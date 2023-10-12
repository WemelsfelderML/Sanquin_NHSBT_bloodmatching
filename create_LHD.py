import gurobipy as grb
import numpy as np
import time
from scipy.spatial import distance


# Single-hospital setup: MINRAR model for matching within a single hospital.
def create_LHD(dim, n_points):

    start = time.perf_counter()

    model = grb.Model(name="model")
    model.Params.LogToConsole = 1
    model.Params.NonConvex = 2
    model.setParam('Threads', 24)
    model.setParam('TimeLimit', 12*60*60) # 12 hours


    # Variables
    x = model.addMVar((n_points, dim), name="x", vtype=grb.GRB.INTEGER, lb=0, ub=n_points-1)
    y = model.addMVar((n_points, n_points, dim), name="y", vtype=grb.GRB.BINARY)  # Binary variables
    dist = model.addVar(name="dist", vtype=grb.GRB.CONTINUOUS, lb=0)

    # Linking constraints between x and y
    for i in range(n_points):
        for d in range(dim):
            model.addConstr(x[i,d] == grb.quicksum(j*y[i,j,d] for j in range(n_points)))

    # Ensure each value can be assigned once
    for j in range(n_points):
        for d in range(dim):
            model.addConstr(grb.quicksum(y[i,j,d] for i in range(n_points)) <= 1)


    # Distance constraints
    for i in range(n_points-1):
        for j in range(i+1, n_points):
            model.addConstr(((x[i,:] - x[j,:])*(x[i,:] - x[j,:])).sum() >= dist)

    # Objective
    model.setObjective(expr=dist, sense=grb.GRB.MAXIMIZE)
    model.update()
    stop = time.perf_counter()
    print(f"model initialization: {(stop - start):0.4f} seconds")

    start = time.perf_counter()
    model.optimize() 
    stop = time.perf_counter()
    print(f"Optimize: {(stop - start):0.4f} seconds")

    x_solved = x.X
    return x_solved


if __name__ == "__main__":

    # HOME_DIR = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Sanquin_NHSBT_bloodmatching/"
    HOME_DIR = "/home/mw922/Sanquin_NHSBT_bloodmatching/"
    dim = 6
    n_points = 500

    LHD = create_LHD(dim, n_points)
    
    with open(HOME_DIR + f"LHD/{dim}_{n_points}.pickle", "wb") as f:
        pickle.dump(LHD, f, pickle.HIGHEST_PROTOCOL)

