# import gurobipy as grb
import numpy as np
import time
import pickle
# from scipy.spatial import distance


def generate_initial_lhd(n, d):
    """Generate an initial Latin Hypercube Design."""
    LHD = np.zeros((n, d))
    for j in range(d):
        LHD[:, j] = np.random.permutation(n)
    return LHD

def pdist(points):
    """Compute pairwise distances."""
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances

def swap_rows_columns(LHD, i1, i2, j1, j2):
    """Swap elements to produce a new LHD."""
    LHD_new = LHD.copy()
    LHD_new[i1, j1], LHD_new[i1, j2] = LHD[i2, j1], LHD[i2, j2]
    LHD_new[i2, j1], LHD_new[i2, j2] = LHD[i1, j1], LHD[i1, j2]
    return LHD_new

def improve_lhd(LHD, max_iterations=1000):
    """Optimize the LHD using the swap heuristic."""
    n, d = LHD.shape
    for iteration in range(max_iterations):
        min_distance = pdist(LHD).min()
        print(iteration, min_distance)
        any_improvement = False
        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                for j1 in range(d - 1):
                    for j2 in range(j1 + 1, d):
                        LHD_new = swap_rows_columns(LHD, i1, i2, j1, j2)
                        new_min_distance = pdist(LHD_new).min()
                        if new_min_distance > min_distance:
                            min_distance = new_min_distance
                            LHD = LHD_new
                            any_improvement = True
        if not any_improvement:
            break
    return LHD



if __name__ == "__main__":

    # HOME_DIR = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Sanquin_NHSBT_bloodmatching/"
    HOME_DIR = "/home/mw922/Sanquin_NHSBT_bloodmatching/"
    n = 500  # Number of points
    d = 6    # Number of dimensions

    # LHD = create_LHD(dim, n_points)
    LHD = generate_initial_lhd(n, d)
    LHD_optimized = improve_lhd(LHD)
    # LHD_optimized_normalized = (LHD_optimized + 0.5) / n
    
    with open(HOME_DIR + f"LHD/{d}_{n}.pickle", "wb") as f:
        pickle.dump(LHD_optimized, f, pickle.HIGHEST_PROTOCOL)


# # Single-hospital setup: MINRAR model for matching within a single hospital.
# def create_LHD(dim, n_points):

#     start = time.perf_counter()

#     model = grb.Model(name="model")
#     model.Params.LogToConsole = 1
#     model.Params.NonConvex = 2
#     model.setParam('Threads', 24)
#     model.setParam('TimeLimit', 12*60*60) # 12 hours


#     # Variables
#     x = model.addMVar((n_points, dim), name="x", vtype=grb.GRB.INTEGER, lb=0, ub=n_points-1)
#     y = model.addMVar((n_points, n_points, dim), name="y", vtype=grb.GRB.BINARY)  # Binary variables
#     dist = model.addVar(name="dist", vtype=grb.GRB.CONTINUOUS, lb=0)

#     # Linking constraints between x and y
#     for i in range(n_points):
#         for d in range(dim):
#             model.addConstr(x[i,d] == grb.quicksum(j*y[i,j,d] for j in range(n_points)))

#     # Ensure each value can be assigned once
#     for j in range(n_points):
#         for d in range(dim):
#             model.addConstr(grb.quicksum(y[i,j,d] for i in range(n_points)) <= 1)


#     # Distance constraints
#     for i in range(n_points-1):
#         for j in range(i+1, n_points):
#             model.addConstr(((x[i,:] - x[j,:])*(x[i,:] - x[j,:])).sum() >= dist)

#     # Objective
#     model.setObjective(expr=dist, sense=grb.GRB.MAXIMIZE)
#     model.update()
#     stop = time.perf_counter()
#     print(f"model initialization: {(stop - start):0.4f} seconds")

#     start = time.perf_counter()
#     model.optimize() 
#     stop = time.perf_counter()
#     print(f"Optimize: {(stop - start):0.4f} seconds")

#     x_solved = x.X
#     return x_solved