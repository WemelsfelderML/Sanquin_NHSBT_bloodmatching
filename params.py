import numpy as np

class Params():
    
    def __init__(self, SETTINGS):


        ####################
        # BLOOD PARAMETERS #
        ####################

        # Shelf life of the blood products in inventory.
        self.max_age = 35           # The age at which inventory products expire, so the maximum age to be issued = 34. 
        self.max_lead_time = 8      # Days 0,1,...,7.

        # Major blood groups, major antigens, and minor antigens.
        self.ABOD =  ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        self.antigens = {0:"A", 1:"B", 2:"D", 3:"C", 4:"c", 5:"E", 6:"e", 7:"K", 8:"k", 9:"Fya", 10:"Fyb", 11:"Jka", 12:"Jkb", 13:"M", 14:"N", 15:"S", 16:"s"}
        self.ethnicities = {0 : "Caucasian", 1 : "African", 2 : "Asian"}

        ##################
        # PATIENT GROUPS #
        ##################

        # Names of all considered patient groups
        self.patgroups = {0 : "Allo", 1 : "SCD", 2 : "Thal", 3 : "MDS", 4 : "AIHA", 5 : "Wu45", 6 : "Other"}

        # Distribution of patient groups in all hospital types.
        self.patgroup_distr = {
            #                   Allo        SCD             Thal            MDS         AIHA        Wu45        Other
            "Other" : [         0,          0,              0,              0,          0,          0,          1       ],
            "regional" : [      0.031632,   0.008607969,    0.008607969,    0.0,        0.014933,   0.049352,   0.886867],
            "university" : [    0.06543,    0.05665,        0.05665,        0.04542,    0.02731,    0.10250,    0.64605 ],
            "manual" : [        0.06543,    0.64605,        0.05665,        0.04542,    0.02731,    0.10250,    0.05665 ]
        }

        ###########################
        # LATIN HYPERCUBE DESIGNS #
        ###########################

        # TODO: normalizing the objective function takes a lot of computing time, so make sure
        # normalization is not needed by adjusting the tried weights.
        # ranges = np.array([
        #     [100, 900],     # shortages
        #     [20, 200],      # mismatches
        #     [0, 40],        # youngblood
        #     [0, 40],        # FIFO
        #     [0, 40],        # usability
        #     [0, 40],        # substitution
        #     [20, 200]])     # today
        ranges = np.array([
            [1000, 1000],   # shortages
            [5, 5],         # mismatches
            [1, 1],         # youngblood
            [1, 1],         # FIFO
            [1, 1],         # usability
            [1, 1],         # substitution
            [1, 1]])        # today

        # LHD configurations
        LHD = np.array(
            [[0, 34, 60, 30, 89, 83, 55],
            [1, 66, 64, 50, 49, 76, 0],
            [2, 38, 73, 16, 59, 21, 20],
            [3, 10, 81, 36, 33, 54, 75],
            [4, 52, 38, 12, 4, 57, 44],
            [5, 83, 91, 22, 43, 65, 61],
            [6, 19, 25, 71, 76, 24, 27],
            [7, 42, 17, 19, 58, 39, 88],
            [8, 56, 33, 69, 45, 90, 92],
            [9, 92, 49, 43, 42, 7, 50],
            [10, 82, 24, 87, 77, 67, 38],
            [11, 35, 88, 90, 41, 84, 46],
            [12, 71, 92, 74, 90, 43, 39],
            [13, 17, 19, 75, 52, 94, 31],
            [14, 68, 37, 84, 2, 72, 37],
            [15, 58, 40, 79, 84, 14, 85],
            [16, 12, 32, 92, 31, 41, 80],
            [17, 26, 85, 88, 46, 8, 43],
            [18, 72, 1, 45, 39, 40, 10],
            [19, 73, 65, 8, 98, 29, 69],
            [20, 2, 57, 64, 13, 52, 16],
            [21, 11, 20, 11, 56, 61, 17],
            [22, 84, 77, 91, 36, 47, 84],
            [23, 51, 51, 48, 0, 22, 94],
            [24, 63, 93, 55, 3, 34, 29],
            [25, 4, 36, 26, 27, 5, 54],
            [26, 76, 18, 10, 70, 87, 40],
            [27, 8, 78, 51, 81, 66, 3],
            [28, 88, 45, 35, 95, 36, 6],
            [29, 45, 27, 80, 11, 3, 28],
            [30, 54, 97, 42, 55, 10, 89],
            [31, 79, 4, 82, 28, 32, 79],
            [32, 97, 31, 24, 24, 68, 87],
            [33, 28, 83, 76, 83, 62, 95],
            [34, 15, 30, 18, 32, 96, 74],
            [35, 23, 12, 62, 96, 69, 76],
            [36, 44, 14, 13, 82, 4, 34],
            [37, 31, 86, 5, 29, 78, 23],
            [38, 96, 61, 54, 87, 80, 81],
            [39, 3, 59, 28, 91, 26, 66],
            [40, 90, 62, 89, 44, 28, 7],
            [41, 95, 6, 40, 85, 33, 67],
            [42, 55, 82, 46, 5, 88, 83],
            [43, 99, 53, 9, 30, 55, 14],
            [44, 21, 0, 53, 6, 56, 45],
            [45, 53, 28, 34, 15, 99, 8],
            [46, 47, 75, 7, 68, 77, 97],
            [47, 48, 63, 86, 97, 95, 36],
            [48, 98, 79, 66, 35, 91, 35],
            [49, 50, 50, 49, 50, 50, 49],
            [50, 43, 11, 57, 92, 70, 2],
            [51, 69, 5, 14, 18, 15, 59],
            [52, 81, 99, 29, 63, 13, 22],
            [53, 74, 76, 2, 14, 17, 62],
            [54, 39, 43, 20, 9, 25, 1],
            [55, 70, 89, 23, 86, 82, 25],
            [56, 22, 2, 61, 60, 9, 82],
            [57, 5, 96, 33, 37, 19, 30],
            [58, 0, 55, 98, 67, 53, 41],
            [59, 37, 70, 67, 94, 12, 13],
            [60, 85, 7, 68, 40, 93, 53],
            [61, 61, 10, 96, 73, 20, 32],
            [62, 16, 42, 85, 21, 97, 72],
            [63, 1, 94, 47, 57, 86, 57],
            [64, 18, 87, 78, 12, 38, 78],
            [65, 13, 74, 6, 16, 46, 86],
            [66, 46, 29, 97, 34, 75, 5],
            [67, 36, 90, 73, 8, 85, 19],
            [68, 57, 41, 15, 66, 11, 99],
            [69, 78, 66, 77, 20, 1, 63],
            [70, 91, 68, 65, 93, 16, 68],
            [71, 62, 34, 99, 74, 59, 90],
            [72, 20, 39, 17, 88, 92, 42],
            [73, 6, 16, 56, 53, 23, 12],
            [74, 14, 9, 3, 51, 44, 64],
            [75, 86, 23, 63, 1, 42, 24],
            [76, 32, 58, 0, 72, 35, 11],
            [77, 60, 98, 94, 61, 51, 47],
            [78, 65, 13, 25, 75, 79, 91],
            [79, 25, 67, 81, 71, 2, 77],
            [80, 80, 35, 38, 54, 0, 15],
            [81, 67, 54, 1, 26, 89, 58],
            [82, 89, 80, 41, 38, 49, 96],
            [83, 49, 21, 39, 7, 60, 93],
            [84, 33, 71, 83, 23, 18, 9],
            [85, 94, 52, 4, 78, 45, 52],
            [86, 77, 56, 93, 10, 71, 71],
            [87, 93, 44, 70, 80, 64, 21],
            [88, 9, 47, 52, 62, 58, 98],
            [89, 27, 22, 95, 19, 31, 60],
            [90, 7, 46, 32, 17, 74, 33],
            [91, 40, 95, 27, 79, 37, 65],
            [92, 64, 8, 21, 48, 63, 18],
            [93, 24, 3, 72, 64, 81, 48],
            [94, 41, 26, 44, 99, 30, 51],
            [95, 29, 72, 59, 69, 73, 4],
            [96, 59, 69, 58, 65, 98, 70],
            [97, 75, 84, 37, 22, 48, 26],
            [98, 30, 48, 31, 25, 6, 56],
            [99, 87, 15, 60, 47, 27, 73]])
        
        self.LHD = ranges[:, 0].reshape(1, -1) + LHD * ((ranges[:,1] - ranges[:,0]) / LHD.shape[0]).reshape(1, -1)
    
        ###########
        # WEIGHTS #
        ###########

        # No mismatch weights - major antigens have a weight of 10 to denote that they can not be mismatched.
        self.major_weights = np.array(([10] * 3) + ([0] * 14))  

         # Normalized probability of alloimmunization occurring as a result of mismatching on 3 units. A weight of 10 means that mismatching is never allowed.
        self.relimm_weights = np.array([
        #    A   B   D   C       c       E       e       K       k  Fya     Fyb     Jka     Jkb     M       N  S       s
            [10, 10, 10, 0.0329, 0.0733, 0.2375, 0.0592, 0.3907, 0, 0.0446, 0.0165, 0.0861, 0.0129, 0.0276, 0, 0.0129, 0.0059]
        ])

        # Normalized patient group specific weights. A weight of 10 means that mismatching is never allowed.   
        self.patgroup_weights = np.array([
        #    A    B    D    C           c           E           e           K           k   Fya         Fyb         Jka         Jkb         M  N  S           s
            [10,  10,  10,  10,         10,         10,         10,         10,         0,  0.26312684, 0.09734513, 0.5079646,  0.07610619, 0, 0, 0.0380531,  0.01740413],  # Allo
            [10,  10,  10,  10,         10,         10,         10,         10,         0,  10,         0.16224189, 10,         10,         0, 0, 0.06342183, 0.02900688],  # SCD
            [10,  10,  10,  10,         10,         10,         10,         10,         0,  0.1754179,  0.06489676, 0.33864307, 0.05073746, 0, 0, 0.02536873, 0.01160275],  # Thal
            [10,  10,  10,  10,         10,         10,         10,         10,         0,  0.1754179,  0.06489676, 0.33864307, 0.05073746, 0, 0, 0.02536873, 0.01160275],  # MDS
            [10,  10,  10,  10,         10,         10,         10,         10,         0,  0.26312684, 0.09734513, 0.5079646,  0.07610619, 0, 0, 0.0380531,  0.01740413],  # AIHA
            [10,  10,  10,  0.00485251, 10,         10,         0.00873156, 10,         0,  0.00438545, 0.00162242, 0.00846608, 0.00126844, 0, 0, 0.00063422, 0.00029007],  # Wu45
            [10,  10,  10,  0.00485251, 0.01081121, 0.0350295 , 0.00873156, 0.05762537, 0,  0.00438545, 0.00162242, 0.00846608, 0.00126844, 0, 0, 0.00063422, 0.00029007]   # Other
        ])

        # Proability of developping alloantibodies against a certain antigen after a certain number of antigen-positive units has been transfused to an antigen-negative patient.
        self.alloimmunization_risks = np.array([
        #    A,     B,     D,     C,     c,     E,     e,     K,     k,     Fya,   Fyb,   Jka,   Jkb,   M,     N,     S,     s
            [1.000, 1.000, 1.000, 0.001, 0.001, 0.007, 0.000, 0.009, 0.000, 0.001, 0.000, 0.003, 0.000, 0.001, 0.000, 0.000, 0.000],  # 1 units
            [1.000, 1.000, 1.000, 0.002, 0.004, 0.015, 0.005, 0.023, 0.000, 0.003, 0.001, 0.005, 0.000, 0.002, 0.000, 0.000, 0.000],  # 2 units
            [1.000, 1.000, 1.000, 0.003, 0.007, 0.019, 0.005, 0.035, 0.000, 0.004, 0.002, 0.007, 0.002, 0.002, 0.000, 0.001, 0.000],  # 3 units
            [1.000, 1.000, 1.000, 0.004, 0.008, 0.024, 0.005, 0.039, 0.000, 0.005, 0.002, 0.009, 0.002, 0.003, 0.000, 0.001, 0.000],  # 4 units
            [1.000, 1.000, 1.000, 0.005, 0.009, 0.028, 0.005, 0.041, 0.000, 0.005, 0.002, 0.014, 0.002, 0.003, 0.000, 0.001, 0.000],  # 5 units
            [1.000, 1.000, 1.000, 0.005, 0.010, 0.033, 0.005, 0.047, 0.000, 0.006, 0.003, 0.016, 0.003, 0.005, 0.000, 0.002, 0.000],  # 6 units
            [1.000, 1.000, 1.000, 0.006, 0.011, 0.035, 0.005, 0.047, 0.000, 0.006, 0.003, 0.018, 0.003, 0.005, 0.000, 0.002, 0.000],  # 7 units
            [1.000, 1.000, 1.000, 0.007, 0.012, 0.040, 0.005, 0.047, 0.000, 0.007, 0.003, 0.019, 0.003, 0.006, 0.001, 0.002, 0.000],  # 8 units
            [1.000, 1.000, 1.000, 0.007, 0.013, 0.043, 0.005, 0.047, 0.000, 0.007, 0.003, 0.021, 0.003, 0.008, 0.001, 0.002, 0.000],  # 9 units
            [1.000, 1.000, 1.000, 0.007, 0.016, 0.043, 0.005, 0.047, 0.000, 0.007, 0.002, 0.021, 0.003, 0.008, 0.001, 0.002, 0.000],  # 10 units
            [1.000, 1.000, 1.000, 0.007, 0.016, 0.046, 0.005, 0.047, 0.000, 0.007, 0.003, 0.021, 0.003, 0.009, 0.001, 0.002, 0.000],  # 11 units
            [1.000, 1.000, 1.000, 0.008, 0.021, 0.046, 0.005, 0.047, 0.000, 0.007, 0.003, 0.023, 0.003, 0.009, 0.001, 0.002, 0.000],  # 12 units
            [1.000, 1.000, 1.000, 0.008, 0.021, 0.049, 0.005, 0.047, 0.000, 0.007, 0.003, 0.023, 0.003, 0.009, 0.001, 0.002, 0.000],  # 13 units
            [1.000, 1.000, 1.000, 0.008, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.009, 0.001, 0.002, 0.000],  # 14 units
            [1.000, 1.000, 1.000, 0.008, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.009, 0.001, 0.002, 0.000],  # 15 units
            [1.000, 1.000, 1.000, 0.012, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.013, 0.001, 0.002, 0.000],  # 16 units
            [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.026, 0.003, 0.013, 0.001, 0.002, 0.000],  # 17 units
            [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 18 units
            [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 19 units
            [1.000, 1.000, 1.000, 0.015, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 20 units
        ])



        #####################
        # SUPPLY AND DEMAND #
        #####################

        # Each column specifies the probability of a request becoming known 0, 1, 2, etc. days before its issuing date.
        self.request_lead_time_probabilities = np.array([
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0],   # Allo
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # SCD
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # Thal
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # MDS
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                 # AIHA
            [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],             # Wu45
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0]    # Other
        ])

        # Each column specifies the probability of a request becoming known 0, 1, 2, etc. days before its issuing date.
        self.request_lead_time_probabilities = np.array([
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0],   # Allo
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # SCD
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # Thal
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                 # MDS
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                 # AIHA
            [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],             # Wu45
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0]    # Other
        ])

        # Each row specifies the probability of the corresponding patient type having a demand for [1,2,3,4] units respectively
        self.request_num_units_probabilities = np.array([
            [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782],    # Allo
            [0, 1, 0, 0],                                                                           # SCD
            [0, 1, 0, 0],                                                                           # Thal
            [0, 1, 0, 0],                                                                           # MDS
            [0, 1, 0, 0],                                                                           # AIHA
            [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782],    # Wu45
            [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782]     # Other
        ])
        
        # Distribution of major blood groups in the donor population.
        self.donor_ABOD_distr = {"O-":0.1551, "O+":0.3731, "A-":0.0700, "A+":0.3074, "B-":0.0158, "B+":0.0604, "AB-":0.0047, "AB+":0.0134}

        # All possible antigen profiles for antigens A, B,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.ABO_phenotypes = np.array([
            [ 0, 0 ],
            [ 0, 1 ],
            [ 1, 0 ],
            [ 1, 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.ABO_prevalences = np.array([
            [0.43, 0.09, 0.44, 0.04],   # Caucasian
            [0.27, 0.49, 0.2 , 0.04],   # African
            [0.27, 0.43, 0.25, 0.05]])  # Asian

        # All possible antigen profiles for antigens D, C, c, E, e,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Rhesus_phenotypes = np.array([
            [0,1,1,1,1],
            [0,1,0,1,0],
            [0,1,0,0,1],
            [0,0,1,1,0],
            [0,0,1,0,1],
            [0,1,1,1,0],
            [0,1,1,0,1],
            [0,1,0,1,1],
            [0,0,1,1,1],
            [1,1,1,1,1],
            [1,1,0,1,0],
            [1,1,0,0,1],
            [1,0,1,1,0],
            [1,0,1,0,1],
            [1,1,1,1,0],
            [1,1,1,0,1],
            [1,1,0,1,1],
            [1,0,1,1,1]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Rhesus_prevalences = np.array([
            [0.   , 0.   , 0.   , 0.   , 0.151, 0.   , 0.008, 0.   , 0.009, 0.133, 0.   , 0.185, 0.023, 0.021, 0.001, 0.349, 0.002, 0.118],     # Caucasian
            [0.   , 0.   , 0.   , 0.   , 0.068, 0.   , 0.   , 0.   , 0.   , 0.056, 0.   , 0.02 , 0.002, 0.458, 0.   , 0.21 , 0.   , 0.186],     # African
            [0.   , 0.   , 0.001, 0.001, 0.001, 0.   , 0.001, 0.   , 0.   , 0.303, 0.   , 0.518, 0.044, 0.003, 0.004, 0.085, 0.014, 0.025]])    # Asian

        # All possible antigen profiles for antigens K, k,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Kell_phenotypes = np.array([
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Kell_prevalences = np.array([
            [0.   , 0.002, 0.91 , 0.088],   # Caucasian
            [0.   , 0.   , 0.98 , 0.02 ],   # African
            [0.   , 0.   , 1.   , 0.   ]])  # Asian

        # All possible antigen profiles for antigens M, N, S, s,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.MNS_phenotypes = np.array([
            [1,0,1,0],
            [1,0,1,1],
            [1,0,0,1],
            [1,1,1,0],
            [1,1,1,1],
            [1,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [0,1,0,1],
            [1,0,0,0],
            [1,1,0,0],
            [0,1,0,0]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.MNS_prevalences  = np.array([
            [0.06 , 0.14 , 0.08 , 0.04 , 0.24 , 0.22 , 0.01 , 0.06 , 0.15 , 0.   , 0.   , 0.   ],   # Caucasian
            [0.02 , 0.07 , 0.16 , 0.02 , 0.13 , 0.325, 0.02 , 0.05 , 0.19 , 0.004, 0.004, 0.007],   # African
            [0.06 , 0.14 , 0.08 , 0.04 , 0.24 , 0.22 , 0.01 , 0.06 , 0.15 , 0.   , 0.   , 0.   ]])  # Asian

        # All possible antigen profiles for antigens Fya, Fyb, 
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Duffy_phenotypes = np.array([
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Duffy_prevalences = np.array([
            [0.   , 0.17 , 0.34 , 0.49 ],   # Caucasian
            [0.68 , 0.09 , 0.22 , 0.01 ],   # African
            [0.   , 0.908, 0.003, 0.089]])  # Asian

        # All possible antigen profiles for antigens Jka, Jkb,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Kidd_phenotypes = np.array([
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Kidd_prevalences = np.array([
            [0.   , 0.263, 0.234, 0.503],   # Caucasian
            [0.   , 0.511, 0.081, 0.488],   # African
            [0.009, 0.232, 0.268, 0.491]])  # Asian

        
        ##########
        # GUROBI #
        ##########
        
        self.status_code = {
            1 : "MODEL IS LOADED, BUT NO SOLUTION IS AVAILABLE",
            2 : "MODEL IS SOLVED OPTIMALLY",
            3 : "MODEL IS INFEASIBLE",
            4 : "MODEL IS EITHER INFEASIBLE OR UNBOUNDED\nTo obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.",
            5 : "MODEL IS UNBOUNDED\nAn unbounded ray allows the objective to improve without limit.",
            6 : "NO SOLUTION AVAILABLE\nThe optimal objective was worse than the Cutoff parameter.",
            7 : "OPTIMIZATION TERMINATED\nIterationLimit or BarIterLimit parameter was exceeded.",
            8 : "OPTIMIZATION TERMINATED\nNodeLimit parameter was exceeded.",
            9 : "OPTIMIZATION TERMINATED\nTimeLimit parameter was exceeded.",
            10 : "OPTIMIZATION TERMINATED\nSolutionLimit parameter was exceeded.",
            11 : "OPTIMIZATION TERMINATED BY USER\nObtained results are not saved.",
            12 : "OPTIMIZATION TERMINATED\nUnrecoverable numerical difficulties.",
            13 : "UNABLE TO SATISFY OPTIMALITY\nA sub-optimal solution is available.",
            14 : "ASYNCHRONOUS CALL WAS MADE, ASSOCIATED OPTIMIZATION NOT YET COMPLETE",
            15 : "LIMIT SET BY USER WAS EXCEEDED\nThis is either a bound on the best objective or the best bound."
        }
