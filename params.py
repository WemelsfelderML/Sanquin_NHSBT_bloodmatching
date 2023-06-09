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

        # Names of all considered patient groups.
        self.patgroups = {0 : "Allo", 1 : "SCD", 2 : "Thal", 3 : "MDS", 4 : "AIHA", 5 : "Wu45", 6 : "Other"}

        # Weekly demand per hospital and patient group.
        self.weekly_demand = {
        #                       Allo    SCD     Thal    MDS     AIHA    Wu45    Other
            "UCLH" : np.array([ 0,      34,     0,      0,      0,      0,      572]),
            "NMUH": np.array([  0,      11,     0,      0,      0,      0,      197]),
            "WH" : np.array([   0,      10,     0,      0,      0,      0,      230]),
        }

        if sum(SETTINGS.n_hospitals.values()) > 1:
            self.supply_size = round((SETTINGS.init_days + SETTINGS.test_days) * SETTINGS.inv_size_factor_dc * sum([SETTINGS.n_hospitals[htype] * (sum(self.weekly_demand[htype])/7) for htype in SETTINGS.n_hospitals.keys()]))
        else:
            self.supply_size = round((SETTINGS.init_days + SETTINGS.test_days) * SETTINGS.inv_size_factor_hosp * sum([SETTINGS.n_hospitals[htype] * (sum(self.weekly_demand[htype])/7) for htype in SETTINGS.n_hospitals.keys()]))


        ###########################
        # LATIN HYPERCUBE DESIGNS #
        ###########################

        # Note: all objectives are normalized so no need to consider
        # that in determining these weights.
        ranges = np.array([
            [10, 1000], # shortages
            [10, 500],  # mismatches
            [0, 100],   # youngblood
            [0, 100],   # FIFO
            [0, 100],   # usability
            [0, 100],   # substitution
            [10, 500]]) # today
        # ranges = np.array([
        #     [1000, 1000],   # shortages
        #     [5, 5],         # mismatches
        #     [1, 1],         # youngblood
        #     [1, 1],         # FIFO
        #     [1, 1],         # usability
        #     [1, 1],         # substitution
        #     [1, 1]])        # today

        # LHD configurations
        LHD = np.array([
            [0, 10, 15, 1, 16, 11, 8],
            [1, 0, 12, 9, 6, 1, 10],
            [2, 12, 13, 17, 15, 7, 18],
            [3, 13, 8, 14, 2, 14, 3],
            [4, 4, 9, 6, 5, 17, 19],
            [5, 9, 5, 13, 19, 3, 2],
            [6, 16, 2, 4, 7, 4, 13],
            [7, 14, 3, 10, 18, 19, 12],
            [8, 1, 1, 2, 11, 13, 4],
            [9, 15, 19, 11, 9, 0, 5],
            [10, 2, 0, 18, 10, 9, 14],
            [11, 19, 18, 7, 8, 15, 16],
            [12, 3, 16, 15, 13, 16, 6],
            [13, 6, 17, 0, 1, 10, 7],
            [14, 8, 14, 16, 0, 6, 17],
            [15, 17, 10, 3, 14, 12, 0],
            [16, 7, 11, 5, 17, 5, 15],
            [17, 5, 6, 12, 4, 2, 1],
            [18, 18, 7, 19, 12, 8, 9],
            [19, 11, 4, 8, 3, 18, 11],
        ])
        
        self.LHD = ranges[:, 0].reshape(1, -1) + LHD * ((ranges[:,1] - ranges[:,0]) / LHD.shape[0]).reshape(1, -1)
    
        ###########
        # WEIGHTS #
        ###########

        # No mismatch weights - major antigens have a weight of 10 to denote that they can not be mismatched.
        self.major_weights = np.array(([10] * 3) + ([0] * 14))  

        # Normalized alloimmunization incidence after mismatching on 3 units. A weight of 10 means that mismatching is never allowed.
        # Based on thesis Dorothea Evers
        self.relimm_weights = np.array([
        #    A   B   D   C       c        E        e        K        k  Fya      Fyb      Jka      Jkb      M        N        S        s
            [10, 10, 10, 0.0344, 0.08028, 0.21789, 0.05734, 0.40138, 0, 0.04587, 0.02294, 0.08028, 0.02294, 0.02294, 0.00115, 0.01147, 0.00115]
        ])

        # # Based on thesis Tom vd Woude (relative probability of forming antibodies after 3 mismatched units)
        # self.relimm_weights = np.array([
        # #    A   B   D   C       c       E       e       K       k  Fya     Fyb     Jka    Jkb     M       N       S       s
        #     [10, 10, 10, 0.0189, 0.0298, 0.3093, 0.0027, 0.5918, 0, 0.0169, 0.0015, 0.019, 0.0018, 0.0048, 0.0002, 0.0032, 0.0002]
        # ])


        # # Based on thesis Tom vd Woude (average antibody formation probabilities, Table 3 of thesis)
        # self.relimm_weights = np.array([
        # #    A   B   D   C        c        E        e        K        k  Fya      Fyb     Jka      Jkb      M        N        S        s
        #     [10, 10, 10, 0.00374, 0.00596, 0.22552, 0.00009, 0.75037, 0, 0.00414, 0.0006, 0.00564, 0.00056, 0.00128, 0.00001, 0.00209, 0.00001]
        # ])

        # Normalized patient group specific weights. A weight of 10 means that mismatching is never allowed.   
        # Based on thesis Dorothea Evers
        self.patgroup_weights = np.array([
        #    A   B   D   C           c           E           e           K           k  Fya         Fyb         Jka         Jkb         M  N  S           s
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # Allo
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 10,         0.21438451, 10,         10,         0, 0, 0.05359613, 0.00537363],  # SCD
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845, 0.00214945],  # Thal
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845, 0.00214945],  # MDS
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # AIHA
            [10, 10, 10, 0.00482225, 10,         10,         0.00803802, 10,         0, 0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596, 0.00005374],  # Wu45
            [10, 10, 10, 0.00482225, 0.01125378, 0.03054419, 0.00803802, 0.05626612, 0, 0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596, 0.00005374]   # Other
        ])

        # # Based on thesis Tom vd Woude (relative probability of forming antibodies after 3 mismatched units)
        # self.patgroup_weights = np.array([
        # #    A   B   D   C           c           E           e           K           k  Fya         Fyb         Jka         Jkb         M  N  S           s
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.41337907, 0.0367557,  0.46459201, 0.04435187, 0, 0, 0.03908356, 0.00183778],  # Allo
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 10,         0.0612595,  10,         10,         0, 0, 0.06513926, 0.00306297],  # SCD
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.27558605, 0.0245038,  0.30972801, 0.02956792, 0, 0, 0.02605571, 0.00122519],  # Thal
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.27558605, 0.0245038,  0.30972801, 0.02956792, 0, 0, 0.02605571, 0.00122519],  # MDS
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.41337907, 0.0367557,  0.46459201, 0.04435187, 0, 0, 0.03908356, 0.00183778],  # AIHA
        #     [10, 10, 10, 0.01160867, 10,         10,         0.00167238, 10,         0, 0.00688965, 0.00061259, 0.0077432,  0.0007392,  0, 0, 0.00065139, 0.00003063],  # Wu45
        #     [10, 10, 10, 0.01160867, 0.01825533, 0.18948787, 0.00167238, 0.36251531, 0, 0.00688965, 0.00061259, 0.0077432,  0.0007392,  0, 0, 0.00065139, 0.00003063]   # Other
        # ])

        # # Based on thesis Tom vd Woude (average antibody formation probabilities, Table 3 of thesis)
        # self.patgroup_weights = np.array([
        # #    A   B   D   C           c           E           e           K           k   Fya         Fyb         Jka         Jkb         M  N  S           s
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.16665437, 0.02415281, 0.22703638, 0.02254262, 0, 0, 0.04206614, 0.00020127],  # Allo
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 10,         0.04025468, 10,         10,         0, 0, 0.07011023, 0.00033546],  # SCD
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.11110291, 0.01610187, 0.15135759, 0.01502841, 0, 0, 0.02804409, 0.00013418],  # Thal
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.11110291, 0.01610187, 0.15135759, 0.01502841, 0, 0, 0.02804409, 0.00013418],  # MDS
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.16665437, 0.02415281, 0.22703638, 0.02254262, 0, 0, 0.04206614, 0.00020127],  # AIHA
        #     [10, 10, 10, 0.00376381, 10,         10,         0.00009057, 10,         0, 0.00277757, 0.00040255, 0.00378394, 0.00037571, 0, 0, 0.0007011,  0.00000335],  # Wu45
        #     [10, 10, 10, 0.00376381, 0.00599795, 0.22695587, 0.00009057, 0.75514757, 0, 0.00277757, 0.00040255, 0.00378394, 0.00037571, 0, 0, 0.0007011,  0.00000335]   # Other
        # ])

        # # Cumulative alloimmunization incidence after a certain number of antigen-positive units has been transfused to an antigen-negative patient.
        # # Based on thesis Dorothea Evers
        # self.alloimmunization_risks = np.array([
        # #    A,     B,     D,     C,     c,     E,     e,     K,     k,     Fya,   Fyb,   Jka,   Jkb,   M,     N,     S,     s
        #     [0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],      # 0 units
        #     [1.000, 1.000, 1.000, 0.001, 0.001, 0.007, 0.000, 0.009, 0.000, 0.001, 0.000, 0.003, 0.000, 0.001, 0.000, 0.000, 0.000],  # 1 unit
        #     [1.000, 1.000, 1.000, 0.002, 0.004, 0.015, 0.005, 0.023, 0.000, 0.003, 0.001, 0.005, 0.000, 0.002, 0.000, 0.000, 0.000],  # 2 units
        #     [1.000, 1.000, 1.000, 0.003, 0.007, 0.019, 0.005, 0.035, 0.000, 0.004, 0.002, 0.007, 0.002, 0.002, 0.000, 0.001, 0.000],  # 3 units
        #     [1.000, 1.000, 1.000, 0.004, 0.008, 0.024, 0.005, 0.039, 0.000, 0.005, 0.002, 0.009, 0.002, 0.003, 0.000, 0.001, 0.000],  # 4 units
        #     [1.000, 1.000, 1.000, 0.005, 0.009, 0.028, 0.005, 0.041, 0.000, 0.005, 0.002, 0.014, 0.002, 0.003, 0.000, 0.001, 0.000],  # 5 units
        #     [1.000, 1.000, 1.000, 0.005, 0.010, 0.033, 0.005, 0.047, 0.000, 0.006, 0.003, 0.016, 0.003, 0.005, 0.000, 0.002, 0.000],  # 6 units
        #     [1.000, 1.000, 1.000, 0.006, 0.011, 0.035, 0.005, 0.047, 0.000, 0.006, 0.003, 0.018, 0.003, 0.005, 0.000, 0.002, 0.000],  # 7 units
        #     [1.000, 1.000, 1.000, 0.007, 0.012, 0.040, 0.005, 0.047, 0.000, 0.007, 0.003, 0.019, 0.003, 0.006, 0.001, 0.002, 0.000],  # 8 units
        #     [1.000, 1.000, 1.000, 0.007, 0.013, 0.043, 0.005, 0.047, 0.000, 0.007, 0.003, 0.021, 0.003, 0.008, 0.001, 0.002, 0.000],  # 9 units
        #     [1.000, 1.000, 1.000, 0.007, 0.016, 0.043, 0.005, 0.047, 0.000, 0.007, 0.002, 0.021, 0.003, 0.008, 0.001, 0.002, 0.000],  # 10 units
        #     [1.000, 1.000, 1.000, 0.007, 0.016, 0.046, 0.005, 0.047, 0.000, 0.007, 0.003, 0.021, 0.003, 0.009, 0.001, 0.002, 0.000],  # 11 units
        #     [1.000, 1.000, 1.000, 0.008, 0.021, 0.046, 0.005, 0.047, 0.000, 0.007, 0.003, 0.023, 0.003, 0.009, 0.001, 0.002, 0.000],  # 12 units
        #     [1.000, 1.000, 1.000, 0.008, 0.021, 0.049, 0.005, 0.047, 0.000, 0.007, 0.003, 0.023, 0.003, 0.009, 0.001, 0.002, 0.000],  # 13 units
        #     [1.000, 1.000, 1.000, 0.008, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.009, 0.001, 0.002, 0.000],  # 14 units
        #     [1.000, 1.000, 1.000, 0.008, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.009, 0.001, 0.002, 0.000],  # 15 units
        #     [1.000, 1.000, 1.000, 0.012, 0.021, 0.058, 0.005, 0.047, 0.000, 0.007, 0.005, 0.026, 0.003, 0.013, 0.001, 0.002, 0.000],  # 16 units
        #     [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.026, 0.003, 0.013, 0.001, 0.002, 0.000],  # 17 units
        #     [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 18 units
        #     [1.000, 1.000, 1.000, 0.012, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 19 units
        #     [1.000, 1.000, 1.000, 0.015, 0.025, 0.058, 0.005, 0.047, 0.000, 0.009, 0.005, 0.029, 0.003, 0.013, 0.001, 0.002, 0.000],  # 20 units
        # ])

        # Based on thesis Tom vd Woude
        self.alloimmunization_risks = np.array([
        #    A, B, D, C,          c,          E,          e,          K,          k, Fya,        Fyb,        Jka,        Jkb,        M,          N  S           s
            [0, 0, 0, 0,          0,          0,          0,          0,          0, 0,          0,          0,          0,          0,          0, 0,          0], # 0 units
            [1, 1, 1, 0.00049106, 0.00077621, 0.00832606, 0.00006923, 0.01617738, 0, 0.00043673, 0.000038,   0.00049141, 0.0000459 , 0.00012247, 0, 0.00008117, 0], # 1 unit
            [1, 1, 1, 0.00075047, 0.00116883, 0.01150707, 0.00011222, 0.02174881, 0, 0.00066994, 0.00006263, 0.00075099, 0.00007526, 0.00019527, 0, 0.00013097, 0], # 2 units
            [1, 1, 1, 0.00037366, 0.00059618, 0.00674999, 0.00005077, 0.01334575, 0, 0.00033155, 0.00002757, 0.00037394, 0.00003341, 0.00009074, 0, 0.00005969, 0], # 3 units
            [1, 1, 1, 0.00029237, 0.00047033, 0.00558556, 0.00003844, 0.01121518, 0, 0.00025888, 0.00002068, 0.00029258, 0.00002514, 0.00006934, 0, 0.00004531, 0], # 4 units
            [1, 1, 1, 0.00035473, 0.00056696, 0.00648485, 0.00004786, 0.01286377, 0, 0.00031461, 0.00002594, 0.00035498, 0.00003146, 0.00008571, 0, 0.0000563,  0], # 5 units
            [1, 1, 1, 0.00018949, 0.00030922, 0.00398899, 0.00002355, 0.00822669, 0, 0.00016719, 0.00001247, 0.00018964, 0.00001523, 0.00004315, 0, 0.00002788, 0], # 6 units
            [1, 1, 1, 0.00016559, 0.00027139, 0.00359084, 0.00002022, 0.0074665,  0, 0.00014594, 0.00001066, 0.00016572, 0.00001304, 0.00003724, 0, 0.00002398, 0], # 7 units
            [1, 1, 1, 0.00027781, 0.00044767, 0.00536918, 0.00003628, 0.01081509, 0, 0.00024588, 0.00001949, 0.00027802, 0.0000237 , 0.00006557, 0, 0.00004279, 0], # 8 units
            [1, 1, 1, 0.00005649, 0.00009574, 0.00154019, 0.00000604, 0.00341214, 0, 0.00004936, 0.00000306, 0.00005653, 0.00000379, 0.00001154, 0, 0.00000723, 0], # 9 units
            [1, 1, 1, 0.0000587,  0.0000936,  0.0035428,  0.0000014,  0.0117878,  0, 0.0000651,  0.0000094,  0.0000886,  0.0000088,  0.0000201,  0, 0.0000328 , 0]  # average probability  
        ])


        #####################
        # SUPPLY AND DEMAND #
        #####################

        # Each column specifies the probability of a request becoming known 0, 1, 2, etc. days before its issuing date.
        self.request_lead_time_probabilities = np.array([
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0],   # Allo
            [0, 0, 0, 0, 0, 0, 0, 1],                 # SCD
            [0, 0, 0, 0, 0, 0, 0, 1],                 # Thal
            [0, 0, 0, 0, 0, 0, 0, 1],                 # MDS
            [1, 0, 0, 0, 0, 0, 0, 0],                 # AIHA
            [1/2, 1/2, 0, 0, 0, 0, 0, 0],             # Wu45
            [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0]    # Other
        ])

        # Each row specifies the probability of the corresponding patient type having a demand for a certain number of units.
        self.request_num_units_probabilities = np.array([
        #    0, 1,        2,        3,        4,        5,        6,        7,       8,         9,        10,       11,       12
            [0, 0.404374, 0.496839, 0.068289, 0.030499, 0,        0,        0,        0,        0,        0,        0,        0],           # Allo
            [0, 0,        0,        0,        0.000001, 0.000134, 0.004432, 0.053998, 0.242003, 0.398996, 0.242003, 0.053998, 0.004432],    # SCD
            [0, 0,        1,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0],           # Thal
            [0, 0,        1,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0],           # MDS
            [0, 0,        1,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0],           # AIHA
            [0, 0.404374, 0.496839, 0.068289, 0.030499, 0,        0,        0,        0,        0,        0,        0,        0],           # Wu45
            [0, 0.404374, 0.496839, 0.068289, 0.030499, 0,        0,        0,        0,        0,        0,        0,        0]            # Other
        ])


        # Distribution of major blood groups in the donor population.
        # self.donor_ABOD_distr = {"O-":0.1551, "O+":0.3731, "A-":0.0700, "A+":0.3074, "B-":0.0158, "B+":0.0604, "AB-":0.0047, "AB+":0.0134}
        self.donor_ABO_Rhesus_distr = np.array([
        #    A, B, D, C, c, E, e  Frequency
            [0, 0, 0, 0, 1, 0, 1, 0.137937],
            [0, 0, 0, 0, 1, 1, 0, 0.000030],
            [0, 0, 0, 0, 1, 1, 1, 0.004127],
            [0, 0, 0, 1, 0, 0, 1, 0.000038],
            [0, 0, 0, 1, 1, 0, 1, 0.003782],
            [0, 0, 0, 1, 1, 1, 1, 0.000074],
            [0, 0, 1, 0, 1, 0, 0, 0.000000],
            [0, 0, 1, 0, 1, 0, 1, 0.016325],
            [0, 0, 1, 0, 1, 1, 0, 0.009938],
            [0, 0, 1, 0, 1, 1, 1, 0.051529],
            [0, 0, 1, 1, 0, 0, 1, 0.079140],
            [0, 0, 1, 1, 0, 1, 0, 0.000002],
            [0, 0, 1, 1, 0, 1, 1, 0.000470],
            [0, 0, 1, 1, 1, 0, 1, 0.149069],
            [0, 0, 1, 1, 1, 1, 0, 0.000161],
            [0, 0, 1, 1, 1, 1, 1, 0.054948],
            [0, 1, 0, 0, 1, 0, 1, 0.025937],
            [0, 1, 0, 0, 1, 1, 0, 0.000010],
            [0, 1, 0, 0, 1, 1, 1, 0.000784],
            [0, 1, 0, 1, 0, 0, 1, 0.000005],
            [0, 1, 0, 1, 1, 0, 1, 0.000776],
            [0, 1, 0, 1, 1, 1, 1, 0.000012],
            [0, 1, 1, 0, 1, 0, 0, 0.000002],
            [0, 1, 1, 0, 1, 0, 1, 0.004007],
            [0, 1, 1, 0, 1, 1, 0, 0.002079],
            [0, 1, 1, 0, 1, 1, 1, 0.010182],
            [0, 1, 1, 1, 0, 0, 1, 0.019064],
            [0, 1, 1, 1, 0, 1, 0, 0.000001],
            [0, 1, 1, 1, 0, 1, 1, 0.000093],
            [0, 1, 1, 1, 1, 0, 1, 0.030917],
            [0, 1, 1, 1, 1, 1, 0, 0.000035],
            [0, 1, 1, 1, 1, 1, 1, 0.011827],
            [1, 0, 0, 0, 1, 0, 1, 0.074229],
            [1, 0, 0, 0, 1, 1, 0, 0.000024],
            [1, 0, 0, 0, 1, 1, 1, 0.002149],
            [1, 0, 0, 1, 0, 0, 1, 0.000028],
            [1, 0, 0, 1, 1, 0, 1, 0.002011],
            [1, 0, 0, 1, 1, 1, 1, 0.000059],
            [1, 0, 1, 0, 0, 0, 0, 0.000000],
            [1, 0, 1, 0, 1, 0, 0, 0.000000],
            [1, 0, 1, 0, 1, 0, 1, 0.012227],
            [1, 0, 1, 0, 1, 1, 0, 0.007993],
            [1, 0, 1, 0, 1, 1, 1, 0.041218],
            [1, 0, 1, 1, 0, 0, 1, 0.062543],
            [1, 0, 1, 1, 0, 1, 1, 0.000307],
            [1, 0, 1, 1, 1, 0, 1, 0.116197],
            [1, 0, 1, 1, 1, 1, 0, 0.000131],
            [1, 0, 1, 1, 1, 1, 1, 0.043497],
            [1, 1, 0, 0, 1, 0, 1, 0.005704],
            [1, 1, 0, 0, 1, 1, 1, 0.000182],
            [1, 1, 0, 1, 0, 0, 1, 0.000004],
            [1, 1, 0, 1, 1, 0, 1, 0.000164],
            [1, 1, 1, 0, 1, 0, 1, 0.001135],
            [1, 1, 1, 0, 1, 1, 0, 0.000445],
            [1, 1, 1, 0, 1, 1, 1, 0.002401],
            [1, 1, 1, 1, 0, 0, 1, 0.004248],
            [1, 1, 1, 1, 0, 1, 1, 0.000012],
            [1, 1, 1, 1, 1, 0, 1, 0.007176],
            [1, 1, 1, 1, 1, 1, 0, 0.000011],
            [1, 1, 1, 1, 1, 1, 1, 0.002605]
        ])

        # All possible antigen profiles for antigens A, B,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.ABO_phenotypes = np.array([
            [ 0, 0 ],
            [ 0, 1 ],
            [ 1, 0 ],
            [ 1, 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.ABO_prevalences = np.array([
            [0.43,     0.09,     0.44,     0.04],       # Caucasian
            [0.27,     0.49,     0.2 ,     0.04],       # African
            [0.27,     0.43,     0.25,     0.05],       # Asian
            [0.484518, 0.118930, 0.362874, 0.033677]])  # R0 donations

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
