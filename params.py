import numpy as np
import pickle

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
        # self.antigens = {0:"A", 1:"B", 2:"D", 3:"C", 4:"c", 5:"E", 6:"e", 7:"K", 8:"k", 9:"Fya", 10:"Fyb", 11:"Jka", 12:"Jkb", 13:"M", 14:"N", 15:"S", 16:"s"}
        self.antigens = {0:"A", 1:"B", 2:"D", 3:"C", 4:"c", 5:"E", 6:"e", 7:"K", 8:"Fya", 9:"Fyb", 10:"Jka", 11:"Jkb", 12:"M", 13:"N", 14:"S"}
        self.ethnicities = {0 : "Caucasian", 1 : "African", 2 : "Asian"}

        #####################
        # DEMAND AND SUPPLY #
        #####################

        # Names of all considered patient groups.
        self.patgroups = {0 : "Allo", 1 : "SCD", 2 : "Thal", 3 : "MDS", 4 : "AIHA", 5 : "Wu45", 6 : "Other"}

        # Weekly demand per hospital and patient group.
        self.weekly_demand = {
        #                       Allo    SCD     Thal    MDS     AIHA    Wu45    Other
            "UCLH" : np.array([ 0,      34,     0,      0,      0,      0,      572]),
            "NMUH": np.array([  0,      11,     0,      0,      0,      0,      197]),
            "WH" : np.array([   0,      10,     0,      0,      0,      0,      230])
        }
        self.weekly_demand["London"] = self.weekly_demand["UCLH"] + self.weekly_demand["NMUH"] + self.weekly_demand["WH"]

        if sum(SETTINGS.n_hospitals.values()) > 1:
            self.supply_size = round((SETTINGS.init_days + SETTINGS.test_days) * SETTINGS.inv_size_factor_dc * sum([SETTINGS.n_hospitals[htype] * (sum(self.weekly_demand[htype])/7) for htype in SETTINGS.n_hospitals.keys()]))
        else:
            self.supply_size = round((SETTINGS.init_days + SETTINGS.test_days) * SETTINGS.inv_size_factor_hosp * sum([SETTINGS.n_hospitals[htype] * (sum(self.weekly_demand[htype])/7) for htype in SETTINGS.n_hospitals.keys()]))


        ###########################
        # LATIN HYPERCUBE DESIGNS #
        ###########################
        # Note: all objectives are normalized so no need to consider that in determining these weights.

        LHD = np.array([
            # shortages, mismatches, youngblood, FIFO, usability, substitution, today 
            [10,         0,        0,        0,  0,       0,          0]
        ])
        # LHD = np.tile(LHD, (SETTINGS.LHD_configs,1))
        # LHD[:,1] = [i/max(1,SETTINGS.LHD_configs-1) for i in range(SETTINGS.LHD_configs)]

        # LHD configurations -- shortages, mismatches, youngblood, FIFO, usability, substitution, today 
        # LHD = np.hstack([np.tile(10, (SETTINGS.LHD_configs,1)).reshape(-1,1), unpickle(SETTINGS.home_dir + f"LHD/{SETTINGS.LHD_configs}")])
        
        if LHD.shape[0] < SETTINGS.episodes[1]:
            LHD = np.tile(LHD, (int(np.ceil(SETTINGS.episodes[1] / LHD.shape[0])), 1))
        self.LHD = LHD

        self.BO_params = []
    
        ###########
        # WEIGHTS #
        ###########

        # No mismatch weights - major antigens have a weight of 10 to denote that they can not be mismatched.
        self.major_weights = np.array(([10] * 3) + ([0] * 12))  

        # Normalized alloimmunization incidence after mismatching on 3 units. A weight of 10 means that mismatching is never allowed.
        # Based on thesis Dorothea Evers
        self.relimm_weights = np.array([
        #    A   B   D   C       c        E        e        K        Fya      Fyb      Jka      Jkb      M        N        S
            [10, 10, 10, 0.0344, 0.08028, 0.21789, 0.05734, 0.40138, 0.04587, 0.02294, 0.08028, 0.02294, 0.02294, 0.00115, 0.01147]
        ])

        # Normalized patient group specific weights, without Fya, Jka and Jkb being mandatory for SCD patients. A weight of 10 means that mismatching is never allowed.   
        # Based on thesis Dorothea Evers
        self.patgroup_weights = np.array([    
        #    A   B   D   C           c           E           e           K           Fya         Fyb         Jka         Jkb         M  N  S          
            [10, 10, 10, 10,         10,         10,         10,         10,         0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461],  # Allo
            [10, 10, 10, 10,         10,         10,         10,         10,         0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768],  # SCD
            [10, 10, 10, 10,         10,         10,         10,         10,         0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307],  # Thal
            [10, 10, 10, 10,         10,         10,         10,         10,         0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307],  # MDS
            [10, 10, 10, 10,         10,         10,         10,         10,         0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461],  # AIHA
            [10, 10, 10, 0.00289335, 10,         10,         0.00482281, 10,         0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158],  # Wu45
            [10, 10, 10, 0.00289335, 0.00675227, 0.01832651, 0.00482281, 0.03375967, 0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158]   # Other
        ])

        # # Normalized patient group specific weights. A weight of 10 means that mismatching is never allowed.   
        # # Based on thesis Dorothea Evers
        # self.patgroup_weights = np.array([
        # #    A   B   D   C           c           E           e           K           Fya         Fyb         Jka         Jkb         M  N  S          
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768],  # Allo
        #     [10, 10, 10, 10,         10,         10,         10,         10,         10,         0.21438451, 10,         10,         0, 0, 0.05359613],  # SCD
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845],  # Thal
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845],  # MDS
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768],  # AIHA
        #     [10, 10, 10, 0.00482225, 10,         10,         0.00803802, 10,         0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596],  # Wu45
        #     [10, 10, 10, 0.00482225, 0.01125378, 0.03054419, 0.00803802, 0.05626612, 0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596]   # Other
        # ])

        # # Only mandatory matches, no other extensive matching weights.
        # self.patgroup_weights = np.array([    
        # #    A   B   D   C   c   E   e   K   Fya Fyb Jka Jkb M  N  S 
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0,  0,  0,  0,  0, 0, 0],    # Allo
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0,  0,  0,  0,  0, 0, 0],    # SCD
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0,  0,  0,  0,  0, 0, 0],    # Thal
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0,  0,  0,  0,  0, 0, 0],    # MDS
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0,  0,  0,  0,  0, 0, 0],    # AIHA
        #     [10, 10, 10, 0,  10, 10, 0,  10, 0,  0,  0,  0,  0, 0, 0],    # Wu45
        #     [10, 10, 10, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0]     # Other
        # ])

        # # Based on thesis Dorothea Evers - by estimating ps and qs
        # self.alloimmunization_risks = np.array([
        # #    A,   B,   D,   C,        c,        E,        e,        K,        Fya,      Fyb,      Jka,      Jkb,      M,        N,        S,      
        #     [0.0, 0.0, 0.0, 0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,    ], # 0 units
        #     [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360], # 1 unit
        #     [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360], # 2 units
        #     [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360], # 3 units
        #     [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360], # 4 units
        #     [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360], # 5 units
        #     [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385], # 6 units
        #     [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385], # 7 units
        #     [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385], # 8 units
        #     [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385], # 9 units
        #     [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385], # 10 units
        #     [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018], # 11 units
        #     [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018], # 12 units
        #     [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018], # 13 units
        #     [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018], # 14 units
        #     [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018], # 15 units
        #     [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019], # 16 units
        #     [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019], # 17 units
        #     [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019], # 18 units
        #     [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019], # 19 units
        #     [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019], # 20 units
        # ])

        # Based on thesis Dorothea Evers - by estimating cohort size and alloimmunized patients
        self.alloimmunization_risks = np.array([
        #    A,   B,   D,   C,        c,        E,        e,        K,        Fya,      Fyb,      Jka,      Jkb,      M,        N,        S
            [0.0, 0.0, 0.0, 0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,    ],  # 0 units
            [1.0, 1.0, 1.0, 0.000806, 0.001932, 0.001829, 0.000841, 0.009394, 0.00145,  0.000458, 0.003131, 0.000682, 0.000831, 0.0,      0.000042],  # 1 units
            [1.0, 1.0, 1.0, 0.001019, 0.002279, 0.019672, 0.003072, 0.033052, 0.001527, 0.000528, 0.003538, 0.000674, 0.001001, 0.0,      0.000724],  # 2 units
            [1.0, 1.0, 1.0, 0.001283, 0.002681, 0.017798, 0.002159, 0.04402,  0.001604, 0.000606, 0.003987, 0.000664, 0.001202, 0.0,      0.000574],  # 3 units
            [1.0, 1.0, 1.0, 0.001607, 0.003142, 0.019416, 0.00184,  0.070947, 0.001679, 0.000695, 0.004479, 0.000652, 0.00144,  0.000049, 0.00055 ],  # 4 units
            [1.0, 1.0, 1.0, 0.002002, 0.003666, 0.022943, 0.001702, 0.124275, 0.00175,  0.000794, 0.005009, 0.000638, 0.001717, 0.000426, 0.000571],  # 5 units
            [1.0, 1.0, 1.0, 0.002476, 0.004257, 0.028181, 0.001644, 0.226898, 0.001813, 0.000904, 0.005577, 0.000621, 0.002037, 0.000435, 0.000621],  # 6 units
            [1.0, 1.0, 1.0, 0.003035, 0.004918, 0.035122, 0.001641, 0.417154, 0.001867, 0.001022, 0.006172, 0.000601, 0.002405, 0.000457, 0.00069 ],  # 7 units
            [1.0, 1.0, 1.0, 0.00368,  0.005635, 0.043673, 0.001655, 0.458333, 0.001906, 0.00115,  0.006782, 0.000578, 0.002818, 0.000486, 0.000776],  # 8 units
            [1.0, 1.0, 1.0, 0.004406, 0.006414, 0.053451, 0.001699, 0.384615, 0.001929, 0.001285, 0.007388, 0.000552, 0.003279, 0.000521, 0.000874],  # 9 units
            [1.0, 1.0, 1.0, 0.005203, 0.007235, 0.063439, 0.001741, 0.291667, 0.001931, 0.001425, 0.007988, 0.000521, 0.003783, 0.000559, 0.00098 ],  # 10 units
            [1.0, 1.0, 1.0, 0.006046, 0.008069, 0.072564, 0.0018,   0.176471, 0.001909, 0.001567, 0.008529, 0.000487, 0.004319, 0.000599, 0.001086],  # 11 units
            [1.0, 1.0, 1.0, 0.006905, 0.008926, 0.079892, 0.001865, 0.071429, 0.00186,  0.001705, 0.009013, 0.00045,  0.004885, 0.000637, 0.001186],  # 12 units
            [1.0, 1.0, 1.0, 0.007736, 0.009732, 0.084433, 0.001895, 0.076923, 0.001787, 0.001839, 0.009404, 0.00041,  0.005464, 0.000673, 0.001275],  # 13 units
            [1.0, 1.0, 1.0, 0.008519, 0.0105,   0.07173,  0.00196,  0.0,      0.00169,  0.001956, 0.009699, 0.000369, 0.006033, 0.000705, 0.001342],  # 14 units
            [1.0, 1.0, 1.0, 0.009223, 0.011196, 0.045455, 0.00199,  0.0,      0.001575, 0.002058, 0.009862, 0.000327, 0.0066,   0.000729, 0.001391],  # 15 units
            [1.0, 1.0, 1.0, 0.009818, 0.01179,  0.028571, 0.002008, 0.0,      0.001442, 0.002138, 0.009892, 0.000286, 0.007117, 0.000747, 0.001415],  # 16 units
            [1.0, 1.0, 1.0, 0.010275, 0.012239, 0.02451,  0.002004, 0.0,      0.001304, 0.002195, 0.009792, 0.000246, 0.007583, 0.000755, 0.001419],  # 17 units
            [1.0, 1.0, 1.0, 0.01062,  0.012588, 0.01005,  0.002015, 0.0,      0.00116,  0.002237, 0.009613, 0.00021,  0.007999, 0.00076,  0.00141 ],  # 18 units
            [1.0, 1.0, 1.0, 0.010879, 0.012794, 0.010152, 0.002042, 0.0,      0.001023, 0.002245, 0.009302, 0.000176, 0.008353, 0.000758, 0.001386],  # 19 units
            [1.0, 1.0, 1.0, 0.01103,  0.01291,  0.005128, 0.001985, 0.0,      0.000889, 0.002233, 0.008907, 0.000146, 0.008645, 0.000748, 0.001358],  # 20 units
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
            [ 0 ],
            [ 1 ]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Kell_prevalences = np.array([
            [0.91 , 0.09 ],   # Caucasian
            [0.98 , 0.02 ],   # African
            [1.   , 0.   ]])  # Asian

        # All possible antigen profiles for antigens M, N, S, s,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.MNS_phenotypes = np.array([
            [1,0,1],
            [1,0,0],
            [1,1,1],
            [1,1,0],
            [0,1,1],
            [0,1,0]])

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.MNS_prevalences  = np.array([
            [0.20 , 0.08  , 0.28 , 0.22 , 0.07 , 0.15  ],   # Caucasian
            [0.09 , 0.164 , 0.15 , 0.329, 0.07 , 0.197 ],   # African
            [0.20 , 0.08  , 0.28 , 0.22 , 0.07 , 0.15  ]])  # Asian

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

def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)
