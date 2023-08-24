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

        # LHD configurations -- shortages, mismatches, youngblood, FIFO, usability, substitution, today 
        # LHD = np.array([
        #     # shortages, mismatches, youngblood, FIFO, usability, substitution, today 
        #     [10,         1,          0.1,        0.1,  0.1,       0.1,          1]
        # ])
        LHD = np.hstack([np.tile(100, (SETTINGS.LHD_configs,1)).reshape(-1,1), unpickle(SETTINGS.home_dir + f"LHD/{SETTINGS.LHD_configs}")])
    
        LHD /= LHD.shape[0] - 1
        if LHD.shape[0] < SETTINGS.episodes[1]:
            LHD = np.tile(LHD, (int(np.ceil(SETTINGS.episodes[1] / LHD.shape[0])), 1))

        self.LHD = LHD

        self.BO_params = []
    
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

        # Normalized patient group specific weights, without Fya, Jka and Jkb being mandatory for SCD patients. A weight of 10 means that mismatching is never allowed.   
        # Based on thesis Dorothea Evers
        self.patgroup_weights = np.array([    
        #    A   B   D   C           c           E           e           K           k  Fya         Fyb         Jka         Jkb         M  N  S           s 
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461, 0.00193451],  # Allo
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # SCD
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307, 0.00128967],  # Thal
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307, 0.00128967],  # MDS
            [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461, 0.00193451],  # AIHA
            [10, 10, 10, 0.00289335, 10,         10,         0.00482281, 10,         0, 0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158, 0.00003224],  # Wu45
            [10, 10, 10, 0.00289335, 0.00675227, 0.01832651, 0.00482281, 0.03375967, 0, 0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158, 0.00003224]   # Other
        ])

        # # Normalized patient group specific weights. A weight of 10 means that mismatching is never allowed.   
        # # Based on thesis Dorothea Evers
        # self.patgroup_weights = np.array([
        # #    A   B   D   C           c           E           e           K           k  Fya         Fyb         Jka         Jkb         M  N  S           s
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # Allo
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 10,         0.21438451, 10,         10,         0, 0, 0.05359613, 0.00537363],  # SCD
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845, 0.00214945],  # Thal
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.17147023, 0.0857538,  0.30010093, 0.0857538,  0, 0, 0.02143845, 0.00214945],  # MDS
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # AIHA
        #     [10, 10, 10, 0.00482225, 10,         10,         0.00803802, 10,         0, 0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596, 0.00005374],  # Wu45
        #     [10, 10, 10, 0.00482225, 0.01125378, 0.03054419, 0.00803802, 0.05626612, 0, 0.00428676, 0.00214385, 0.00750252, 0.00214385, 0, 0, 0.00053596, 0.00005374]   # Other
        # ])

        # # Only mandatory matches, no other extensive matching weights.
        # self.patgroup_weights = np.array([    
        # #    A   B   D   C   c   E   e   K   k  Fya Fyb Jka Jkb M  N  S  s 
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Allo
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # SCD
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Thal
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # MDS
        #     [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # AIHA
        #     [10, 10, 10, 0,  10, 10, 0,  10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Wu45
        #     [10, 10, 10, 0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0, 0, 0, 0]     # Other
        # ])

        # # Based on thesis Dorothea Evers - by predicting cohort sizes
        # self.alloimmunization_risks = np.array([
        # #    A,   B,   D,   C,           c,           E,           e,           K,           k,   Fya,         Fyb,         Jka,         Jkb,         M,           N,           S,           s
        #     [0.0, 0.0, 0.0, 0.0,         0.0,         0.0,         0.0,         0.0,         0.0, 0.0,         0.0,         0.0,         0.0,         0.0,         0.0,         0.0,         0.0],  # 0 units
        #     [1.0, 1.0, 1.0, 0.001294636, 0.001234593, 0.010870199, 0.003632550, 0.020513955, 0.0, 0.001273227, 0.000758467, 0.003738532, 0.001978347, 0.001233873, 0.002980283, 0.001243637, 0.0],  # 1 units
        #     [1.0, 1.0, 1.0, 0.001669054, 0.004556867, 0.019214740, 0.007265101, 0.072653743, 0.0, 0.003230319, 0.001516935, 0.003094937, 0.002637796, 0.001513514, 0.003973711, 0.001658182, 0.0],  # 2 units
        #     [1.0, 1.0, 1.0, 0.002141318, 0.005585708, 0.014771184, 0.007265101, 0.141460550, 0.0, 0.002041797, 0.001860332, 0.003830787, 0.003956694, 0.001883272, 0.004470425, 0.002487274, 0.0],  # 3 units
        #     [1.0, 1.0, 1.0, 0.002731609, 0.002272338, 0.028128720, 0.007265101, 0.106486188, 0.0, 0.002570150, 0.002356421, 0.004720077, 0.003895424, 0.002253030, 0.004768453, 0.003601397, 0.0],  # 4 units
        #     [1.0, 1.0, 1.0, 0.003458547, 0.002759584, 0.033829978, 0.007265101, 0.118975309, 0.0, 0.003285354, 0.002852510, 0.014476021, 0.003834154, 0.004415970, 0.004967139, 0.004715520, 0.0],  # 5 units
        #     [1.0, 1.0, 1.0, 0.004423851, 0.003333595, 0.062333059, 0.007265101, 0.776134228, 0.0, 0.004000557, 0.003348598, 0.007061372, 0.003772884, 0.006578910, 0.005109057, 0.005829643, 0.0],  # 6 units
        #     [1.0, 1.0, 1.0, 0.005389155, 0.003995296, 0.035791716, 0.007265101, 0.776134228, 0.0, 0.005014972, 0.005819539, 0.008548951, 0.003772884, 0.005627483, 0.005215495, 0.005829643, 0.0],  # 7 units
        #     [1.0, 1.0, 1.0, 0.006606061, 0.004755879, 0.124154351, 0.007265101, 0.776134228, 0.0, 0.006029387, 0.008290481, 0.005131165, 0.003772884, 0.004676056, 0.005960566, 0.005829643, 0.0],  # 8 units
        #     [1.0, 1.0, 1.0, 0.008142281, 0.005610304, 0.098836601, 0.007265101, 0.776134228, 0.0, 0.009767681, 0.010761422, 0.012225000, 0.003772884, 0.011014218, 0.005960566, 0.005829643, 0.0],  # 9 units
        #     [1.0, 1.0, 1.0, 0.009678502, 0.019631741, 0.124777105, 0.007265101, 0.776134228, 0.0, 0.013505975, 0.013232364, 0.014567323, 0.003772884, 0.009213648, 0.005960566, 0.005829643, 0.0],  # 10 units
        #     [1.0, 1.0, 1.0, 0.011214723, 0.031396155, 0.150717608, 0.007265101, 0.776134228, 0.0, 0.017244269, 0.015703305, 0.016909646, 0.003772884, 0.007413078, 0.005960566, 0.005829643, 0.0],  # 11 units
        #     [1.0, 1.0, 1.0, 0.012750943, 0.043160569, 0.171067665, 0.007265101, 0.776134228, 0.0, 0.020982563, 0.018174246, 0.019251969, 0.003772884, 0.016145847, 0.005960566, 0.005829643, 0.0],  # 12 units
        #     [1.0, 1.0, 1.0, 0.028180287, 0.045704771, 0.191417722, 0.007265101, 0.776134228, 0.0, 0.024720857, 0.020645188, 0.028009443, 0.003772884, 0.024878616, 0.005960566, 0.005829643, 0.0],  # 13 units
        #     [1.0, 1.0, 1.0, 0.043609631, 0.048248973, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.028459151, 0.023116129, 0.036766917, 0.003772884, 0.033611385, 0.005960566, 0.005829643, 0.0],  # 14 units
        #     [1.0, 1.0, 1.0, 0.059038976, 0.050793175, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.032197445, 0.023116129, 0.040353934, 0.003772884, 0.042344154, 0.005960566, 0.005829643, 0.0],  # 15 units
        #     [1.0, 1.0, 1.0, 0.074468320, 0.053337377, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.035935739, 0.023116129, 0.043940950, 0.003772884, 0.051076923, 0.005960566, 0.005829643, 0.0],  # 16 units
        #     [1.0, 1.0, 1.0, 0.072414965, 0.055881579, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.039674033, 0.023116129, 0.047527966, 0.003772884, 0.051076923, 0.005960566, 0.005829643, 0.0],  # 17 units
        #     [1.0, 1.0, 1.0, 0.070361611, 0.055881579, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.039674033, 0.023116129, 0.051114983, 0.003772884, 0.051076923, 0.005960566, 0.005829643, 0.0],  # 18 units
        #     [1.0, 1.0, 1.0, 0.068308256, 0.055881579, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.039674033, 0.023116129, 0.051114983, 0.003772884, 0.051076923, 0.005960566, 0.005829643, 0.0],  # 19 units
        #     [1.0, 1.0, 1.0, 0.066254902, 0.055881579, 0.618627273, 0.007265101, 0.776134228, 0.0, 0.039674033, 0.023116129, 0.051114983, 0.003772884, 0.051076923, 0.005960566, 0.005829643, 0.0],  # 20 units
        # ])

        # Based on thesis Dorothea Evers - by estimating ps and qs
        self.alloimmunization_risks = np.array([
        #    A,   B,   D,   C,        c,        E,        e,        K,        k,   Fya,      Fyb,      Jka,      Jkb,      M,        N,        S,        s
            [0.0, 0.0, 0.0, 0.0,      0.0,      0.0,      0.0,      0.0,      0.0, 0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0     ], # 0 units
            [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.0, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360, 0.000020], # 1 unit
            [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.0, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360, 0.000020], # 2 units
            [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.0, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360, 0.000020], # 3 units
            [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.0, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360, 0.000020], # 4 units
            [1.0, 1.0, 1.0, 0.001655, 0.002751, 0.011995, 0.001471, 0.024007, 0.0, 0.001616, 0.000612, 0.004349, 0.000635, 0.000928, 0.000019, 0.000360, 0.000020], # 5 units
            [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.0, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385, 0.000019], # 6 units
            [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.0, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385, 0.000019], # 7 units
            [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.0, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385, 0.000019], # 8 units
            [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.0, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385, 0.000019], # 9 units
            [1.0, 1.0, 1.0, 0.002057, 0.005491, 0.04177,  0.000019, 0.143273, 0.0, 0.001905, 0.000795, 0.005853, 0.000902, 0.003732, 0.000946, 0.001385, 0.000019], # 10 units
            [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.0, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018, 0.000019], # 11 units
            [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.0, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018, 0.000019], # 12 units
            [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.0, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018, 0.000019], # 13 units
            [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.0, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018, 0.000019], # 14 units
            [1.0, 1.0, 1.0, 0.002313, 0.008074, 0.127877, 0.000019, 0.000021, 0.0, 0.000019, 0.003397, 0.008961, 0.000019, 0.001507, 0.000019, 0.000018, 0.000019], # 15 units
            [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.0, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019, 0.000010], # 16 units
            [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.0, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019, 0.000010], # 17 units
            [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.0, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019, 0.000010], # 18 units
            [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.0, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019, 0.000010], # 19 units
            [1.0, 1.0, 1.0, 0.028287, 0.011268, 0.000018, 0.00001,  0.000017, 0.0, 0.008210, 0.000019, 0.009680, 0.000019, 0.012417, 0.000019, 0.000019, 0.000010], # 20 units
        ])

        # # Based on thesis Tom vd Woude -- age/sex parameter: -0.137
        # self.alloimmunization_risks = np.array([
        # #    A, B, D, C,          c,          E,          e,          K,          k, Fya,        Fyb,        Jka,        Jkb,        M,          N  S           s
        #     [0, 0, 0, 0,          0,          0,          0,          0,          0, 0,          0,          0,          0,          0,          0, 0,          0], # 0 units
        #     [1, 1, 1, 0.00049106, 0.00077621, 0.00832606, 0.00006923, 0.01617738, 0, 0.00043673, 0.000038,   0.00049141, 0.0000459 , 0.00012247, 0, 0.00008117, 0], # 1 unit
        #     [1, 1, 1, 0.00075047, 0.00116883, 0.01150707, 0.00011222, 0.02174881, 0, 0.00066994, 0.00006263, 0.00075099, 0.00007526, 0.00019527, 0, 0.00013097, 0], # 2 units
        #     [1, 1, 1, 0.00037366, 0.00059618, 0.00674999, 0.00005077, 0.01334575, 0, 0.00033155, 0.00002757, 0.00037394, 0.00003341, 0.00009074, 0, 0.00005969, 0], # 3 units
        #     [1, 1, 1, 0.00029237, 0.00047033, 0.00558556, 0.00003844, 0.01121518, 0, 0.00025888, 0.00002068, 0.00029258, 0.00002514, 0.00006934, 0, 0.00004531, 0], # 4 units
        #     [1, 1, 1, 0.00035473, 0.00056696, 0.00648485, 0.00004786, 0.01286377, 0, 0.00031461, 0.00002594, 0.00035498, 0.00003146, 0.00008571, 0, 0.0000563,  0], # 5 units
        #     [1, 1, 1, 0.00018949, 0.00030922, 0.00398899, 0.00002355, 0.00822669, 0, 0.00016719, 0.00001247, 0.00018964, 0.00001523, 0.00004315, 0, 0.00002788, 0], # 6 units
        #     [1, 1, 1, 0.00016559, 0.00027139, 0.00359084, 0.00002022, 0.0074665,  0, 0.00014594, 0.00001066, 0.00016572, 0.00001304, 0.00003724, 0, 0.00002398, 0], # 7 units
        #     [1, 1, 1, 0.00027781, 0.00044767, 0.00536918, 0.00003628, 0.01081509, 0, 0.00024588, 0.00001949, 0.00027802, 0.0000237 , 0.00006557, 0, 0.00004279, 0], # 8 units
        #     [1, 1, 1, 0.00005649, 0.00009574, 0.00154019, 0.00000604, 0.00341214, 0, 0.00004936, 0.00000306, 0.00005653, 0.00000379, 0.00001154, 0, 0.00000723, 0], # 9 units
        #     [1, 1, 1, 0.0000587,  0.0000936,  0.0035428,  0.0000014,  0.0117878,  0, 0.0000651,  0.0000094,  0.0000886,  0.0000088,  0.0000201,  0, 0.0000328,  0]  # average probability  
        # ])    


        # self.alloimmunization_risks[:,-4:] = 1


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
