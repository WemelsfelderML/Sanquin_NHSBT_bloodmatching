import numpy as np

class Params():
    
    def __init__(self, SETTINGS, BO_params=[]):


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

        # Note: all objectives are normalized so no need to consider
        # that in determining these weights.
        # ranges = np.array([
        #     [10, 1000], # shortages
        #     [10, 500],  # mismatches
        #     [0, 100],   # youngblood
        #     [0, 100],   # FIFO
        #     [0, 100],   # usability
        #     [0, 100],   # substitution
        #     [10, 500]]) # today
        ranges = np.array([
            [1000, 1000],   # shortages
            [10, 10],       # mismatches
            [1, 1],         # youngblood
            [1, 1],         # FIFO
            [1, 1],         # usability
            [1, 1],         # substitution
            [10, 10]        # today
        ])

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

        self.BO_params = np.array(BO_params)
    
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

        # # Normalized patient group specific weights, without Fya, Jka and Jkb being mandatory for SCD patients. A weight of 10 means that mismatching is never allowed.   
        # # Based on thesis Dorothea Evers
        # self.patgroup_weights = np.array([    
        # #    A   B   D   C           c           E           e           K           k  Fya         Fyb         Jka         Jkb         M  N  S           s 
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461, 0.00193451],  # Allo
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.25720534, 0.12863071, 0.4501514,  0.12863071, 0, 0, 0.03215768, 0.00322418],  # SCD
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307, 0.00128967],  # Thal
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.10288214, 0.05145228, 0.18006056, 0.05145228, 0, 0, 0.01286307, 0.00128967],  # MDS
        #     [10, 10, 10, 10,         10,         10,         10,         10,         0, 0.1543232,  0.07717842, 0.27009084, 0.07717842, 0, 0, 0.01929461, 0.00193451],  # AIHA
        #     [10, 10, 10, 0.00289335, 10,         10,         0.00482281, 10,         0, 0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158, 0.00003224],  # Wu45
        #     [10, 10, 10, 0.00289335, 0.00675227, 0.01832651, 0.00482281, 0.03375967, 0, 0.00257205, 0.00128631, 0.00450151, 0.00128631, 0, 0, 0.00032158, 0.00003224]   # Other
        # ])

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


        # Only mandatory matches, no other extensive matching weights.
        self.patgroup_weights = np.array([    
        #    A   B   D   C   c   E   e   K   k  Fya Fyb Jka Jkb M  N  S  s 
            [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Allo
            [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # SCD
            [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Thal
            [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # MDS
            [10, 10, 10, 10, 10, 10, 10, 10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # AIHA
            [10, 10, 10, 0,  10, 10, 0,  10, 0, 0,  0,  0,  0,  0, 0, 0, 0],    # Wu45
            [10, 10, 10, 0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0, 0, 0, 0]     # Other
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

        # Based on thesis Tom vd Woude -- age/sex parameter: 0.0
        self.alloimmunization_risks = np.array([
        #    A, B, D, C,          c,          E,          e,          K,          k, Fya,        Fyb,        Jka,        Jkb,        M,          N  S           s
            [0, 0, 0, 0,          0,          0,          0,          0,          0, 0,          0,          0,          0,          0,          0, 0,          0], # 0 units
            [0, 0, 0, 0.00079264, 0.00123211, 0.01199467, 0.00011944, 0.02258864, 0, 0.00070793, 0.00006681, 0.00079319, 0.00008023, 0.00020739, 0, 0.00013931, 0], # 1 unit
            [0, 0, 0, 0.00119275, 0.0018269 , 0.01633156, 0.00019058, 0.0299248 , 0, 0.00106922, 0.0001084 , 0.00119354, 0.0001295 , 0.00032553, 0, 0.00022129, 0], # 2 units
            [0, 0, 0, 0.00060907, 0.00095559, 0.0098162 , 0.00008845, 0.01880868, 0, 0.0005427 , 0.00004896, 0.00060949, 0.00005899, 0.00015518, 0, 0.00010347, 0], # 3 units
            [0, 0, 0, 0.00048068, 0.00076036, 0.00819082, 0.00006757, 0.01593649, 0, 0.00042742, 0.00003705, 0.00048102, 0.00004477, 0.00011963, 0, 0.00007924, 0], # 4 units
            [0, 0, 0, 0.00057926, 0.00091041, 0.0094474 , 0.00008354, 0.01816119, 0, 0.00051592, 0.00004615, 0.00057967, 0.00005564, 0.00014685, 0, 0.00009777, 0], # 5 units
            [0, 0, 0, 0.00031623, 0.00050738, 0.00593476, 0.00004202, 0.01185797, 0, 0.00028019, 0.00002267, 0.00031646, 0.00002753, 0.00007557, 0, 0.00004949, 0], # 6 units
            [0, 0, 0, 0.00027761, 0.00044735, 0.0053661 , 0.00003625, 0.01080938, 0, 0.0002457 , 0.00001947, 0.00027781, 0.00002368, 0.00006552, 0, 0.00004276, 0], # 7 units
            [0, 0, 0, 0.00045756, 0.00072502, 0.00788707, 0.00006389, 0.01539408, 0, 0.00040669, 0.00003497, 0.00045789, 0.00004228, 0.00011332, 0, 0.00007496, 0], # 8 units
            [0, 0, 0, 0.00009808, 0.00016345, 0.0023818 , 0.00001122, 0.00510993, 0, 0.00008608, 0.0000058 , 0.00009816, 0.00000714, 0.00002104, 0, 0.00001336, 0], # 9 units
            [1, 1, 1, 0.0000587,  0.0000936,  0.0035428,  0.0000014,  0.0117878,  0, 0.0000651,  0.0000094,  0.0000886,  0.0000088,  0.0000201,  0, 0.0000328,  0]  # average probability  
        ])
        
        # # Based on thesis Tom vd Woude -- age/sex parameter: +1.0
        # self.alloimmunization_risks = np.array([
        # #    A, B, D, C,          c,          E,          e,          K,          k, Fya,        Fyb,        Jka,        Jkb,        M,          N  S           s
        #     [0, 0, 0, 0,          0,          0,          0,          0,          0, 0,          0,          0,          0,          0,          0, 0,          0], # 0 units
        #     [0, 0, 0, 0.01544061, 0.02129544, 0.10432251, 0.00374874, 0.15793043, 0, 0.01421143, 0.00240343, 0.01544837, 0.00276603, 0.005695  , 0, 0.00421431, 0], # 1 unit
        #     [0, 0, 0, 0.02079998, 0.0282862 , 0.12793644, 0.00534304, 0.18891544, 0, 0.01921213, 0.00348159, 0.02080999, 0.00398663, 0.00798939, 0, 0.00598038, 0], # 2 units
        #     [0, 0, 0, 0.01272181, 0.01770231, 0.09121669, 0.00298067, 0.14029386, 0, 0.0116829 , 0.00189157, 0.01272838, 0.0021839 , 0.00457364, 0, 0.00336011, 0], # 3 units
        #     [0, 0, 0, 0.01067887, 0.01497693, 0.08071175, 0.00242448, 0.12589758, 0, 0.00978739, 0.00152477, 0.01068452, 0.00176538, 0.00375321, 0, 0.00273978, 0], # 4 units
        #     [0, 0, 0, 0.01225943, 0.01708753, 0.08889354, 0.00285313, 0.13713105, 0, 0.01125353, 0.00180714, 0.01226579, 0.00208768, 0.00438619, 0, 0.003218  , 0], # 5 units
        #     [0, 0, 0, 0.00781806, 0.01111543, 0.06475968, 0.00168068, 0.10352838, 0, 0.00714083, 0.00104058, 0.00782236, 0.00121068, 0.00264192, 0, 0.0019073 , 0], # 6 units
        #     [0, 0, 0, 0.0070914 , 0.01012455, 0.0604149 , 0.00149921, 0.09731559, 0, 0.0064703 , 0.00092379, 0.00709534, 0.0010764 , 0.00236772, 0, 0.00170356, 0], # 7 units
        #     [0, 0, 0, 0.01029553, 0.01446276, 0.07866627, 0.00232231, 0.12306512, 0, 0.00943219, 0.0014578 , 0.01030099, 0.00168882, 0.00360161, 0, 0.00262565, 0], # 8 units
        #     [0, 0, 0, 0.0032258 , 0.00475714, 0.034182  , 0.00059933, 0.05840557, 0, 0.00291939, 0.00035589, 0.00322776, 0.0004195 , 0.00098141, 0, 0.00068785, 0], # 9 units
        #     [1, 1, 1, 0.0000587,  0.0000936,  0.0035428,  0.0000014,  0.0117878,  0, 0.0000651,  0.0000094,  0.0000886,  0.0000088,  0.0000201,  0, 0.0000328,  0]  # average probability  
        # ])
        #         # Based on thesis Tom vd Woude -- age/sex parameter: +2.0
        # self.alloimmunization_risks = np.array([
        # #    A, B, D, C,          c,          E,          e,          K,          k, Fya,        Fyb,        Jka,        Jkb,        M,          N  S           s
        #     [0, 0, 0, 0,          0,          0,          0,          0,          0, 0,          0,          0,          0,          0,          0, 0,          0], # 0 units
        #     [0, 0, 0, 0.12330963, 0.15204548, 0.3984736 , 0.04707513, 0.49880317, 0, 0.1167483 , 0.03440235, 0.12335042, 0.03800677, 0.06294651, 0, 0.05108744, 0], # 1 unit
        #     [0, 0, 0, 0.14975145, 0.18230918, 0.44583159, 0.06023546, 0.54700579, 0, 0.14224215, 0.04469729, 0.14979803, 0.04914508, 0.07935846, 0, 0.06510174, 0], # 2 units
        #     [0, 0, 0, 0.1084897 , 0.13486165, 0.36945392, 0.04006779, 0.46851631, 0, 0.10250615, 0.02899986, 0.10852694, 0.03213516, 0.05408205, 0, 0.04359597, 0], # 3 units
        #     [0, 0, 0, 0.09652658, 0.12085969, 0.34446778, 0.03461621, 0.44196069, 0, 0.0910364 , 0.02484052, 0.09656079, 0.02759991, 0.04711445, 0, 0.03775123, 0], # 4 units
        #     [0, 0, 0, 0.10585083, 0.1317835 , 0.3640703 , 0.03884916, 0.46283262, 0, 0.09997395, 0.02806661, 0.10588741, 0.03111875, 0.05253027, 0, 0.04229078, 0], # 5 units
        #     [0, 0, 0, 0.07819762, 0.09915175, 0.3029272 , 0.02664275, 0.39677637, 0, 0.07351532, 0.01883627, 0.07822685, 0.02102626, 0.03679131, 0, 0.02917229, 0], # 6 units
        #     [0, 0, 0, 0.07316742, 0.09313403, 0.29071402, 0.02453977, 0.38323325, 0, 0.06871895, 0.01727045, 0.07319521, 0.01930591, 0.03403826, 0, 0.0269027 , 0], # 7 units
        #     [0, 0, 0, 0.09418775, 0.1181076 , 0.33940044, 0.03357267, 0.43651931, 0, 0.08879703, 0.02404909, 0.09422135, 0.02673534, 0.04577284, 0, 0.0366306 , 0], # 8 units
        #     [0, 0, 0, 0.04236295, 0.0555801 , 0.20536775, 0.01257145, 0.28491564, 0, 0.03948448, 0.00853956, 0.04238101, 0.0096522 , 0.01804997, 0, 0.01391409, 0], # 9 units
        #     [1, 1, 1, 0.0000587,  0.0000936,  0.0035428,  0.0000014,  0.0117878,  0, 0.0000651,  0.0000094,  0.0000886,  0.0000088,  0.0000201,  0, 0.0000328,  0]  # average probability  
        # ])


        # self.alloimmunization_risks[:,-4:] = 1


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
