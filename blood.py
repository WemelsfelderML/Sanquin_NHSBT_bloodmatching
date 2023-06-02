import random
import numpy as np

class Blood:

    # An instance of this class is created for both patient requests and inventory products, to contain their phenotype.
    def __init__(self, PARAMS, index = None, ethnicity = None, patgroup = None, major = [], minor = [], num_units = 0, day_issuing = 0, day_available = 0, age = 0, antibodies = [], mism_units = []):

        vector = np.zeros(len(PARAMS.antigens))

        # If no major blood group is given, randomly generate a phenotype for the ABO and Rhesus systems.
        if major == []:
            vector[:2] = random.choices(PARAMS.ABO_phenotypes, weights = PARAMS.ABO_prevalences[ethnicity], k=1)[0]
            vector[2:7] = random.choices(PARAMS.Rhesus_phenotypes, weights = PARAMS.Rhesus_prevalences[ethnicity], k=1)[0]

        else:
            vector[:3] = major

            # If no minor blood group is given..
            if minor == []:
                Rhesus_phenotypes_Dpos = []
                Rhesus_phenotypes_Dneg = []
                Rhesus_prevalences_Dpos = []
                Rhesus_prevalences_Dneg = []

                for i in range(len(PARAMS.Rhesus_phenotypes)):
                    phenotype = PARAMS.Rhesus_phenotypes[i]
                    prevalence = PARAMS.Rhesus_prevalences[ethnicity][i]
                    if phenotype[0] == 1:
                        Rhesus_phenotypes_Dpos.append(phenotype)
                        Rhesus_prevalences_Dpos.append(prevalence)
                    else:
                        Rhesus_phenotypes_Dneg.append(phenotype)
                        Rhesus_prevalences_Dneg.append(prevalence)

                # Add RhD+ or RhD- to the blood vector, and randomly generate other Rhesus antigens based on their prevalences given RhD.
                if major[2] == 1:
                    vector[2:7] = random.choices(Rhesus_phenotypes_Dpos, weights = Rhesus_prevalences_Dpos, k=1)[0]
                else:
                    vector[2:7] = random.choices(Rhesus_phenotypes_Dneg, weights = Rhesus_prevalences_Dneg, k=1)[0]

        # If no minor blood group is given, randomly generate a phenotype for the Kell, MNS, Duffy and Kidd systems.
        if minor == []:
            vector[7:9] = random.choices(PARAMS.Kell_phenotypes, weights = PARAMS.Kell_prevalences[ethnicity], k=1)[0]
            vector[9:11] = random.choices(PARAMS.Duffy_phenotypes, weights = PARAMS.Duffy_prevalences[ethnicity], k=1)[0]
            vector[11:13] = random.choices(PARAMS.Kidd_phenotypes, weights = PARAMS.Kidd_prevalences[ethnicity], k=1)[0]
            vector[13:] = random.choices(PARAMS.MNS_phenotypes, weights = PARAMS.MNS_prevalences[ethnicity], k=1)[0]
        else:
            vector[3:] = minor
        
        self.vector = vector

        # # Binary string for antigens considered mandatory per patient group.
        # if ("patgroups" in SETTINGS.strategy) or SETTINGS.patgroup_musts:
        #     self.vector_strings = {p : "".join([str(self.vector[ag]) for ag in (PARAMS.antigens.keys()) if PARAMS.patgroup_weights.loc[p,ag] == 10]) for p in SETTINGS.patgroups}
        # else:
        #     self.vector_string = "".join([str(self.vector[ag]) for ag in (PARAMS.antigens.keys()) if PARAMS.relimm_weights[ag][0] == 10])

        # Retrieve the major blood group name from the phenotype vector.
        self.major = vector_to_major(vector)
        self.R0 = 1 if list(self.vector[2:7]) == [1,0,1,0,1] else 0

        # Only used for inventory products.
        self.ethnicity = ethnicity
        self.age = age
        self.index = index

        # Only used for requests.
        self.patgroup = patgroup            # patient group
        self.antibodies = antibodies        # vector containing a 1 for antigens that the patient has developed antibodies against, 0 for all other antigens
        self.mism_units = mism_units        # total number of mismatched units recieved over all transfusions so far
        self.num_units = num_units          # number of requested units
        self.day_issuing = day_issuing      # day that the patient is transfused
        self.day_available = day_available  # day that the requests becomes known
        self.allocated_from_dc = 0          # number of products allocated to this request from the distribution center's inventory
        

    # Transform the binary antigen vector to a blood group index.
    def vector_to_bloodgroup_index(self):
        return int("".join(str(i) for i in self.vector),2)

    # Get the usability of the blood's phenotype with respect to the distribution of either a given set of antigens, or of the major blood types, in the patient population.
    def get_usability(self, PARAMS, hospitals, antigens = []):

        # TODO this is now hardcoded for the case where SCD patients are Africans and all others are Caucasions.
        avg_daily_demand_african = sum([PARAMS.patgroup_distr[hospital.htype][1] * hospital.avg_daily_demand for hospital in hospitals])
        avg_daily_demand_total = sum([hospital.avg_daily_demand for hospital in hospitals])
        part_african = avg_daily_demand_african / avg_daily_demand_total

        if antigens == []:
            usability_ABO = 0
            usability_RhD = 1

            # Calculate the ABO-usability of this blood product, by summing all prevalences of the phenotypes that can receive this product.
            ABO_v = self.vector[:2] 
            ABO_g = PARAMS.ABO_phenotypes
            for g in range(len(ABO_g)):
                if all(v <= g for v, g in zip(ABO_v, ABO_g[g])):
                    usability_ABO += PARAMS.ABO_prevalences[1][g] * part_african
                    usability_ABO += PARAMS.ABO_prevalences[0][g] * (1 - part_african)

            # Calculate the RhD-usability of this blood product, by summing all prevalences of the phenotypes that can receive this product.
            # If the considered blood product is RhD negative, usability is always 1. Therefore usability is only calculated when the product is RhD positive.
            Dpos = np.array([g[0] for g in PARAMS.Rhesus_phenotypes])
            Dpos_prevalence = sum(np.array(PARAMS.Rhesus_prevalences[1]) * part_african * Dpos) + sum(np.array(PARAMS.Rhesus_prevalences[0]) * (1 - part_african) * Dpos)
            if self.vector[2] == 1:
                usability_RhD = Dpos_prevalence

            # Return the product of all the individual system usabilities to compute the final usabilty.
            return usability_ABO * usability_RhD

        else:
            # Get intersection of all antigens given to consider, and all antigens in the model.
            usability_ABO = self.get_usability_system([0,1], antigens, PARAMS.ABO_phenotypes, PARAMS.ABO_prevalences, part_african)
            usability_Rhesus = self.get_usability_system([2,3,4,5,6], antigens, PARAMS.Rhesus_phenotypes, PARAMS.Rhesus_prevalences, part_african)
            usability_Kell = self.get_usability_system([7,8], antigens, PARAMS.Kell_phenotypes, PARAMS.Kell_prevalences, part_african)
            usability_Duffy = self.get_usability_system([9,10], antigens, PARAMS.Duffy_phenotypes, PARAMS.Duffy_prevalences, part_african)
            usability_Kidd = self.get_usability_system([11,12], antigens, PARAMS.Kidd_phenotypes, PARAMS.Kidd_prevalences, part_african)
            usability_MNS = self.get_usability_system([13,14,15,16], antigens, PARAMS.MNS_phenotypes, PARAMS.MNS_prevalences, part_african)

        # Return the product of all the individual system usabilities to compute the final usabilty.
        return usability_ABO * usability_Rhesus * usability_Kell * usability_MNS * usability_Duffy * usability_Kidd


    def get_usability_system(self, system_antigens, antigens, phenotypes, prevalences, part_african):

        usability = 0

        # Calculate the ABO-usability of this blood product, by summing all prevalences of the phenotypes that can receive this product.
        vector = [self.vector[i] for i in system_antigens]
        for g in range(len(phenotypes)):
            if all(v <= g for v, g in zip(vector, phenotypes[g])):
                usability += prevalences[1][g] * part_african
                usability += prevalences[0][g] * (1 - part_african)

        return usability
            

# Obtain the major blood group from a blood antigen vector.
def vector_to_major(vector):

    major = ""

    if vector[0] == 1:
        major += "A"
    if vector[1] == 1:
        major += "B"
    if len(major) == 0:
        major += "O"
    if vector[2] == 1:
        major += "+"
    else:
        major += "-"

    return major


# For each inventory product i∈I and request r∈R, T[i,r] = 1 if product i 
# will not yet be outdated by the time request r needs to be issued.
def timewise_possible(SETTINGS, PARAMS, I, R, day):
    
    T = np.zeros([len(I), len(R)])
    for i in range(len(I)):
        for r in range(len(R)):
            T[i,r] = 1 if (PARAMS.max_age - 1 - I[i].age) >= (R[r].day_issuing - day) else 0
    return T

# For each inventory product i∈I and request r∈R, C[i,r] = 1 if 
# i and r are compatible on the major and mandatory antigens.
def precompute_compatibility(SETTINGS, PARAMS, R, Iv, Rv, Rb):

    A = range(len(PARAMS.antigens))

    # If a patient has antibodies against a certain antigen, inventory products 
    # that are positive for that antigen are not considered compatible
    C = 1 - (Iv @ Rb.T)

    if ("patgroups" in SETTINGS.strategy) or SETTINGS.patgroup_musts:
        for r in range(len(R)):
            for i in range(len(Iv)):
                v_musts_ir = [(Iv[i,k], Rv[r,k]) for k in A if PARAMS.patgroup_weights[R[r].patgroup,k] == 10]
                if all(vi <= vr for vi, vr in v_musts_ir):
                    C[i,r] = 1

    else:
        for i in range(len(Iv)):
            for r in range(len(Rv)):
                if all(vi <= vr for vi, vr in zip(Iv[i,:3], Rv[r,:3])):
                    C[i,r] = 1

    return C