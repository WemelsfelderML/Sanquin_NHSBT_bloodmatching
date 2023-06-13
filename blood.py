import random
import numpy as np

class Blood:

    # An instance of this class is created for both patient requests and inventory products, to contain their phenotype.
    def __init__(self, PARAMS, index = None, ethnicity = None, patgroup = None, antigens = [], num_units = 0, day_issuing = 0, day_available = 0, age = 0, antibodies = [], mism_units = []):

        # Store antigen profile of the blood product or patient request
        self.vector = antigens

        # Retrieve the major blood group name from the phenotype vector.
        self.major = vector_to_major(antigens)
        self.R0 = 1 if list(antigens[2:7]) == [1,0,1,0,1] else 0

        # Only used for inventory products.
        # self.ethnicity = ethnicity
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
        avg_daily_demand_african = sum([PARAMS.weekly_demand[hospital.htype][1] / 7 for hospital in hospitals])
        avg_daily_demand_total = sum([sum(PARAMS.weekly_demand[hospital.htype]) / 7 for hospital in hospitals])
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
    C = np.zeros([len(Iv), len(R)])

    if ("patgroups" in SETTINGS.strategy) or SETTINGS.patgroup_musts:
        for r in range(len(R)):
            mask = PARAMS.patgroup_weights[R[r].patgroup, :] == 10
            v_musts_ir = np.stack((Iv[:, mask], np.repeat(Rv[r, mask][np.newaxis, :], len(Iv), axis=0)))
            C[:, r] = np.where(np.all(v_musts_ir[0, :] <= v_musts_ir[1, :], axis=1), 1, 0)

    else:
        for i in range(len(Iv)):
            for r in range(len(Rv)):
                if np.all(Iv[i, :3] <= Rv[r, :3]):
                    C[i, r] = 1


    # If a patient has antibodies against a certain antigen, inventory products 
    # that are positive for that antigen are not considered compatible
    C[np.any(Iv[:, np.newaxis] * Rb == 1, axis=2)] = 0

    return C