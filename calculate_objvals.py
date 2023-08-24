
def get_total_antibodies(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):
    
    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[18:35]
                        rq_index = f"{e}_{rq[52]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return len(list(chain.from_iterable(antibodies_per_patient.values())))


def get_patients_with_antibodies(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[18:35]
                        rq_index = f"{e}_{rq[52]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return len(antibodies_per_patient.keys())


def get_max_antibodies_per_patients(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens.keys()
    antibodies_per_patient = defaultdict(set)
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="patients", scenario=scenario, name=htype+f"_{e}", day=day)).astype(int)
                    for rq in data:
                        rq_antibodies = rq[18:35]
                        rq_index = f"{e}_{rq[52]}"
                        antibodies_per_patient[rq_index].update(k for k in antigens if rq_antibodies[k] > 0)

    return max([len(antibodies) for antibodies in antibodies_per_patient.values()])


def get_shortages(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    shortages = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                    data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                    shortages += data["num shortages"].sum()

    return shortages


def get_outdates(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    outdates = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                    data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                    outdates += data["num outdates"].sum()

    return outdates


def get_total_alloimmunization_risk(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    antigens = PARAMS.antigens
    total_alloimm_risk = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            P = [pg for p, pg in PARAMS.patgroups.items() if PARAMS.weekly_demand[htype][p] > 0]
            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                    data = pd.read_csv(SETTINGS.generate_filename(method=method, output_type="results", scenario=scenario, name=scenario_name, e=e)+".csv")

                    for k in antigens.keys():
                        total_alloimm_risk += PARAMS.alloimmunization_risks[2,k] * data[[f"num mismatched patients {pg} {antigens[k]}" for pg in P]].sum().sum()
                        
    return total_alloimm_risk


def get_issued_products_nonoptimal_age_SCD(SETTINGS, PARAMS, method, scenario, episode_start=0, num_init_points=1, p=0):

    total_products = 0
    nonoptimal_age = 0
    for r in range(episode_start, episode_start + SETTINGS.replications):
        for htype in SETTINGS.n_hospitals.keys():

            n = SETTINGS.n_hospitals[htype]
            for i in range(n):
                e = (((r * num_init_points) + p) * n) + i
                for day in range(SETTINGS.init_days, SETTINGS.init_days + SETTINGS.test_days):

                    scenario_name = '-'.join([str(SETTINGS.n_hospitals[htype]) + htype for htype in SETTINGS.n_hospitals.keys() if SETTINGS.n_hospitals[htype]>0])
                    data = unpickle(SETTINGS.generate_filename(method=method, output_type="results", subtype="issuing_age", scenario=scenario, name=scenario_name, e=e))

                    total_products += data[1,:].sum()
                    nonoptimal_age += data[1,:7].sum() + data[1,11:].sum()

    return nonoptimal_age / total_products