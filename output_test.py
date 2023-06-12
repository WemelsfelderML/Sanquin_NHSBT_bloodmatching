HOME_DIR = "/home/mw922/Sanquin_NHSBT_bloodmatching/"
episode = 0
htype = "UCLH"
init_days = 2 * 35
test_days = 14

def unpickle(path):
    with open(path+".pickle", 'rb') as f:
        return pickle.load(f)

for day in range(init_days + test_days):
    print(unpickle(HOME_DIR + f"{log_type}/{folders[0]}/{episode}/patients_patgroups_{htype}/{day}"))