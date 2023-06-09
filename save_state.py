import os
import pickle

from hospital import *

def save_state(SETTINGS, logs, e, day, dc, hospitals):

	path = SETTINGS.home_dir + f"wip/{SETTINGS.model_name}/{e}/{SETTINGS.strategy}_{'-'.join([str(SETTINGS.n_hospitals[ds]) + ds for ds in SETTINGS.n_hospitals.keys()])}"	

	# df.to_csv(path + "_df.csv", sep=',', index=True)
	with open(path + "_logs.pickle", 'wb') as f:
		pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)

	# df_matches.to_csv(path + "_matches.csv", sep=',', index=True)
	dc.pickle(path + "_dc")
	for h in range(len(hospitals)):
		hospitals[h].pickle(path + f"_h{h}")


def load_state(SETTINGS, PARAMS, e, logs, dc, hospitals):

	path = SETTINGS.home_dir + f"wip/{SETTINGS.model_name}/{e}/{SETTINGS.strategy}_{'-'.join([str(SETTINGS.n_hospitals[ds]) + ds for ds in SETTINGS.n_hospitals.keys()])}"	

	if os.path.exists(path + "_logs.pickle") == True:

		# df = pd.read_csv(path + "_df.csv")
		# day = max(df[df["logged"]==True]["day"]) + 1
		# df = df.set_index(["day", "location"])
		logs = unpickle(path + "_logs")

		dc = unpickle(path + "_dc")

		hospitals = []
		for h in range(sum(SETTINGS.n_hospitals.values())):
			hospitals.append(unpickle(path + f"_h{h}"))
		
		# htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])
		# hospital = Hospital(SETTINGS, PARAMS, htype, e)
		# hospital.inventory += dc.sample_supply_single_day(PARAMS, hospital.inventory_size, 0)
		# hospitals = [hospital]

		day = np.sum(logs[:,SETTINGS.column_indices["logged"]]) / len(hospitals)

	else:
		day = 0

	return logs, day, dc, hospitals


def unpickle(path):
	with open(path+".pickle", 'rb') as f:
		return pickle.load(f)