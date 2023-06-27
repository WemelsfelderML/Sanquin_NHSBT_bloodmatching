import os
import pickle

from hospital import *

def save_state(SETTINGS, path, logs, e, day, dc, hospitals):

	with open(path + "logs.pickle", 'wb') as f:
		pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)

	dc.pickle(path + "dc")
	for h in range(len(hospitals)):
		hospitals[h].pickle(path + f"h{h}")


def load_state(SETTINGS, PARAMS, path, e, logs, dc, hospitals):

	if os.path.exists(path + "logs.pickle") == True:

		logs = unpickle(path + "logs")
		dc = unpickle(path + "dc")

		hospitals = []
		for h in range(sum(SETTINGS.n_hospitals.values())):
			hospitals.append(unpickle(path + f"h{h}"))
		
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