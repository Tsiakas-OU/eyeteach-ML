import pandas as pd
import numpy as np
import json
import os

PRE_PROCESS_DATA_FOLDER = "pre_processed_data"
TEXT_DATA_FOLDER = os.path.join("experiment_data", "set_texts")
FIXATION_DATA_FOLDER = os.path.join("pre_processed_data","fixation_data_per_part")
data_list = []

for file in os.listdir(PRE_PROCESS_DATA_FOLDER):
    if ".json" not in file:
        continue
    with open(os.path.join(PRE_PROCESS_DATA_FOLDER, file), "r") as f:
        load_data = json.load(f)
    data_list.append(pd.Series(list(load_data.values()), index=load_data.keys()))
data = pd.DataFrame(data_list)

print(data.head())

# Filters to apply:
approved_only = (data.approved_flag > 0).to_numpy() 
no_fixation_error = (data.fixation_error == False).to_numpy()
no_target_error = (data.target_error == False).to_numpy()
sample_higher_10 = (data.webgazer_sample_rate > 10).to_numpy()
acc_higher = (data.avg_roi_last_val > 0).to_numpy()
filter_mturks = np.array([False if "link" in worker_id else True for worker_id in data["worker_id"]])
filter_sets = np.array([True if set_lang in ["EN","ES","DE"] else False for set_lang in data["set_language"]])

screen_x_above_1280 = (data.screen_x > 1110).to_numpy() # Some tolerance
screen_y_above_720 = (data.screen_y > 615).to_numpy() # Some Tolerance
screen_above_1280_720 = screen_x_above_1280 & screen_y_above_720
dict_filter = {
    "filter_mturks" : filter_mturks,
    "filter_sets" : filter_sets,
    "Approved":approved_only,
    "Fix_Error, Target_Error": no_fixation_error & no_target_error,
    "Sample Rate": sample_higher_10,
    "acc_thresh": acc_higher,
    "screen_above_1280_720": screen_above_1280_720,
}
n_total = len(data)
current_filter = np.ones(len(data),dtype=bool)
for condition, f in dict_filter.items():
    n_data_filtered = len(data.iloc[~f & current_filter])
    per_cent = n_data_filtered/n_total * 100
    print(f"For condition ({condition}), {per_cent:.2f}% has been filtered. ({n_data_filtered} out of {n_total})")
    current_filter = current_filter & f
    n_total = len(data.iloc[current_filter])

mask = filter_mturks & approved_only & no_fixation_error & no_target_error & sample_higher_10 & screen_above_1280_720 & acc_higher & filter_sets
data_filtered = data[mask].copy()

# Sets collected (each will have different texts)
data_filtered.set_name.unique()
print(data_filtered.head())

# Selecting a single participant
participant_EN_v01_n1 = data_filtered[data_filtered.set_name == "mturk_EN_v01"].iloc[0]

print(participant_EN_v01_n1.head)
example_of_data_load = json.loads(participant_EN_v01_n1["webgazer_filtered_data"])
print(example_of_data_load)

example_of_data_load = json.loads(participant_EN_v01_n1["webgazer_raw_data"])
print(example_of_data_load)