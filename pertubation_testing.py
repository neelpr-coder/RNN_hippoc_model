import shutil
import os
import os.path
import small_model
import figure2_generation as f2g
import torch
import numpy as np

COPY_FOLDER = "Copy_Transition_Cache"
COPY_NAME_N = "copy_n_state_min100_max151_sd42_bin_size0.5.npz"
COPY_NAME_B = "copy_b_state_min100_max151_sd42_bin_size0.5.npz"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# make a copy of the neural state transition .npz 
if not os.path.exists(COPY_NAME_N) and not os.path.exists(COPY_NAME_B):
    os.makedirs(COPY_FOLDER, exist_ok=True)
    shutil.copy("RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.5.npz", os.path.join(COPY_FOLDER, COPY_NAME_N))
    shutil.copy("RNN_cache/b_state/b_state_min100_max151_sd42_bin_size0.5.npz", os.path.join(COPY_FOLDER, COPY_NAME_B))

def run_pertubation(model, total_num_pertubations, seed=42):
    '''Run pertubation such that one node is randomly knockout hooked at each step
        A minimum of 1000 total pertubations must take place
        Each node must be pertubed between 50 to a 100 times such that the sum of each node's pertubation count is 1000
        Update observed transitions to the copied transition dicts'''
    pertub_counter = 0
    neuron_visit_count_list = np.zeros(10)
    hl_size = model.hidden_size
    max_pertubs = total_num_pertubations * hl_size
    while not pertub_counter > max_pertubs:
        neurons_availble_for_pertub = [n for n in neuron_visit_count_list if neuron_visit_count_list[n] <=100]

if __name__ == "__main__":
    model = small_model.RNN().to(device) # load the RNN
