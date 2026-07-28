import shutil
import os
import os.path
import small_model
import figure2_generation as f2g
import torch
import numpy as np
import random
from PIL import Image
import data
from collections import defaultdict

COPY_FOLDER = "Copy_Transition_Cache"
COPY_NAME_N = "copy_n_state_min100_max151_sd42_bin_size0.5.npz"
COPY_NAME_B = "copy_b_state_min100_max151_sd42_bin_size0.5.npz"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

n_path_copy = os.path.join(COPY_FOLDER, COPY_NAME_N)
b_path_copy = os.path.join(COPY_FOLDER, COPY_NAME_B)
os.makedirs(COPY_FOLDER, exist_ok=True)
# make a copy of the neural state transition .npz 
if not os.path.exists(n_path_copy):
    shutil.copy("RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.5.npz", n_path_copy)
if not os.path.exists(b_path_copy):
    shutil.copy("RNN_cache/b_state/b_state_min100_max151_sd42_bin_size0.5.npz", b_path_copy)

def inner_dict():
    return defaultdict(int)
def outer_dict():
    return defaultdict(inner_dict)

def run_pertubation(model, total_num_pertubations, sd=42):
    '''Run pertubation such that one node is randomly knockout hooked at each step
        A minimum of 1000 total pertubations must take place
        Each node must be pertubed between 50 to a 100 times such that the sum of each node's pertubation count is 1000
        Update observed transitions to the copied transition dicts'''
    pertub_counter = 0
    neuron_visit_count_list = np.zeros(model.hidden_size, dtype=int)
    max_pertubs = total_num_pertubations 
    np.random.seed(sd)
    random.seed(sd)
    is_first_visit = True

    # load and open the saved dicts for writing
    with np.load(n_path_copy, allow_pickle=True) as data_n:
        loaded_n = data_n["neural_state_dict"].item()

    with np.load(b_path_copy, allow_pickle=True) as data_b:
        loaded_b = data_b["b_transition_dict"].item()

    updated_n_transition_dict = defaultdict(lambda:defaultdict(int))
    updated_b_transition_dict = defaultdict(lambda: defaultdict(int))

    for cur_state, next_states in loaded_n.items():
        updated_n_transition_dict[cur_state] = defaultdict(int, next_states)

    for cur_state, next_states in loaded_b.items():
            updated_b_transition_dict[cur_state] = defaultdict(int, next_states)


    b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count
    all_b_states = list(all_visit_count_dict.keys())
    starting_point = all_b_states[np.random.randint(0, len(all_b_states))] # randomly select a behavioral state as a starting point

    with torch.no_grad():
        while pertub_counter < max_pertubs:
            neurons_availble_for_pertub = [index for index, count in enumerate(neuron_visit_count_list) if count < 100]
            if not neurons_availble_for_pertub:
                break

            gen_rand_neuron_index = random.choice(neurons_availble_for_pertub)
           

            if is_first_visit:
                cur_b_state = starting_point
                cur_b_state_img_paths = b_state_img_path_dict[cur_b_state]
                cur_b_state_img_path = cur_b_state_img_paths[np.random.randint(0, len(cur_b_state_img_paths))]
                cur_b_state_img = Image.open(cur_b_state_img_path).convert("L")
                cur_b_state_img = cur_b_state_img.resize((25,25))
                cur_b_state_img_array = np.array(cur_b_state_img) / 255.0
                cur_b_state_img_tensor = torch.tensor(cur_b_state_img_array, dtype=torch.float32, device=device)
                h = None
                cur_neural_state, h = model(cur_b_state_img_tensor, h)
            
        
                # find the next behavioral state and the neural state associated with it
                next_b_state = f2g.gaussian_sample_next_state(all_b_states, cur_b_state)
                next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                next_b_state_img = next_b_state_img.resize((25,25))
                next_b_state_img_array = np.array(next_b_state_img) / 255.0
                next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)

                handle = model.knockout_neuron(gen_rand_neuron_index)
                try:
                    next_neural_state, h = model(next_b_state_img_tensor, h)
                finally:
                    handle.remove()

                cur_neural_state_key = f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.5)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.5)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz, needs to be fixed
                updated_n_transition_dict[cur_neural_state_key][next_neural_state_key] += 1
                updated_b_transition_dict[cur_b_state_key][next_b_state_key] += 1

                # update the visit and pertub count lists
                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1
                
                is_first_visit = False
            else:
                cur_b_state = next_b_state
                cur_neural_state = next_neural_state
                
                next_b_state = f2g.gaussian_sample_next_state(all_b_states, cur_b_state)
                next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                next_b_state_img = next_b_state_img.resize((25,25))
                next_b_state_img_array = np.array(next_b_state_img) / 255.0
                next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)

                handle = model.knockout_neuron(gen_rand_neuron_index)
                try:
                    next_neural_state, h = model(next_b_state_img_tensor, h)
                finally:
                    handle.remove()
                                    
                cur_neural_state_key =  f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.5)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.5)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz, needs to be fixed
                updated_n_transition_dict[cur_neural_state_key][next_neural_state_key] += 1
                updated_b_transition_dict[cur_b_state_key][next_b_state_key] += 1
                
                # update the visit and pertub count lists
                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1

    save_n = {state: dict(next_states) for state, next_states in updated_n_transition_dict.items()}

    save_b = {state: dict(next_states) for state, next_states in updated_b_transition_dict.items()}

    np.savez_compressed(n_path_copy, neural_state_dict=np.array(save_n, dtype=object))

    np.savez_compressed(b_path_copy, b_transition_dict=np.array(save_b, dtype=object))

    return updated_b_transition_dict, updated_n_transition_dict
                
if __name__ == "__main__":
    model = small_model.RNN().to(device)

    checkpoint = torch.load(
        "post_stage_2_model.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint)
    model.eval()

    new_b_t_dict, new_n_t_dict = run_pertubation(model, total_num_pertubations=1000, sd=42)

    # gen figures for comparison using methods from figure2_generation.py
    
