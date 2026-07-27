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

# make a copy of the neural state transition .npz 
if not os.path.exists(os.path.join(COPY_FOLDER, COPY_NAME_N)) or not os.path.join(COPY_FOLDER, COPY_NAME_B):
    os.makedirs(COPY_FOLDER, exist_ok=True)
    shutil.copy("RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.5.npz", os.path.join(COPY_FOLDER, COPY_NAME_N))
    shutil.copy("RNN_cache/b_state/b_state_min100_max151_sd42_bin_size0.5.npz", os.path.join(COPY_FOLDER, COPY_NAME_B))

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
    is_first_visit = True
    with torch.no_grad():
        while pertub_counter <= max_pertubs:
            neurons_availble_for_pertub = [index for index, count in enumerate(neuron_visit_count_list) if count <= 100]
            gen_rand_neuron_index = random.choice(neurons_availble_for_pertub)
            while neurons_availble_for_pertub[gen_rand_neuron_index] == 100:
                gen_rand_neuron_index = random.randint(0,9)
            handle = model.knockout_neuron(gen_rand_neuron_index)
           
            b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count

            all_b_states = list(all_visit_count_dict.keys())
            starting_point = all_b_states[np.random.randint(0, len(all_b_states))] # randomly select a behavioral state as a starting point

            if is_first_visit:
                h = None
                cur_b_state = starting_point
                cur_b_state_img_paths = b_state_img_path_dict[cur_b_state]
                cur_b_state_img_path = cur_b_state_img_paths[np.random.randint(0, len(cur_b_state_img_paths))]
                cur_b_state_img = Image.open(cur_b_state_img_path).convert("L")
                cur_b_state_img = cur_b_state_img.resize((25,25))
                cur_b_state_img_array = np.array(cur_b_state_img) / 255.0
                cur_b_state_img_tensor = torch.tensor(cur_b_state_img_array, dtype=torch.float32, device=device)
                cur_neural_state, h = model(cur_b_state_img_tensor, h)
        
                # find the next behavioral state and the neural state associated with it
                next_b_state = f2g.gaussian_sample_next_state(all_b_states, cur_b_state)
                next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                next_b_state_img = next_b_state_img.resize((25,25))
                next_b_state_img_array = np.array(next_b_state_img) / 255.0
                next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)
                next_neural_state, h = model(next_b_state_img_tensor, h)

                cur_neural_state_key = f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.5)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.5)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz, needs to be fixed
                with np.load(os.path.join(COPY_FOLDER, COPY_NAME_N), allow_pickle=True) as data_n:
                    archived_transition_n = dict(data_n)

                if cur_neural_state_key in archived_transition_n:
                    raw_dict_n = archived_transition_n[cur_neural_state_key].item()
                    current_defaultdict_n = outer_dict()
                    for outer_key, inner_dict in raw_dict_n.items():
                        current_defaultdict_n[outer_key].update(inner_dict)
                else:
                    current_defaultdict_n = outer_dict()

                current_defaultdict_n[cur_neural_state_key][next_neural_state_key] += 1
                serializable_dict = {k: dict(v) for k, v in current_defaultdict_n.items()}

                archived_transition_n[cur_neural_state_key] = np.array(serializable_dict, dtype=object)
                np.savez(os.path.join(COPY_FOLDER, COPY_NAME_N), **archived_transition_n)

                # update the copied b transition .npz, needs to be fixed
                with np.load(os.path.join(COPY_FOLDER, COPY_NAME_B), allow_pickle=True) as data_b:
                    archived_transition_b = dict(data_b)
                
                if cur_b_state_key in archived_transition_b:
                    raw_dict_b = archived_transition_b[cur_b_state_key].item()
                    current_defaultdict_b = outer_dict()
                    for outer_key, inner_dict in raw_dict_b.items():
                        current_defaultdict_b[outer_key].update(inner_dict)
                else:
                    current_defaultdict_b = outer_dict()
                
                current_defaultdict_b[cur_b_state_key][next_b_state_key] += 1
                serializable_dict_b = {k: dict(v) for k, v in current_defaultdict_b.items()}
                
                archived_transition_b[cur_b_state_key] = np.array(serializable_dict_b, dtype=object)
                np.savez(os.path.join(COPY_FOLDER, COPY_NAME_B), **archived_transition_b)

                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1
                gen_rand_neuron_index = random.randint(0,9)
                is_first_visit = False
                handle.remove()
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
                next_neural_state, h = model(next_b_state_img_tensor, h)
                                    
                cur_neural_state_key =  f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.5)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.5)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz
                with np.load(os.path.join(COPY_FOLDER, COPY_NAME_N), allow_pickle=True) as data_n:
                    archived_transition_n = dict(data_n)

                if cur_neural_state_key in archived_transition_n:
                    raw_dict_n = archived_transition_n[cur_neural_state_key].item()
                    current_defaultdict_n = outer_dict()
                    for outer_key, inner_dict in raw_dict_n.items():
                        current_defaultdict_n[outer_key].update(inner_dict)
                else:
                    current_defaultdict_n = outer_dict()

                current_defaultdict_n[cur_neural_state_key][next_neural_state_key] += 1
                serializable_dict = {k: dict(v) for k, v in current_defaultdict_n.items()}

                archived_transition_n[cur_neural_state_key] = np.array(serializable_dict, dtype=object)
                np.savez(os.path.join(COPY_FOLDER, COPY_NAME_B), **archived_transition_n)

                # update the copied b transition .npz
                with np.load(os.path.join(COPY_FOLDER, COPY_NAME_B), allow_pickle=True) as data_b:
                    archived_transition_b = dict(data_b)
                
                if cur_b_state_key in archived_transition_n:
                    raw_dict_b = archived_transition_n[cur_b_state_key].item()
                    current_defaultdict_b = outer_dict()
                    for outer_key, inner_dict in raw_dict_b.items():
                        current_defaultdict_b[outer_key].update(inner_dict)
                else:
                    current_defaultdict_b = outer_dict()
                
                current_defaultdict_b[cur_b_state_key][next_b_state_key] += 1
                serializable_dict_b = {k: dict(v) for k, v in current_defaultdict_b.items()}
                
                archived_transition_b[cur_b_state_key] = np.array(serializable_dict_b, dtype=object)
                np.savez(os.path.join(COPY_FOLDER, COPY_NAME_B), **archived_transition_b)

                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1
                gen_rand_neuron_index = random.randint(0,9)
                handle.remove()


if __name__ == "__main__":
    model = small_model.RNN().to(device)

    checkpoint = torch.load(
        "your_trained_model.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint)
    model.eval()

    run_pertubation(model, total_num_pertubations=1000, seed=42)
