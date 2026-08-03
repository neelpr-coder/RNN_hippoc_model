import connected_models as cm
import figure2_generation as f2g
import manifold_visualization as mv
import perturbation_testing as pt
import pearson_correlation_confusion_matrix as pccm
import os, os.path
import torch
import numpy as np
import random
from collections import defaultdict
import data
from PIL import Image
from tqdm import tqdm

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.environ.get(
    "Connected_Cache",
    os.path.join(SCRIPT_DIR, "Connected_Cache")
)
os.makedirs(CACHE_DIR, exist_ok=True)

bin_size = 0.3

def generate_dicts(net, min_visits = 100, max_visits = 151, sd = 42):
    """After image preprocessing, each tuple in the list contain image path, the associated behavioral state (x,y,heading), 
    is_valid_location, and num_visits.
    Unpack the tuple, load the image, train the model
    Create a hashmap of each behavioral state and the neural states that led to it
    Model moves randomly even in training and must visit each behavioral state at least 100 times but no more than 150 times inclusive"""
    
    Na_transition_dir = os.path.join(CACHE_DIR, "Na_transition")
    Nb_transition_dir = os.path.join(CACHE_DIR, "Nb_transition")
    b_state_dir = os.path.join(CACHE_DIR, "b_state")
    Na_Nb_transition_dir = os.path.join(CACHE_DIR, "Na_Nb_transition")

    Na_b_transition_dir = os.path.join(CACHE_DIR, "Na_b_transition")
    Nb_b_transition_dir = os.path.join(CACHE_DIR, "Nb_b_transition")

    Na_Nb_b_transition_dir = os.path.join(CACHE_DIR, "Na_Nb_b_transition")

    all_visit_b_count_dir = os.path.join(CACHE_DIR, "all_visit_b_count")

    os.makedirs(b_state_dir, exist_ok=True)
    os.makedirs(all_visit_b_count_dir, exist_ok=True)
    os.makedirs(Na_transition_dir, exist_ok=True)
    os.makedirs(Nb_transition_dir, exist_ok=True)
    os.makedirs(Na_b_transition_dir, exist_ok=True)
    os.makedirs(Nb_b_transition_dir, exist_ok=True)
    os.makedirs(Na_Nb_transition_dir, exist_ok=True)
    os.makedirs(Na_Nb_b_transition_dir, exist_ok=True)

    b_cache_path = os.path.join(
        b_state_dir,
        f"b_state_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    all_visit_b_count_path = os.path.join(
            all_visit_b_count_dir,
            f"all_visit_b_count_dict_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    Na_cache_path = os.path.join(
        Na_transition_dir,
        f"Na_transition_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    Nb_cache_path = os.path.join(
        Nb_transition_dir,
        f"Nb_transition_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    Na_Nb_cache_path = os.path.join(
        Na_Nb_transition_dir,
        f"Na_Nb_transition_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    Na_b_cache_path = os.path.join(
        Na_b_transition_dir,
        f"all_visit_count_n_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    Nb_b_cache_path = os.path.join(
            Nb_b_transition_dir,
            f"Nb_b_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
        )

    Na_Nb_b_cache_path = os.path.join(
            Na_Nb_b_transition_dir,
            f"Na_Nb_b_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
        )
   
    if (
        os.path.exists(b_cache_path)
        and os.path.exists(all_visit_b_count_path)
        and os.path.exists(Na_cache_path)
        and os.path.exists(Nb_cache_path)
        and os.path.exists(Na_b_cache_path)
        and os.path.exists(Nb_b_cache_path)
        and os.path.exists(Na_Nb_cache_path)
        and os.path.exists(Na_Nb_b_cache_path)
        
    ):
        loaded_b = np.load(b_cache_path, allow_pickle=True)
        loaded_all_visit_b_count = np.load(all_visit_b_count_path, allow_pickle=True)
        loaded_Na = np.load(Na_cache_path, allow_pickle=True)
        loaded_Nb = np.load(Nb_cache_path, allow_pickle=True)
        loaded_Na_Nb = np.load(Na_Nb_cache_path, allow_pickle=True)
        loaded_Na_b = np.load(Na_b_cache_path, allow_pickle=True)
        loaded_Nb_b = np.load(Nb_b_cache_path, allow_pickle=True)
        loaded_Na_Nb_b = np.load(Na_Nb_b_cache_path, allow_pickle=True)
        
        loaded_b_transition_dict = loaded_b["b_transition_dict"].item()
        loaded_all_visit_b_count_dict = loaded_all_visit_b_count["all_visit_b_count_dict"].item()
        loaded_Na_transition_dict = loaded_Na["Na_transition_dict"].item()
        loaded_Nb_transition_dict = loaded_Nb["Nb_transition_dict"].item()
        loaded_Na_Nb_transition_dict = loaded_Na_Nb["Na_Nb_transition_dict"].item()
        loaded_Na_b_transition_dict = loaded_Na_b["Na_b_transition_dict"].item()
        loaded_Nb_b_transition_dict = loaded_Nb_b["Nb_b_transition_dict"].item()
        loaded_Na_Nb_b_transition_dict = loaded_Na_Nb_b["Na_Nb_b_transition_dict"].item()

        
        b_transition_dict = defaultdict(lambda: defaultdict(int))
        all_visit_b_count_dict = defaultdict(int)
        Na_transition_dict = defaultdict(lambda: defaultdict(int))
        Nb_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_Nb_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_b_transition_dict = defaultdict(lambda: defaultdict(int))
        Nb_b_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_Nb_b_transition_dict = defaultdict(lambda: defaultdict(int))

        for state, freq_dict in loaded_b_transition_dict.items():
            b_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq in loaded_all_visit_b_count_dict.items():
            all_visit_b_count_dict[state] = freq        

        for state, freq_dict in loaded_Na_transition_dict.items():
            Na_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_Nb_transition_dict.items():
            Nb_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_Na_Nb_transition_dict.items():
            Na_Nb_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_Na_b_transition_dict.items():
            Na_b_transition_dict[state] = defaultdict(int, freq_dict)
        
        for state, freq_dict in loaded_Nb_b_transition_dict.items():
            Nb_b_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_Na_Nb_b_transition_dict.items():
            Na_Nb_b_transition_dict[state] = defaultdict(int, freq_dict)

        print(f"[Log] cache already exists for min_visits={min_visits}")
        return b_transition_dict, all_visit_b_count_dict, Na_transition_dict, Nb_transition_dict, Na_Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_b_transition_dict
    else:
        print("[Log] creating the cache...")
    
        b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count
        np.random.seed(sd)
        all_b_states = list(all_visit_count_dict.keys())
        np.random.seed(sd)
        random.seed(sd)
        starting_point = all_b_states[np.random.randint(0, len(all_b_states))] # randomly select a behavioral state as a starting point
        
        b_transition_dict = defaultdict(lambda: defaultdict(int))
        all_visit_b_count_dict = defaultdict(int, all_visit_count_dict)
        Na_transition_dict = defaultdict(lambda: defaultdict(int))
        Nb_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_Nb_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_b_transition_dict = defaultdict(lambda: defaultdict(int))
        Nb_b_transition_dict = defaultdict(lambda: defaultdict(int))
        Na_Nb_b_transition_dict = defaultdict(lambda: defaultdict(int))
        
        done = min(all_visit_b_count_dict.values()) >= min_visits
        is_first_visit = True
        model = net
        model.eval()

        total_targets = len(all_b_states) * min_visits
        current_progress = sum(min(v, min_visits) for v in all_visit_b_count_dict.values())
        progress_bar = tqdm(total=total_targets, initial=current_progress, desc="Generating the table")

        with torch.no_grad():
            h_a = torch.zeros(1, model.hidden_size1, device=device)
            h_b = torch.zeros(1, model.hidden_size2, device=device)
            while not done:
                available_states = [s for s in all_b_states if all_visit_b_count_dict[s] < max_visits] # want the list of available states to keep updating each iteration
                if not available_states:
                    progress_bar.close()
                    raise ValueError("no available states")
                
                if is_first_visit:
                    # convert starting point to tensor and find the neural state associated with it
                    cur_b_state = starting_point
                    cur_b_state_img_paths = b_state_img_path_dict[cur_b_state]
                    cur_b_state_img_path = cur_b_state_img_paths[np.random.randint(0, len(cur_b_state_img_paths))]
                    cur_b_state_img = Image.open(cur_b_state_img_path).convert("L")
                    cur_b_state_img = cur_b_state_img.resize((25,25))
                    cur_b_state_img_array = np.array(cur_b_state_img) / 255.0
                    cur_b_state_img_tensor = torch.tensor(cur_b_state_img_array, dtype=torch.float32, device=device)
                    cur_neural_state_a, cur_neural_state_b, cur_I_aa, cur_I_ab, cur_I_bb, cur_I_ba = model(cur_b_state_img_tensor, cur_b_state_img_tensor, h_a, h_b)
                    h_a = cur_neural_state_a
                    h_b = cur_neural_state_b

                    # find the next behavioral state and the neural state associated with it
                    next_b_state = f2g.gaussian_sample_next_state(available_states, cur_b_state)
                    next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                    next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                    next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                    next_b_state_img = next_b_state_img.resize((25,25))
                    next_b_state_img_array = np.array(next_b_state_img) / 255.0
                    next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)
                    next_neural_state_a, next_neural_state_b, next_I_aa, next_I_ab, next_I_bb, next_I_ba = model(next_b_state_img_tensor, next_b_state_img_tensor, h_a, h_b)
                    h_a = next_neural_state_a
                    h_b = next_neural_state_b
                    cur_count = all_visit_b_count_dict[cur_b_state]
                    all_visit_b_count_dict[cur_b_state] += 1

                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_a_key = f2g.neural_state_to_dict_key(cur_neural_state_a.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    cur_neural_state_b_key = f2g.neural_state_to_dict_key(cur_neural_state_b.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                    next_neural_state_a_key = f2g.neural_state_to_dict_key(next_neural_state_a.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    next_neural_state_b_key = f2g.neural_state_to_dict_key(next_neural_state_b.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                    # update the other dictionaries
                    b_transition_dict[cur_b_state_key][next_b_state_key] += 1
                    Na_transition_dict[cur_neural_state_a_key][next_neural_state_a_key] += 1
                    Nb_transition_dict[cur_neural_state_b_key][next_neural_state_b_key] += 1
                    Na_b_transition_dict[(cur_neural_state_a_key, cur_b_state_key)][(next_neural_state_a_key, next_b_state_key)] += 1
                    Nb_b_transition_dict[(cur_neural_state_b_key, cur_b_state_key)][(next_neural_state_b_key, next_b_state_key)] += 1
                    Na_Nb_transition_dict[(cur_neural_state_a_key, cur_neural_state_b_key)][(next_neural_state_a_key, next_neural_state_b_key)] += 1
                    Na_Nb_b_transition_dict[(cur_neural_state_a_key, cur_neural_state_b_key, cur_b_state_key)][(next_neural_state_a_key, next_neural_state_b_key, next_b_state_key)] += 1
                    is_first_visit = False
                else: 
                    cur_b_state = next_b_state
                    cur_neural_state_a, cur_neural_state_b = next_neural_state_a, next_neural_state_b

                    next_b_state = f2g.gaussian_sample_next_state(available_states, cur_b_state)
                    next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                    next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                    next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                    next_b_state_img = next_b_state_img.resize((25,25))
                    next_b_state_img_array = np.array(next_b_state_img) / 255.0
                    next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)
                    next_neural_state_a, next_neural_state_b, next_I_aa, next_I_ab, next_I_bb, next_I_ba = model(next_b_state_img_tensor, next_b_state_img_tensor, h_a, h_b)
                    h_a = next_neural_state_a
                    h_b = next_neural_state_b

                    cur_count = all_visit_b_count_dict[cur_b_state]
                    all_visit_b_count_dict[cur_b_state] += 1

                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_a_key = f2g.neural_state_to_dict_key(cur_neural_state_a.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    cur_neural_state_b_key = f2g.neural_state_to_dict_key(cur_neural_state_b.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                    next_neural_state_a_key = f2g.neural_state_to_dict_key(next_neural_state_a.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    next_neural_state_b_key = f2g.neural_state_to_dict_key(next_neural_state_b.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                    next_b_state_key = f2g.behavioral_state_to_key(next_b_state)
                    
                    # update the other dictionaries
                    b_transition_dict[cur_b_state_key][next_b_state_key] += 1
                    Na_transition_dict[cur_neural_state_a_key][next_neural_state_a_key] += 1
                    Nb_transition_dict[cur_neural_state_b_key][next_neural_state_b_key] += 1
                    Na_b_transition_dict[(cur_neural_state_a_key, cur_b_state_key)][(next_neural_state_a_key, next_b_state_key)] += 1
                    Nb_b_transition_dict[(cur_neural_state_b_key, cur_b_state_key)][(next_neural_state_b_key, next_b_state_key)] += 1
                    Na_Nb_transition_dict[(cur_neural_state_a_key, cur_neural_state_b_key)][(next_neural_state_a_key, next_neural_state_b_key)] += 1
                    Na_Nb_b_transition_dict[(cur_neural_state_a_key, cur_neural_state_b_key, cur_b_state_key)][(next_neural_state_a_key, next_neural_state_b_key, next_b_state_key)] += 1

                done = min(all_visit_b_count_dict.values()) >= min_visits

        progress_bar.close()

        save_b_transition = {k: dict(v) for k, v in b_transition_dict.items()}
        save_Na = {k: dict(v) for k, v in Na_transition_dict.items()}
        save_Nb = {k: dict(v) for k, v in Nb_transition_dict.items()}
        save_all_visit_b_count = dict(all_visit_b_count_dict)
        save_Na_Nb = {k: dict(value) for k, value in Na_Nb_transition_dict.items()}
        save_Na_b = {k: dict(v) for k, v in Na_b_transition_dict.items()}
        save_Nb_b = {k: dict(v) for k, v in Nb_b_transition_dict.items()}
        save_Na_Nb_b = {k: dict(v) for k, v in Na_Nb_b_transition_dict.items()}
        
        np.savez_compressed(
            all_visit_b_count_path,
            all_visit_b_count_dict=np.array(save_all_visit_b_count, dtype=object)
        )
        np.savez_compressed(
            b_cache_path,
            b_transition_dict=np.array(save_b_transition, dtype=object)
        )
        np.savez_compressed(
            Na_cache_path,
            Na_transition_dict=np.array(save_Na, dtype=object)
        )
        np.savez_compressed(
            Nb_cache_path,
            Nb_transition_dict=np.array(save_Nb, dtype=object)
        )
        np.savez_compressed(
            Na_b_cache_path,
            Na_b_transition_dict=np.array(save_Na_b, dtype=object)
        )

        np.savez_compressed(
            Nb_b_cache_path,
            Nb_b_transition_dict=np.array(save_Nb_b, dtype=object)
        )

        np.savez_compressed(
            Na_Nb_cache_path,
            Na_Nb_transition_dict=np.array(save_Na_Nb, dtype=object)
        )

        np.savez_compressed(
            Na_Nb_b_cache_path,
            Na_Nb_b_transition_dict=np.array(save_Na_Nb_b, dtype=object)
        )

        return b_transition_dict, all_visit_b_count_dict, Na_transition_dict, Nb_transition_dict, Na_Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_b_transition_dict

    
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = cm.connected_models().to(device)
    model.eval()
    CONNECTED_MODEL_PATH = os.path.join(SCRIPT_DIR, "post_stage1_connected_model_sd42.pt")
    torch.save(model.state_dict(), CONNECTED_MODEL_PATH)
    b_transition_dict, all_visit_b_count_dict, Na_transition_dict, Nb_transition_dict, Na_Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_b_transition_dict = generate_dicts(model)
    