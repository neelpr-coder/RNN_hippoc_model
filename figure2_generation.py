import torch 
import numpy as np
import os
import json
import random
import ast
import math

from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

import data
import small_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.environ.get(
    "RNN_CACHE_DIR",
    os.path.join(SCRIPT_DIR, "RNN_cache")
)

os.makedirs(CACHE_DIR, exist_ok=True)

#bin_size = 0.5

def neural_state_to_dict_key(neural_state, bin_size):
    return tuple(round(float(x) / bin_size) * bin_size for x in neural_state)

def behavioral_state_to_key(b_state):
    x, y, h = b_state
    return (int(x), int(y), int(h))
        
def gaussian_sample_next_state(available_states, current_state, sigma = 2.0, max_attempts = 150): 
    mu_x, mu_y, mu_h = current_state
    available_states_set = set(available_states)
    available_states_list = list(available_states)

    if not available_states_list:
        raise ValueError("No available states remain")

    for _ in range(max_attempts):
        sampled_x = int(round(np.random.normal(mu_x, sigma)))
        sampled_y = int(round(np.random.normal(mu_y, sigma)))
        sampled_x = np.clip(sampled_x, 0, 9)
        sampled_y = np.clip(sampled_y, 0, 9)
        sampled_h = np.random.randint(0, 4)

        sampled_state = (sampled_x, sampled_y, sampled_h)
        
        if sampled_state in available_states_set:
            return sampled_state

    return random.choice(available_states_list)

def generate_dicts(net, min_visits = 100, max_visits = 151, sd = 42):
    """After image preprocessing, each tuple in the list contain image path, the associated behavioral state (x,y,heading), 
    is_valid_location, and num_visits.
    Unpack the tuple, load the image, train the model
    Create a hashmap of each behavioral state and the neural states that led to it
    Model moves randomly even in training and must visit each behavioral state at least 100 times but no more than 150 times inclusive"""
    
    pair_dir = os.path.join(CACHE_DIR, "pair_transition")
    b_state_dir = os.path.join(CACHE_DIR, "b_state")
    n_state_dir = os.path.join(CACHE_DIR, "n_state")
    all_visit_count_b_dir = os.path.join(CACHE_DIR, "all_visit_count_b")
    all_visit_count_n_dir = os.path.join(CACHE_DIR, "all_visit_count_n")
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(b_state_dir, exist_ok=True)
    os.makedirs(n_state_dir, exist_ok=True)
    os.makedirs(all_visit_count_b_dir, exist_ok=True)
    os.makedirs(all_visit_count_n_dir, exist_ok=True)

    pair_cache_path = os.path.join(
        pair_dir,
        f"pair_transition_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    b_cache_path = os.path.join(
        b_state_dir,
        f"b_state_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    n_cache_path = os.path.join(
        n_state_dir,
        f"n_state_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    all_visit_count_b_cache_path = os.path.join(
        all_visit_count_b_dir,
        f"all_visit_count_b_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

    all_visit_count_n_cache_path = os.path.join(
        all_visit_count_n_dir,
        f"all_visit_count_n_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz"
    )

   
    if (
        os.path.exists(pair_cache_path)
        and os.path.exists(b_cache_path)
        and os.path.exists(n_cache_path)
        and os.path.exists(all_visit_count_b_cache_path)
        and os.path.exists(all_visit_count_n_cache_path)
        
    ):
        loaded_pair = np.load(pair_cache_path, allow_pickle=True)
        loaded_b = np.load(b_cache_path, allow_pickle=True)
        loaded_n = np.load(n_cache_path, allow_pickle=True)
        loaded_all_visit_count_b = np.load(all_visit_count_b_cache_path, allow_pickle=True)
        loaded_all_visit_count_n = np.load(all_visit_count_n_cache_path, allow_pickle=True)
        
        loaded_pair_transition_dict = loaded_pair["pair_dict"].item()
        loaded_b_transition_dict = loaded_b["b_transition_dict"].item()
        loaded_neural_state_dict = loaded_n["neural_state_dict"].item()
        loaded_all_visit_count_b_dict = loaded_all_visit_count_b["all_visit_count_b_dict"].item()
        loaded_all_visit_count_n_dict = loaded_all_visit_count_n["all_visit_count_n_dict"].item()
        

        pair_transition_dict = defaultdict(lambda: defaultdict(int))
        b_transition_dict = defaultdict(lambda: defaultdict(int))
        neural_state_dict = defaultdict(lambda: defaultdict(int))
        all_visit_count_b_dict = defaultdict(int)
        all_visit_count_n_dict = defaultdict(int)

        for state, freq_dict in loaded_pair_transition_dict.items():
            pair_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_b_transition_dict.items():
            b_transition_dict[state] = defaultdict(int, freq_dict)

        for state, freq_dict in loaded_neural_state_dict.items():
            neural_state_dict[state] = defaultdict(int, freq_dict)

        for state, freq in loaded_all_visit_count_b_dict.items():
            all_visit_count_b_dict[state] = freq

        for state, freq in loaded_all_visit_count_n_dict.items():
            all_visit_count_n_dict[state] = freq
        

        print(f"[Log] cache already exists for min_visits={min_visits}")
        return pair_transition_dict, b_transition_dict, neural_state_dict, all_visit_count_b_dict, all_visit_count_n_dict
    else:
        print("[Log] creating the cache...")
    
        b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count
        np.random.seed(sd)
        all_b_states = list(all_visit_count_dict.keys())
        starting_point = all_b_states[np.random.randint(0, len(all_b_states))] # randomly select a behavioral state as a starting point
        
        pair_transition_dict = defaultdict(lambda: defaultdict(int))
        behavioral_transition_dict = defaultdict(lambda: defaultdict(int))
        neural_state_dict = defaultdict(lambda: defaultdict(int))
        all_visit_count_b_dict = defaultdict(int, all_visit_count_dict) # initialize the behavioral state visit count dict with the counts from preprocessing
        all_visit_count_n_dict = defaultdict(int)
        
        done = min(all_visit_count_b_dict.values()) >= min_visits
        is_first_visit = True
        model = net
        model.eval()

        total_targets = len(all_b_states) * min_visits
        current_progress = sum(min(v, min_visits) for v in all_visit_count_dict.values())
        progress_bar = tqdm(total=total_targets, initial=current_progress, desc="Generating the table")

        with torch.no_grad():
            h = None
            while not done:
                available_states = [s for s in all_b_states if all_visit_count_b_dict[s] < max_visits] # want the list of available states to keep updating each iteration
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
                    cur_neural_state, h = model(cur_b_state_img_tensor, h)

                    # find the next behavioral state and the neural state associated with it
                    next_b_state = gaussian_sample_next_state(available_states, cur_b_state)
                    next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                    next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                    next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                    next_b_state_img = next_b_state_img.resize((25,25))
                    next_b_state_img_array = np.array(next_b_state_img) / 255.0
                    next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)
                    next_neural_state, h = model(next_b_state_img_tensor, h)

                    cur_count = all_visit_count_b_dict[cur_b_state]
                    all_visit_count_b_dict[cur_b_state] += 1


                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_key = neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.3)
                    cur_b_state_key = behavioral_state_to_key(cur_b_state)
                    next_neural_state_key = neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.3)
                    next_b_state_key = behavioral_state_to_key(next_b_state)

                    all_visit_count_n_dict[cur_neural_state_key] += 1

                    # update the other dictionaries
                    behavioral_transition_dict[cur_b_state_key][next_b_state_key] += 1
                    neural_state_dict[cur_neural_state_key][next_neural_state_key] += 1
                    pair_transition_dict[(cur_b_state_key, cur_neural_state_key)][(next_b_state_key, next_neural_state_key)] += 1
                    
                    is_first_visit = False
                else: 
                    cur_b_state = next_b_state
                    cur_neural_state = next_neural_state

                    next_b_state = gaussian_sample_next_state(available_states, cur_b_state)
                    next_b_state_img_paths = b_state_img_path_dict[next_b_state]
                    next_b_state_img_path = next_b_state_img_paths[np.random.randint(0, len(next_b_state_img_paths))]
                    next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                    next_b_state_img = next_b_state_img.resize((25,25))
                    next_b_state_img_array = np.array(next_b_state_img) / 255.0
                    next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)
                    next_neural_state, h = model(next_b_state_img_tensor, h)

                    cur_count = all_visit_count_b_dict[cur_b_state]
                    all_visit_count_b_dict[cur_b_state] += 1

                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_key =  neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.3)
                    cur_b_state_key = behavioral_state_to_key(cur_b_state)

                    next_neural_state_key = neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.3)
                    next_b_state_key = behavioral_state_to_key(next_b_state)
                    
                    all_visit_count_n_dict[cur_neural_state_key] += 1

                    # update the other dictionaries
                    behavioral_transition_dict[cur_b_state_key][next_b_state_key] += 1
                    neural_state_dict[cur_neural_state_key][next_neural_state_key] += 1
                    pair_transition_dict[(cur_b_state_key, cur_neural_state_key)][(next_b_state_key, next_neural_state_key)] += 1

                done = min(all_visit_count_b_dict.values()) >= min_visits

        progress_bar.close()
        save_pair = {k: dict(v) for k, v in pair_transition_dict.items()}
        save_b_transition = {k: dict(v) for k, v in behavioral_transition_dict.items()}
        save_neural_state = {k: dict(v) for k, v in neural_state_dict.items()}
        save_all_visit_count_b_dict = dict(all_visit_count_b_dict)
        save_all_visit_count_n_dict = dict(all_visit_count_n_dict)

        np.savez_compressed(
            pair_cache_path,
            pair_dict=np.array(save_pair, dtype=object)
        )
        np.savez_compressed(
            b_cache_path,
            b_transition_dict=np.array(save_b_transition, dtype=object)
        )
        np.savez_compressed(
            n_cache_path,
            neural_state_dict=np.array(save_neural_state, dtype=object)
        )
        np.savez_compressed(
            all_visit_count_b_cache_path,
            all_visit_count_b_dict=np.array(save_all_visit_count_b_dict, dtype=object)
        )
        np.savez_compressed(
            all_visit_count_n_cache_path,
            all_visit_count_n_dict=np.array(save_all_visit_count_n_dict, dtype=object)
        )


        return pair_transition_dict, behavioral_transition_dict, neural_state_dict, all_visit_count_b_dict, all_visit_count_n_dict

def json_b_to_n_state(raw_dict, purpose="count"):
    """json conversion method to make the behavioral state to neural state, frequency dict easily readable"""
    out = {}
    sorted_raw_dict = sorted(raw_dict.items(), key=lambda item: item[0])
    if purpose == "count":
        for b_state, freq_dict in sorted_raw_dict:
            out[str(b_state)] = {
                str(neural_state): int(count)
                for neural_state, count in freq_dict.items()
            }
    elif purpose == "probability":
        for b_state, freq_dict in sorted_raw_dict:
            out[str(b_state)] = {
                str(neural_state): float(count)
                for neural_state, count in freq_dict.items()
            }
    return out

def convert_count_to_probability(freq_dict):  
    converted_dict = defaultdict(lambda: defaultdict(float))     
    for cur_key, next_keys in freq_dict.items():
        total_visits = sum(next_keys.values())
        for next_key, count in next_keys.items():
            probability = count / total_visits if total_visits > 0 else 0
            converted_dict[cur_key][next_key] = probability
    return converted_dict

def one_step_probability(b_trans_dict, n_state_dict, pair_dict):
    """Normalize count dictionaries into 1-step probabilities."""
    if b_trans_dict is None or n_state_dict is None or pair_dict is None:
        raise ValueError("Missing required dictionaries")

    freq_b_trans_dict = convert_count_to_probability(b_trans_dict)
    freq_n_trans_dict = convert_count_to_probability(n_state_dict)
    freq_pair_trans_dict = convert_count_to_probability(pair_dict)

    return freq_b_trans_dict, freq_n_trans_dict, freq_pair_trans_dict

def propagate_one_step(current_dist, one_step_pair_prob, tol=1e-12):
    """
    Push a probability distribution forward by one step.

    current_dist:
        dict: state -> probability mass
    one_step_pair_prob:
        dict: cur_state -> {next_state: P(next_state | cur_state)}
    """
    next_dist = defaultdict(float)

    for cur_state, cur_prob in current_dist.items():
        if cur_prob <= tol:
            continue

        if cur_state not in one_step_pair_prob:
            continue

        for next_state, trans_prob in one_step_pair_prob[cur_state].items():
            mass = cur_prob * trans_prob
            if mass > tol:
                next_dist[next_state] += mass

    return dict(next_dist)

def n_step_from_start(start_state, one_step_pair_prob, num_steps, tol=1e-12):
    """
    Compute the n-step transition distribution starting from one state.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    current_dist = {start_state: 1.0}

    for _ in range(num_steps):
        current_dist = propagate_one_step(current_dist, one_step_pair_prob, tol=tol)
        if not current_dist:
            break

    return current_dist

def n_step_sparse_probability(num_steps=1, b_trans_dict=None, n_state_dict=None, pair_dict=None, tol=1e-12):
    """
    Compute sparse n-step probabilities without building a full matrix.
    """
    if b_trans_dict is None or n_state_dict is None or pair_dict is None:
        raise ValueError("Missing required dictionaries")

    freq_b = convert_count_to_probability(b_trans_dict)
    freq_n = convert_count_to_probability(n_state_dict)
    freq_pair = convert_count_to_probability(pair_dict)

    n_step_dict = defaultdict(lambda: defaultdict(float))

    for start_state in freq_pair.keys():
        end_dist = n_step_from_start(start_state, freq_pair, num_steps, tol=tol)
        for end_state, prob in end_dist.items():
            if prob > tol:
                n_step_dict[start_state][end_state] = prob

    return freq_b, freq_n, freq_pair, n_step_dict
    
def compute_avg_b_to_n_degeneracy(pair_states=None):
    b_to_n = defaultdict(set)
    for b, n in pair_states:
        b_to_n[b].add(n)
    return sum(len(v) for v in b_to_n.values()) / len(b_to_n)

def compute_avg_n_to_b_degeneracy(pair_states=None):
    n_to_b = defaultdict(set)
    for b, n in pair_states:
        n_to_b[n].add(b)
    return sum(len(v) for v in n_to_b.values()) / len(n_to_b)

def permute_b_to_n_states(setting=None, pair_states=None):
    b_states = [b for b, _ in pair_states]
    n_states = [n for _, n in pair_states]

    if setting == 'b_to_n':
        random.shuffle(n_states)
        return list(zip(b_states, n_states))
    elif setting == 'n_to_b':
        random.shuffle(b_states)
        return list(zip(b_states, n_states))
    else:
        raise ValueError("Invalid setting. Use 'b_to_n' to permute neural states or 'n_to_b' to permute behavioral states.")

def test_degeneracy(pair_transition_dict, num_trials=100000, alpha_level = 0.05):
    """Perform a permutation test to see if the observed degeneracy is significantly higher than what would be expected by chance.
    
    Null hypothesis: The observed degeneracy is not significantly higher than what would be expected by chance.
    Alternative hypothesis: The observed degeneracy is significantly higher than what would be expected by chance.

    1. Compute the observed average degeneracy from the actual pair states.
    2. Generate a null distribution of average degeneracy by randomly permuting the neural states while keeping the behavioral states fixed, and computing the average degeneracy for each permutation.
    3. Calculate the p-value as the proportion of permuted average degeneracy values that are less than or equal to the observed average degeneracy.
    4. If the p-value is less than the chosen significance level (e.g., 0.05), reject the null hypothesis and conclude that the observed degeneracy is significantly lower than what would be expected by chance."""
    
    pair_states = [key for key in pair_transition_dict.keys()]
    observed_b_to_n_degeneracy = compute_avg_b_to_n_degeneracy(pair_states)
    observed_n_to_b_degeneracy = compute_avg_n_to_b_degeneracy(pair_states)

    null_b_to_n_distribution = []
    null_n_to_b_distribution = []
    for _ in range(num_trials):
        permuted_b_to_n_pairs = permute_b_to_n_states(setting="b_to_n", pair_states=pair_states)
        permuted_n_to_b_pairs = permute_b_to_n_states(setting="n_to_b", pair_states=pair_states)
        
        null_b_to_n_distribution.append(compute_avg_b_to_n_degeneracy(permuted_b_to_n_pairs))
        null_n_to_b_distribution.append(compute_avg_n_to_b_degeneracy(permuted_n_to_b_pairs))
    
    print(f"Observed avg b to n degeneracy: {observed_b_to_n_degeneracy:.6f}")
    print(f"Null mean (b_to_n): {sum(null_b_to_n_distribution)/len(null_b_to_n_distribution):.6f}")
    print(f"Null min (b_to_n): {min(null_b_to_n_distribution):.6f}")
    print(f"Null max (b_to_n): {max(null_b_to_n_distribution):.6f}\n")
    print(f"Observed avg n to b degeneracy: {observed_n_to_b_degeneracy:.6f}")
    print(f"Null mean (n_to_b): {sum(null_n_to_b_distribution)/len(null_n_to_b_distribution):.6f}")
    print(f"Null min (n_to_b): {min(null_n_to_b_distribution):.6f}")
    print(f"Null max (n_to_b): {max(null_n_to_b_distribution):.6f}")

    extreme_count = sum(x >= observed_b_to_n_degeneracy for x in null_b_to_n_distribution)
    p_value_b_to_n = (extreme_count + 1) / (num_trials + 1)

    extreme_count_n_to_b = sum(x >= observed_n_to_b_degeneracy for x in null_n_to_b_distribution)
    p_value_n_to_b = (extreme_count_n_to_b + 1) / (num_trials + 1)

    z_b_to_n = (observed_b_to_n_degeneracy - np.mean(null_b_to_n_distribution)) / np.std(null_b_to_n_distribution)
    z_n_to_b = (observed_n_to_b_degeneracy - np.mean(null_n_to_b_distribution)) / np.std(null_n_to_b_distribution)

    print(f"Z-score for b to n degeneracy: {z_b_to_n:.6f}")
    print(f"Z-score for n to b degeneracy: {z_n_to_b:.6f}")

    if p_value_b_to_n <= alpha_level:
        print(f"Reject null hypothesis: observed b to n degeneracy is significantly higher than random (p = {p_value_b_to_n:.9f})")
    else: 
        print(f"Fail to reject null hypothesis: observed degeneracy is not significantly higher than random (p = {p_value_b_to_n:.9f})")

    if p_value_n_to_b <= alpha_level:
        print(f"Reject null hypothesis: observed n to b degeneracy is significantly higher than random (p = {p_value_n_to_b:.9f})")
    else: 
        print(f"Fail to reject null hypothesis: observed degeneracy is not significantly higher than random (p = {p_value_n_to_b:.9f})")

    return null_b_to_n_distribution, null_n_to_b_distribution, p_value_b_to_n, p_value_n_to_b, observed_b_to_n_degeneracy, observed_n_to_b_degeneracy

def build_b_to_n_map(pair_transition_dict):
    b_to_n_states_dict = defaultdict(set)
    for (b_state, n_state) in pair_transition_dict.keys():
        b_to_n_states_dict[b_state].add(n_state)

    return b_to_n_states_dict

def build_n_to_b_map(pair_transition_dict):
    n_to_b_states_dict = defaultdict(set)
    for (b_state, n_state) in pair_transition_dict.keys():
        n_to_b_states_dict[n_state].add(b_state)
    return n_to_b_states_dict

def l1_distance(state1, state2):
    keys = set(state1) | set(state2)
    return sum(abs(state1.get(k, 0) - state2.get(k, 0)) for k in keys)

def dynamic_degeneracy_probability(pair_transition_dict, num_steps=2, gen_JSON = False):
    freq_pair = convert_count_to_probability(pair_transition_dict)
    b_to_n = build_b_to_n_map(pair_transition_dict)
    n_to_b = build_n_to_b_map(pair_transition_dict)

    cache = {}
    def get_dist(state, freq_pair, num_steps):
        if state not in cache:
            cache[state] = n_step_from_start(state, freq_pair, num_steps=num_steps)
        return cache[state]
    
    def top_k_dist(dist, k=10):
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
        return {str(state): float(prob) for state, prob in items}

    scores_b_to_n = []
    scores_n_to_b = []

    out_b_to_n = {}
    out_n_to_b = {}

    print("[Log] Computing dynamic degeneracy probability b_to_n...")

    for b, n_set in b_to_n.items():
        if len(n_set) < 2:
            continue

        n_list = list(n_set)
        b_to_n_comparisons = []

        for i in range(len(n_list)):
            for j in range(i + 1, len(n_list)):
                n1, n2 = n_list[i], n_list[j]

                state1 = (b, n1)
                state2 = (b, n2)

                if state1 not in freq_pair or state2 not in freq_pair:
                    continue

                dist1 = get_dist(state1, freq_pair, num_steps=num_steps)
                dist2 = get_dist(state2, freq_pair, num_steps=num_steps)

                score = l1_distance(dist1, dist2)
                scores_b_to_n.append(score)

                if gen_JSON:
                    b_to_n_comparisons.append({
                        "state_1": str(state1),
                        "state_2": str(state2),
                        "dynamic_prob_dist_1":
                            top_k_dist(dist1, k=10)
                        ,
                        "dynamic_prob_dist_2":
                            top_k_dist(dist2, k=10)
                        
                    })

        if gen_JSON and b_to_n_comparisons:
            out_b_to_n[str(b)] = b_to_n_comparisons

    print("[Log] Completed dynamic degeneracy probability b_to_n computation.")
    print("[Log] Computing dynamic degeneracy probability n_to_b...")

    for n, b_set in n_to_b.items():
        if len(b_set) < 2:
            continue

        b_list = list(b_set)
        n_to_b_comparisons = []

        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                b1, b2 = b_list[i], b_list[j]

                state1 = (b1, n)
                state2 = (b2, n)

                if state1 not in freq_pair or state2 not in freq_pair:
                    continue

                dist1 = get_dist(state1, freq_pair, num_steps=num_steps)
                dist2 = get_dist(state2, freq_pair, num_steps=num_steps)

                score_n_to_b = l1_distance(dist1, dist2)
                scores_n_to_b.append(score_n_to_b)

                if gen_JSON:
                    n_to_b_comparisons.append({
                        "state_1": str(state1),
                        "state_2": str(state2),
                        "dynamic_prob_dist_1":
                            top_k_dist(dist1, k=10),
                        "dynamic_prob_dist_2":
                            top_k_dist(dist2, k=10)
                        
                        })
        
        if gen_JSON and n_to_b_comparisons:
            out_n_to_b[str(n)] = n_to_b_comparisons
    

    if gen_JSON:
        return out_b_to_n, out_n_to_b

    return scores_b_to_n, scores_n_to_b


def sweep_data_gen(net, step_size=5, min_attempts=50, max_attempts=101, sd=42):
    """Generate a sweep of dictionaries as we vary the min number of attempts."""
    results = {}

    for attempt in range(min_attempts, max_attempts, step_size):
        print(f"\n[Log] Generating data for min_attempts = {attempt}...")

        pair_transition_dict, b_transition_dict, n_state_dict, all_visit_count_b_dict, all_visit_count_n_dict = generate_dicts(
            net,
            min_visits=attempt,
            sd=sd
        )

        results[attempt] = {"pair_transition_dict": pair_transition_dict, "b_transition_dict": b_transition_dict, "n_state_dict": n_state_dict, "all_visit_count_b_dict": all_visit_count_b_dict, "all_visit_count_n_dict": all_visit_count_n_dict}

        print(f"[Log] Completed data generation for min_attempts = {attempt}.")

    return results

def b_state_distribution_barplots():
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    min_attempt_values = sorted(sweep_results.keys())

    plots_per_page = 2
    cols = 2
    rows = 1

    for page_start in range(0, len(min_attempt_values), plots_per_page):
        page_values = min_attempt_values[page_start:page_start + plots_per_page]

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5))
        axes = axes.flatten()

        for i, min_attempts in enumerate(page_values):
            ax = axes[i]

            b_state_dict = sweep_results[min_attempts]["all_visit_count_b_dict"]

            coord_visit_counts = defaultdict(int)

            for b_state, visit_count in b_state_dict.items():
                x_coord, y_coord, heading = b_state

                # Collapse across rotations/headings
                coord = (x_coord, y_coord, heading)
                coord_visit_counts[coord] += visit_count

            sorted_coords = sorted(coord_visit_counts.keys())
            counts = [coord_visit_counts[coord] for coord in sorted_coords]
            labels = [str(coord) for coord in sorted_coords]

            ax.bar(labels, counts, width=0.8)

            ax.set_title(f"min_attempts = {min_attempts}", fontsize=12)
            ax.set_xlabel("Behavioral coordinate", fontsize=10)
            ax.set_ylabel("Number of visits", fontsize=10)

            # y-axis increments of 50
            y_max = max(counts)
            ax.set_ylim(0, y_max + 25)
            ax.set_yticks(range(0, y_max + 51, 50))

            # show only every 10th neural state label
            tick_step = 10
            ax.set_xticks(range(0, len(labels), tick_step))
            ax.set_xticklabels(labels[::tick_step], rotation=60, ha="right", fontsize=6)
            ax.grid(axis="y", alpha=0.5)

        for j in range(len(page_values), len(axes)):
            axes[j].axis("off")


        fig.suptitle(
            f"Behavioral Coordinate Visit Counts Across min_attempts",
            fontsize=14
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

def top_k_n_state_distribution_barplots(top_k=100):
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    min_attempt_values = sorted(sweep_results.keys())

    plots_per_page = 2
    cols = 2
    rows = 1

    for page_start in range(0, len(min_attempt_values), plots_per_page):
        page_values = min_attempt_values[page_start:page_start + plots_per_page]

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5))
        axes = axes.flatten()

        for i, min_attempts in enumerate(page_values):
            ax = axes[i]

            n_state_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]

            sorted_counts = sorted(
                n_state_dict.values(),
                reverse=True
            )

            top_counts = sorted_counts[:top_k]
            labels = [f"N{rank + 1}" for rank in range(len(top_counts))]

            ax.bar(labels, top_counts, width=0.8)

            ax.set_title(f"Top {top_k} Neural State Visits, min_attempts = {min_attempts}")
            ax.set_xlabel("Neural state rank")
            ax.set_ylabel("Number of visits")

            y_max = max(top_counts)
            ax.set_ylim(0, y_max + 50)
            ax.set_yticks(range(0, y_max + 51, 50))

            tick_step = 5
            ax.set_xticks(range(0, len(labels), tick_step))
            ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)

            ax.grid(axis="y", alpha=0.5)

        for j in range(len(page_values), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Ranked Neural State Visit Counts Across min_attempts",
            fontsize=14
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

def bin_neural_states(counts, bin_size, reduce="sum"):
        binned_values = []
        bin_labels = []

        for start in range(0, len(counts), bin_size):
            end = min(start + bin_size, len(counts))
            bin_counts = counts[start:end]

            if reduce == "mean":
                value = np.mean(bin_counts)
            elif reduce == "sum":
                value = np.sum(bin_counts)
            else:
                raise ValueError("reduce must be 'mean' or 'sum'")

            binned_values.append(value)
            bin_labels.append(f"N{start + 1}-N{end}")

        return binned_values, bin_labels

def n_state_distribution_overlay_binned_barplots(top_k=100, bin_size=20, reduce="mean"):
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    min_attempt_values = sorted(sweep_results.keys())

    plt.figure(figsize=(12, 7))

    colors = plt.cm.tab20(np.linspace(0, 1, len(min_attempt_values)))

    num_sweeps = len(min_attempt_values)
    group_width = 0.85
    bar_width = group_width / num_sweeps

    for sweep_idx, (color, min_attempts) in enumerate(zip(colors, min_attempt_values)):
        n_state_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]

        sorted_counts = sorted(n_state_dict.values(), reverse=True)
        top_counts = sorted_counts[:top_k]

        binned_values, bin_labels = bin_neural_states(
            top_counts,
            bin_size=bin_size,
            reduce=reduce
        )

        base_positions = np.arange(len(bin_labels))

        offset = (sweep_idx - num_sweeps / 2) * bar_width + bar_width / 2
        x_positions = base_positions + offset

        plt.bar(
            x_positions,
            binned_values,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=f"min={min_attempts}"
        )

    plt.title(
        f"Binned Top {top_k} Neural State Visit Counts Across min_attempts\n"
        f"{bin_size} neural states per bin, reduce={reduce}"
    )
    plt.xlabel("Neural state rank bin")
    
    if reduce == "mean":
        plt.ylabel("Average visits per neural state")
    else:
        plt.ylabel("Total visits per rank bin")

    plt.xticks(
        range(len(bin_labels)),
        bin_labels,
        rotation=45,
        ha="right"
    )

    plt.grid(axis="y", alpha=0.5)
    plt.legend(title="min_attempts", fontsize=8)
    plt.tight_layout()
    plt.show()

def n_state_distribution_histograms_individual(bin_size=100):
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    min_attempt_values = sorted(sweep_results.keys())

    plots_per_page = 2
    cols = 2
    rows = 1

    for page_start in range(0, len(min_attempt_values), plots_per_page):
        page_values = min_attempt_values[page_start:page_start + plots_per_page]

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5))
        axes = axes.flatten()

        for i, min_attempts in enumerate(page_values):
            ax = axes[i]

            n_state_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]
            visit_counts = list(n_state_dict.values())

            max_count = max(visit_counts)
            upper = math.ceil(max_count / bin_size) * bin_size

            bins = list(range(0, upper + bin_size, bin_size))
            hist_counts, bin_edges = np.histogram(visit_counts, bins=bins)

            labels = []
            for j in range(len(bin_edges) - 1):
                start = int(bin_edges[j])
                end = int(bin_edges[j + 1])

                if start == 0:
                    labels.append(f"{start}-{end}")
                else:
                    labels.append(f"{start + 1}-{end}")

            ax.bar(labels, hist_counts, edgecolor="black", alpha=0.75)

            ax.set_title(f"min_attempts = {min_attempts}", fontsize=12)
            ax.set_xlabel("Visit-count bin")
            ax.set_ylabel("Number of neural states")

            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(axis="y", alpha=0.5)

        for j in range(len(page_values), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Neural State Visit Count Distribution",
            fontsize=14
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

def n_state_histogram_overlay_binned(
    bin_size=100,
    reduce_to_same_bins=True,
    max_display_bin=1000,
    log_y=True
):
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    min_attempt_values = sorted(sweep_results.keys())

    # Collect all visit counts for each sweep
    all_counts_by_min = {}
    for min_attempts in min_attempt_values:
        n_state_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]
        all_counts_by_min[min_attempts] = list(n_state_dict.values())

    # If max_display_bin is not provided, use rounded global max
    if max_display_bin is None:
        global_max_count = max(max(counts) for counts in all_counts_by_min.values())
        max_display_bin = math.ceil(global_max_count / bin_size) * bin_size

    # Build fixed bins up to max_display_bin
    bins = list(range(0, max_display_bin + bin_size, bin_size))

    # Labels for regular bins
    bin_labels = []
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]
        if start == 0:
            bin_labels.append(f"{start}-{end}")
        else:
            bin_labels.append(f"{start + 1}-{end}")

    # Add one overflow bin label
    bin_labels.append(f">{max_display_bin}")

    base_positions = np.arange(len(bin_labels))

    colors = plt.cm.tab20(np.linspace(0, 1, len(min_attempt_values)))

    num_sweeps = len(min_attempt_values)
    group_width = 0.85
    bar_width = group_width / num_sweeps

    plt.figure(figsize=(14, 7))

    for sweep_idx, (color, min_attempts) in enumerate(zip(colors, min_attempt_values)):
        visit_counts = np.array(all_counts_by_min[min_attempts])

        # Histogram counts for regular bins
        hist_counts, _ = np.histogram(visit_counts, bins=bins)

        # Overflow bin: counts greater than max_display_bin
        overflow_count = np.sum(visit_counts > max_display_bin)

        # Append overflow bin
        hist_counts = np.append(hist_counts, overflow_count)

        offset = (sweep_idx - num_sweeps / 2) * bar_width + bar_width / 2
        x_positions = base_positions + offset

        plt.bar(
            x_positions,
            hist_counts,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=f"min={min_attempts}"
        )

    plt.title(
        f"Neural State Visit Count Distribution Across min_attempts\n"
        f"Visit-count bin size = {bin_size}, tail grouped as >{max_display_bin}"
    )
    plt.xlabel("Visit-count bin")
    plt.ylabel("Number of neural states")

    plt.xticks(
        base_positions,
        bin_labels,
        rotation=45,
        ha="right"
    )

    plt.grid(axis="y", alpha=0.5)

    if log_y:
        plt.yscale("log")

    plt.legend(
        title="min_attempts",
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

def n_state_distribution_summary():
    sweep_results = sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=100,
        sd=42
    )

    for min_attempts in sorted(sweep_results.keys()):
        n_state_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]
        counts = sorted(n_state_dict.values(), reverse=True)

        total_visits = sum(counts)
        num_states = len(counts)

        top_10_frac = sum(counts[:10]) / total_visits
        top_50_frac = sum(counts[:50]) / total_visits
        top_100_frac = sum(counts[:100]) / total_visits

        print(f"\nmin_attempts = {min_attempts}")
        print(f"Number of unique neural states: {num_states}")
        print(f"Total neural visits: {total_visits}")
        print(f"Top 10 fraction: {top_10_frac:.3f}")
        print(f"Top 50 fraction: {top_50_frac:.3f}")
        print(f"Top 100 fraction: {top_100_frac:.3f}")
        print(f"Max visits: {max(counts)}")
        print(f"Median visits: {np.median(counts):.2f}")

def b_state_distribution_heatmap(all_visit_count_b_dict, agg="mean", show=True):
    coord_to_heading_counts = defaultdict(list)

    for b_state, visit_count in all_visit_count_b_dict.items():
        x, y, _ = b_state
        coord = (x, y)

        coord_to_heading_counts[coord].append(visit_count)

    x_vals = sorted({coord[0] for coord in coord_to_heading_counts.keys()})
    y_vals = sorted({coord[1] for coord in coord_to_heading_counts.keys()})

    x_to_idx = {x: i for i, x in enumerate(x_vals)}
    y_to_idx = {y: i for i, y in enumerate(y_vals)}

    grid = np.full((len(y_vals), len(x_vals)), np.nan)

    for coord, counts in coord_to_heading_counts.items():
        x, y = coord

        if agg == "mean":
            value = np.mean(counts)
        elif agg == "sum":
            value = np.sum(counts)
        elif agg == "max":
            value = np.max(counts)
        else:
            raise ValueError("agg must be 'mean', 'sum', or 'max'")

        grid[y_to_idx[y], x_to_idx[x]] = value

    if show:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(grid, origin="lower", aspect="auto")

        plt.title(f"Behavioral State Visit Count Heatmap, agg={agg}")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")

        plt.xticks(range(len(x_vals)), x_vals)
        plt.yticks(range(len(y_vals)), y_vals)

        plt.colorbar(im, label="Number of visits")
        plt.tight_layout()
        plt.show()
    
    return grid, x_vals, y_vals

def n_state_distribution_heatmap(b_to_n_states_dict, agg="union", show=True):
    coord_to_heading_counts = defaultdict(list)
    coord_to_n_union = defaultdict(set)

    for b_state, n_state_set in b_to_n_states_dict.items():
        x, y, _ = b_state
        coord = (x, y)

        coord_to_heading_counts[coord].append(len(n_state_set))
        coord_to_n_union[coord].update(n_state_set)

    x_vals = sorted({coord[0] for coord in coord_to_heading_counts.keys()})
    y_vals = sorted({coord[1] for coord in coord_to_heading_counts.keys()})

    x_to_idx = {x: i for i, x in enumerate(x_vals)}
    y_to_idx = {y: i for i, y in enumerate(y_vals)}

    grid = np.full((len(y_vals), len(x_vals)), np.nan)

    for coord, counts in coord_to_heading_counts.items():
        x, y = coord

        if agg == "union":
            value = len(coord_to_n_union[coord])
        elif agg == "mean":
            value = np.mean(counts)
        elif agg == "sum":
            value = np.sum(counts)
        elif agg == "max":
            value = np.max(counts)
        else:
            raise ValueError("agg must be 'union', 'mean', 'sum', or 'max'")

        grid[y_to_idx[y], x_to_idx[x]] = value

    if show:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(grid, origin="lower", aspect="auto")

        plt.title(f"Associated Neural States per Coordinate, agg={agg}")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")

        plt.xticks(range(len(x_vals)), x_vals)
        plt.yticks(range(len(y_vals)), y_vals)

        plt.colorbar(im, label="Number of associated neural states")
        plt.tight_layout()
        plt.show()

    return grid, x_vals, y_vals

def normalize_grid(grid):
    grid_min = np.nanmin(grid)
    grid_max = np.nanmax(grid)

    if grid_max == grid_min:
        return np.zeros_like(grid)

    return (grid - grid_min) / (grid_max - grid_min)

def joint_b_n_dist_heatmap(normalize=True, b_agg="mean", n_agg="union"):
    """Generate a heatmap showing the joint distribution of neural and behavioral states. Generate from the overlay
    of the two heatmaps."""
    b_grid, b_x_vals, b_y_vals = b_state_distribution_heatmap(all_visit_count_b_dict_100, agg="mean", show=False)
    n_grid, n_x_vals, n_y_vals = n_state_distribution_heatmap(b_to_n_dict, agg="union", show=False)
    
    if b_x_vals != n_x_vals or b_y_vals != n_y_vals:
        raise ValueError("Behavioral grid and neural grid coordinates do not match.")

    if normalize:
        b_used = normalize_grid(b_grid)
        n_used = normalize_grid(n_grid)
        joint_grid = b_used * n_used
        colorbar_label = "Normalized behavioral × neural score"
    else:
        joint_grid = b_grid * n_grid
        colorbar_label = "Behavioral visits × associated neural states"

    plt.figure(figsize=(8, 6))
    im = plt.imshow(joint_grid, origin="lower", aspect="auto")

    plt.title(
        f"Joint Behavioral-Neural Distribution\n"
        f"b_agg={b_agg}, n_agg={n_agg}, normalize={normalize}"
    )
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")

    plt.xticks(range(len(b_x_vals)), b_x_vals)
    plt.yticks(range(len(b_y_vals)), b_y_vals)

    plt.colorbar(im, label=colorbar_label)
    plt.tight_layout()
    plt.show()

    return joint_grid

def paired_transition_density_heatmap(
    pair_transition_dict,
    grid_size=10,
    mode="count",
    title="Paired transition density by position",
    cmap="viridis"
):
    """
    Creates a heatmap showing paired-transition density at each behavioral position.
    
    "count" = total outgoing transition counts from each (x, y)
    "unique_edges" = number of unique outgoing paired transitions from each (x, y)
    "unique_pair_states" = number of unique current (B, N) pair states at each (x, y)
    """

    heatmap = np.zeros((grid_size, grid_size), dtype=float)

    seen_pair_states_by_pos = {}

    for current_pair, next_dict in pair_transition_dict.items():
        B0, N0 = current_pair
        x, y, heading = B0

        if not (0 <= x < grid_size and 0 <= y < grid_size):
            continue

        if mode == "count":
            value = sum(next_dict.values())

        elif mode == "unique_edges":
            value = len(next_dict)

        elif mode == "unique_pair_states":
            seen_pair_states_by_pos.setdefault((x, y), set()).add(current_pair)
            continue

        else:
            raise ValueError("mode must be 'count', 'unique_edges', or 'unique_pair_states'")

        # heatmap indexed as [row=y, col=x]
        heatmap[y, x] += value

    if mode == "unique_pair_states":
        for (x, y), pair_states in seen_pair_states_by_pos.items():
            heatmap[y, x] = len(pair_states)


    plt.figure(figsize=(7, 6))

    im = plt.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        aspect="equal"
    )

    plt.title(title)
    plt.xlabel("x position")
    plt.ylabel("y position")

    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))

    cbar = plt.colorbar(im)

    if mode == "count":
        label = "Total paired transition count"
    elif mode == "unique_edges":
        label = "Number of unique outgoing paired transitions"
    else:
        label = "Number of unique current paired states"


    cbar.set_label(label)

    plt.tight_layout()
    plt.show()

    return heatmap

def make_n_state_labels(neural_state_dict):
    all_n_states = set()

    for n0, next_dict in neural_state_dict.items():
        all_n_states.add(n0)
        for n1 in next_dict.keys():
            all_n_states.add(n1)

    sorted_states = sorted(all_n_states, key=str)

    return {
        n_state: f"N{i + 1}"
        for i, n_state in enumerate(sorted_states)
    }



def get_top_k_items(next_dict, k=3):
    """
    Returns top-k transitions sorted by count descending.
    next_dict format:
        next_state -> count
    """
    return sorted(
        next_dict.items(),
        key=lambda item: item[1],
        reverse=True
    )[:k]

def transition_prob_from_counts(next_dict, next_state):
    """
    Returns probability, count, and total outgoing count for a transition.
    """
    total = sum(next_dict.values())

    if total == 0:
        return 0.0, 0, 0

    count = next_dict.get(next_state, 0)
    prob = count / total

    return prob, count, total

def top_choice_prob(next_dict):
    """
    Returns the highest transition probability from a current state.
    """
    total = sum(next_dict.values())

    if total == 0 or len(next_dict) == 0:
        return 0.0, None, 0, 0

    best_next, best_count = max(
        next_dict.items(),
        key=lambda item: item[1]
    )

    best_prob = best_count / total

    return best_prob, best_next, best_count, total

def route_prob_if_in_top3(next_dict, observed_next_state, top_k=3):
    """
    Version A:
    Use the observed transition probability only if observed_next_state
    appears in the top-k transitions. Otherwise return 0.
    """
    total = sum(next_dict.values())

    if total == 0:
        return 0.0, 0, 0, False, None

    top_items = get_top_k_items(next_dict, k=top_k)
    top_states = [state for state, count in top_items]

    observed_count = next_dict.get(observed_next_state, 0)
    observed_prob = observed_count / total if total > 0 else 0.0

    if observed_next_state in top_states:
        rank = top_states.index(observed_next_state) + 1
        return observed_prob, observed_count, total, True, rank

    return 0.0, observed_count, total, False, None

def compute_top3_route_choice_error(
    b_transition_dict,
    neural_state_dict,
    pair_transition_dict,
    B0,
    N0,
    B1,
    N1,
    top_k=3
):
    """
        raw_error = p_best - p_route

        C = total available nonzero transition options
          = len(B outgoing options) + len(N outgoing options) + len(pair outgoing options)

        normalized_error = raw_error / C
    """

    b_next_dict = b_transition_dict.get(B0, {})
    n_next_dict = neural_state_dict.get(N0, {})
    pair_next_dict = pair_transition_dict.get((B0, N0), {})

    if top_k is None:
    # No top-k filtering. Use the actual route transition probabilities directly.
        p_b_route, b_route_count, b_total = transition_prob_from_counts(
            b_next_dict,
            B1
        )

        p_n_route, n_route_count, n_total = transition_prob_from_counts(
            n_next_dict,
            N1
        )

        p_route = p_b_route * p_n_route

        b_in_top3 = None
        n_in_top3 = None
        b_rank_top3 = None
        n_rank_top3 = None

    else:
        # Existing top-k filtering logic
        p_b_route, b_route_count, b_total, b_in_top3, b_rank_top3 = route_prob_if_in_top3(
            b_next_dict, B1, top_k=top_k
        )

        p_n_route, n_route_count, n_total, n_in_top3, n_rank_top3 = route_prob_if_in_top3(
            n_next_dict, N1, top_k=top_k
        )

        if b_in_top3 and n_in_top3:
            p_route = p_b_route * p_n_route
        else:
            p_route = 0.0

    # Best independent transition
    p_b_best, b_best_state, b_best_count, _ = top_choice_prob(b_next_dict)
    p_n_best, n_best_state, n_best_count, _ = top_choice_prob(n_next_dict)

    p_best = p_b_best * p_n_best

    # Actual pair probability, still useful to display but no longer used in error
    actual_pair_prob, pair_count, pair_total = transition_prob_from_counts(
        pair_next_dict,
        (B1, N1)
    )

    # Normalization constant
    C = len(b_next_dict) + len(n_next_dict) + len(pair_next_dict)

    raw_error = p_best - p_route
    normalized_error = raw_error / C if C > 0 else 0.0

    actual_lookup_error = actual_pair_prob - p_best
    normalized_actual_lookup_error = actual_lookup_error / C if C > 0 else 0.0

    return {
        "p_route": p_route,
        "p_best": p_best,
        "raw_error": raw_error,
        "normalized_error": normalized_error,
        
        "actual_pair_prob": actual_pair_prob,
        "actual_lookup_error": actual_lookup_error,
        "normalized_actual_lookup_error": normalized_actual_lookup_error,

        "C": C,

        "p_b_route": p_b_route,
        "p_n_route": p_n_route,
        "p_b_best": p_b_best,
        "p_n_best": p_n_best,

        "b_in_top3": b_in_top3,
        "n_in_top3": n_in_top3,
        "b_rank_top3": b_rank_top3,
        "n_rank_top3": n_rank_top3,

        "b_route_count": b_route_count,
        "n_route_count": n_route_count,
        "b_total": b_total,
        "n_total": n_total,

        "b_best_state": b_best_state,
        "n_best_state": n_best_state,
        "b_best_count": b_best_count,
        "n_best_count": n_best_count,

        "actual_pair_prob": actual_pair_prob,
        "pair_count": pair_count,
        "pair_total": pair_total,

        "num_b_options": len(b_next_dict),
        "num_n_options": len(n_next_dict),
        "num_pair_options": len(pair_next_dict)
    }

def rank_of_transition(next_dict, target_key):
    ranked = sorted(next_dict.items(), key=lambda x: x[1], reverse=True)
    for idx, (key, _) in enumerate(ranked, start=1):
        if key == target_key:
            return idx
    return None

def generate_supported_observed_pair_route(
    pair_transition_dict,
    neural_state_dict,
    b_transition_dict,
    route_len=50,
    seed=42,
    min_pair_total=10,
    min_pair_count=1,
    min_n_total=10,
    min_b_total=20,
    max_tries=1000
):
    import random

    rng = random.Random(seed)

    valid_start_states = [
        pair_state for pair_state, next_dict in pair_transition_dict.items()
        if len(next_dict) > 0
    ]

    for _ in range(max_tries):
        current_pair = rng.choice(valid_start_states)
        route = []

        for _step in range(route_len):
            B0, N0 = current_pair

            pair_next_dict = pair_transition_dict.get(current_pair, {})
            n_next_dict = neural_state_dict.get(N0, {})
            b_next_dict = b_transition_dict.get(B0, {})

            pair_total = sum(pair_next_dict.values())
            n_total = sum(n_next_dict.values())
            b_total = sum(b_next_dict.values())

            candidate_next_pairs = []

            for next_pair, pair_count in pair_next_dict.items():
                B1, N1 = next_pair

                b_count = b_next_dict.get(B1, 0)
                n_count = n_next_dict.get(N1, 0)

                if (
                    pair_total >= min_pair_total and
                    pair_count >= min_pair_count and
                    n_total >= min_n_total and
                    b_total >= min_b_total and
                    b_count > 0 and
                    n_count > 0
                ):
                    candidate_next_pairs.append((next_pair, pair_count))

            if len(candidate_next_pairs) == 0:
                break

            next_pairs = [item[0] for item in candidate_next_pairs]
            weights = [item[1] for item in candidate_next_pairs]

            next_pair = rng.choices(next_pairs, weights=weights, k=1)[0]

            B1, N1 = next_pair
            observed_count = pair_next_dict[next_pair]

            route.append((B0, N0, B1, N1, observed_count))
            current_pair = next_pair

        if len(route) == route_len:
            return route

    raise ValueError(
        "Could not generate a full supported route. "
        "Try lowering thresholds or increasing bin size."
    )

def save_top3_route_transitions(
    route_sequence,
    b_transition_dict,
    neural_state_dict,
    pair_transition_dict,
    top_k=3
):
    """
    Save the exact top-k transitions shown in the animation for each frame as a list.
    """

    top3_history = []

    for frame_idx, (B0, N0, B1, N1, observed_count) in enumerate(route_sequence):
        b_count_nexts = b_transition_dict.get(B0, {})
        n_count_nexts = neural_state_dict.get(N0, {})
        pair_count_nexts = pair_transition_dict.get((B0, N0), {})

        top3_b = get_top_k_items(b_count_nexts, k=top_k)
        top3_n = get_top_k_items(n_count_nexts, k=top_k)
        top3_pair = get_top_k_items(pair_count_nexts, k=top_k)

        top3_history.append({
            "frame": frame_idx + 1,

            "B0": B0,
            "N0": N0,
            "B1": B1,
            "N1": N1,
            "observed_count": observed_count,

            # Exact top 3 lists shown in the animation text boxes
            "top3_b": top3_b,
            "top3_n": top3_n,
            "top3_pair": top3_pair
        })

    return top3_history

def animate_json_lookup_transition_clean(
    pair_transition_dict,
    b_transition_dict,
    neural_state_dict,
    max_steps=50,
    interval=2000,
    save_path=None,
    seed=42,
    route_sequence=None,
    show_plot=True
):
    """
    Animate lookup along a predetermined 50-step route.
    """

    # Probability versions
    b_prob_dict = convert_count_to_probability(b_transition_dict)
    n_prob_dict = convert_count_to_probability(neural_state_dict)
    pair_prob_dict = convert_count_to_probability(pair_transition_dict)

    # Labels for neural states
    n_label_map = make_n_state_labels(neural_state_dict)

    # Route
    if route_sequence is None:
        route_sequence = generate_supported_observed_pair_route(
            pair_transition_dict,
            neural_state_dict,
            b_transition_dict,
            route_len=max_steps,
            seed=seed
        )
    else:
        route_sequence = route_sequence[:max_steps]

    if len(route_sequence) == 0:
        raise ValueError("Route sequence is empty.")

    # Build full behavioral route for plotting in arena
    route_b_states = [route_sequence[0][0]] + [step[2] for step in route_sequence]
    route_coords = [(b[0], b[1]) for b in route_b_states]

    print(f"Generated support-filtered route with {len(route_sequence)} transitions")

    error_history = []

    fig = plt.figure(figsize=(15.5, 10.0))

    gs = gridspec.GridSpec(
        4,
        2,
        height_ratios=[0.60, 1.45, 1.65, 1.35],  # made row 3 taller for Box 3/4
        hspace=0.38,
        wspace=0.22
    )

    header_ax  = fig.add_subplot(gs[0, :])
    b_ax       = fig.add_subplot(gs[1, 0])
    n_ax       = fig.add_subplot(gs[1, 1])
    calc_ax    = fig.add_subplot(gs[2, 0])
    actual_ax  = fig.add_subplot(gs[2, 1])
    summary_ax = fig.add_subplot(gs[3, 0])
    route_ax   = fig.add_subplot(gs[3, 1])

    text_box_axes = [b_ax, n_ax, calc_ax, actual_ax, summary_ax]

    def setup_text_box(ax, facecolor="#f7f7f7"):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(facecolor)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#444444")

    def setup_route_ax(ax):
        ax.set_facecolor("#fbfbfb")

        # Actual arena is still x = 0..9, y = 0..9.
        # Extra x-space from 9.5..13 is reserved for annotation + legend.
        ax.set_xlim(-0.5, 13.0)
        ax.set_ylim(-0.5, 9.5)

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            "5. Predetermined route through arena",
            fontsize=12,
            fontweight="bold",
            pad=8
        )

        # Mark the true arena boundary.
        ax.axvline(
            x=9.5,
            color="#999999",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7
        )

        # Lightly shade the reserved legend/annotation region.
        ax.axvspan(
            9.5,
            13.0,
            color="white",
            alpha=0.85,
            zorder=0
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#444444")

    def heading_index_to_degrees(heading_idx):
        return int(heading_idx) * 90


    def fmt_b(b_state):
        x, y, heading_idx = b_state
        heading_degrees = heading_index_to_degrees(heading_idx)
        return f"({x}, {y}, {heading_degrees}°)"

    def fmt_n(n_state):
        return n_label_map.get(n_state, "N?")

    def fmt_pair(pair_state):
        b_state, n_state = pair_state
        return f"({fmt_b(b_state)}, {fmt_n(n_state)})"

    def update(frame_idx):
        # Clear
        for ax in [header_ax, b_ax, n_ax, calc_ax, actual_ax, summary_ax, route_ax]:
            ax.clear()

        for ax in text_box_axes:
            setup_text_box(ax)

        setup_route_ax(route_ax)
        header_ax.axis("off")

        # Current transition on predetermined route
        B0, N0, B1, N1, observed_count = route_sequence[frame_idx]

        # Counts
        b_count_nexts = b_transition_dict.get(B0, {})
        n_count_nexts = neural_state_dict.get(N0, {})
        pair_count_nexts = pair_transition_dict.get((B0, N0), {})

        b_total = sum(b_count_nexts.values()) if len(b_count_nexts) > 0 else 0
        n_total = sum(n_count_nexts.values()) if len(n_count_nexts) > 0 else 0
        pair_total = sum(pair_count_nexts.values()) if len(pair_count_nexts) > 0 else 0

        b_count = b_count_nexts.get(B1, 0)
        n_count = n_count_nexts.get(N1, 0)
        pair_count = pair_count_nexts.get((B1, N1), 0)

        # Probabilities
        error_info = compute_top3_route_choice_error(
            b_transition_dict,
            neural_state_dict,
            pair_transition_dict,
            B0,
            N0,
            B1,
            N1,
            top_k=3
        )


        p_b = error_info["p_b_route"]
        p_n = error_info["p_n_route"]

        route_independent_prob = error_info["p_route"]
        best_independent_prob = error_info["p_best"]

        raw_error = error_info["raw_error"]
        normalized_error = error_info["normalized_error"]
        C = error_info["C"]

        actual_prob = error_info["actual_pair_prob"]
        actual_lookup_error = error_info["actual_lookup_error"]

        b_count = error_info["b_route_count"]
        n_count = error_info["n_route_count"]
        pair_count = error_info["pair_count"]

        b_total = error_info["b_total"]
        n_total = error_info["n_total"]
        pair_total = error_info["pair_total"]

        b_in_top3 = error_info["b_in_top3"]
        n_in_top3 = error_info["n_in_top3"]
        b_rank_top3 = error_info["b_rank_top3"]
        n_rank_top3 = error_info["n_rank_top3"]


        if frame_idx >= len(error_history):
            error_history.append({
                "frame": frame_idx + 1,
                "raw_error": raw_error,
                "normalized_error": normalized_error,
                "C": C,
                "p_route": route_independent_prob,
                "p_best": best_independent_prob,
                "actual_pair_prob": actual_prob,
                "pair_count": pair_count,
                "pair_total": pair_total,
                "b_in_top3": b_in_top3,
                "n_in_top3": n_in_top3
            })

        # Labels
        B0_str = fmt_b(B0)
        B1_str = fmt_b(B1)
        N0_str = fmt_n(N0)
        N1_str = fmt_n(N1)

        # Ranks
        b_rank = rank_of_transition(b_count_nexts, B1)
        n_rank = rank_of_transition(n_count_nexts, N1)
        pair_rank = rank_of_transition(pair_count_nexts, (B1, N1))

        # Top 3
        top3_b = get_top_k_items(b_count_nexts, k=3)
        top3_n = get_top_k_items(n_count_nexts, k=3)
        top3_pair = get_top_k_items(pair_count_nexts, k=3)

        # ---------------- HEADER ----------------
        header_ax.text(
            0.5, 0.84,
            "Transition Probability Lookup",
            ha="center", va="center",
            fontsize=26, fontweight="bold"
        )

        header_ax.text(
            0.5, 0.56,
            f"Frame {frame_idx + 1} / {len(route_sequence)}",
            ha="center", va="center",
            fontsize=13, color="#555555"
        )

        header_ax.text(
            0.5, 0.20,
            f"Observed transition: ({B0_str}, {N0_str})  →  ({B1_str}, {N1_str})",
            ha="center", va="center",
            fontsize=15,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#eef5ff",
                edgecolor="#4a78c2"
            )
        )

        # ---------------- 1. BEHAVIORAL LOOKUP ----------------
        b_ax.text(0.03, 0.94, "1. Behavioral transition lookup",
                  ha="left", va="top", fontsize=15, fontweight="bold")

        b_lines = [
            "Observed:",
            f'  "{B0_str}" → "{B1_str}"',
            f"  P(B₁ | B₀) = {p_b:.6f} ({b_count}/{b_total})"
        ]
        if b_rank is not None:
            b_lines.append(f"  Rank of observed B transition: #{b_rank}")

        b_lines += ["", "Top 3 transitions from B₀:"]
        for idx, (next_b, cnt) in enumerate(top3_b, start=1):
            prob = cnt / b_total if b_total > 0 else 0.0
            b_lines.append(f"  {idx}. {fmt_b(next_b)}: {prob:.4f} ({cnt}/{b_total})")

        b_ax.text(
            0.03, 0.84,
            "\n".join(b_lines),
            ha="left", va="top",
            fontsize=11.2, family="monospace", linespacing=1.18
        )

        # ---------------- 2. NEURAL LOOKUP ----------------
        n_ax.text(0.03, 0.94, "2. Neural transition lookup",
                  ha="left", va="top", fontsize=15, fontweight="bold")

        n_lines = [
            "Observed:",
            f'  "{N0_str}" → "{N1_str}"',
            f"  P(N₁ | N₀) = {p_n:.6f} ({n_count}/{n_total})"
        ]
        if n_rank is not None:
            n_lines.append(f"  Rank of observed N transition: #{n_rank}")

        n_lines += ["", "Top 3 transitions from N₀:"]
        for idx, (next_n, cnt) in enumerate(top3_n, start=1):
            prob = cnt / n_total if n_total > 0 else 0.0
            n_lines.append(f"  {idx}. {fmt_n(next_n)}: {prob:.4f} ({cnt}/{n_total})")

        n_ax.text(
            0.03, 0.84,
            "\n".join(n_lines),
            ha="left", va="top",
            fontsize=11.2, family="monospace", linespacing=1.18
        )

        # ---------------- 3. CALCULATED / PSEUDO ----------------
        calc_ax.text(0.03, 0.94, "3. Calculated pair probability",
                     ha="left", va="top", fontsize=15, fontweight="bold")

        calc_lines = [
            "                      ",
            "Route-choice estimate:",
            "",
            "Use P(B₁|B₀) × P(N₁|N₀)",
            "only if both are in top 3.",
            "",
            f"B transition in top 3: {b_in_top3}",
            f"N transition in top 3: {n_in_top3}",
            "",
            f"P_route = {p_b:.6f} × {p_n:.6f} = {route_independent_prob:.6f}"
        ]

        calc_ax.text(
            0.03,
            0.86,
            "\n".join(calc_lines),
            ha="left",
            va="top",
            fontsize=11.4,
            linespacing=1.12
        )

        # ---------------- 4. ACTUAL PAIR LOOKUP ----------------
        actual_ax.text(0.03, 0.94, "4. Actual pair transition lookup",
                       ha="left", va="top", fontsize=15, fontweight="bold")

        pair_lines = [
            "Observed:",
            f'  "{fmt_pair((B0, N0))}"',
            f'      → "{fmt_pair((B1, N1))}"',
            f"  Observed pair prob = {actual_prob:.6f} ({pair_count}/{pair_total})"
        ]
        if pair_rank is not None:
            pair_lines.append(f"  Rank of observed pair transition: #{pair_rank}")

        pair_lines += ["", "Top 3 transitions from pair₀:"]
        for idx, (next_pair, cnt) in enumerate(top3_pair, start=1):
            prob = cnt / pair_total if pair_total > 0 else 0.0
            next_b, next_n = next_pair
            pair_lines.append(
                f"  {idx}. ({fmt_b(next_b)}, {fmt_n(next_n)}): {prob:.4f} ({cnt}/{pair_total})"
            )

        actual_ax.text(
            0.03, 0.84,
            "\n".join(pair_lines),
            ha="left", va="top",
            fontsize=10.4, family="monospace", linespacing=1.15
        )

        # ---------------- SUMMARY ----------------
        summary_ax.text(
            0.03,
            0.90,
            "Comparison",
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold"
        )

        summary_lines = [
            f"Best independent transition:  {best_independent_prob:.6f}",
            f"Route independent transition: {route_independent_prob:.6f}",
            f"Actual paired lookup:         {actual_prob:.6f}",
            f"Route-choice error = best - route:   {raw_error:.6f}",
            f"Actual lookup error = actual - best: {actual_lookup_error:.6f}"
        ]

        summary_ax.text(
            0.03,
            0.70,
            "\n".join(summary_lines),
            ha="left",
            va="top",
            fontsize=10.8,
            family="monospace",
            linespacing=1.55
        )

        # ---------------- ROUTE PANEL ----------------
        full_x = [c[0] for c in route_coords]
        full_y = [c[1] for c in route_coords]

        traversed_coords = route_coords[:frame_idx + 2]
        cur_x = [c[0] for c in traversed_coords]
        cur_y = [c[1] for c in traversed_coords]

        full_route_line, = route_ax.plot(
            full_x,
            full_y,
            "--",
            color="lightgray",
            linewidth=1.5,
            label="Full route"
        )

        traversed_line, = route_ax.plot(
            cur_x,
            cur_y,
            "-",
            color="#1f77b4",
            linewidth=2.5,
            label="Traversed"
        )

        start_point = route_ax.scatter(
            cur_x[0],
            cur_y[0],
            s=45,
            color="green",
            zorder=5,
            label="Start"
        )

        current_point = route_ax.scatter(
            cur_x[-1],
            cur_y[-1],
            s=65,
            color="red",
            zorder=6,
            label="Current"
        )

        # Put current-state annotation in the reserved right-side area.
        current_heading_degrees = heading_index_to_degrees(B1[2])

        route_ax.text(
            9.75,
            9.15,
            f"Current state:\n{fmt_b(B1)}",
            ha="left",
            va="top",
            fontsize=9.0,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.95,
                edgecolor="#666666"
            ),
            zorder=10
        )

        # Put legend below the annotation in the reserved right-side area.
        route_ax.legend(
            handles=[full_route_line, traversed_line, start_point, current_point],
            labels=["Full route", "Traversed", "Start", "Current"],
            loc="upper left",
            bbox_to_anchor=(9.75, 6.35),
            bbox_transform=route_ax.transData,
            fontsize=8.8,
            frameon=True
        )

        # Optional: visually mark the real arena boundary so the extra legend area is clearly separate.
        route_ax.axvline(
            x=9.5,
            color="#999999",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(route_sequence),
        interval=interval,
        repeat=True
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.06,
        hspace=0.36,
        wspace=0.22
    )

    if save_path is not None:
        fps = max(1, round(1000 / interval))
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end in .gif or .mp4")

    if show_plot:
        plt.show()
    return anim, route_sequence, error_history


def evaluate_error_history_on_fixed_route(route_sequence, b_transition_dict, neural_state_dict, pair_transition_dict, top_k=3, apply_top3_filter=False):
    """
    Evaluates route-choice error over a fixed paired route.

    The route_sequence must be:
        [(B0, N0, B1, N1, count), ...]

    The exact same route_sequence can be evaluated against different sweeps,
    such as min_attempts=50 and min_attempts=100.
    """

    error_history = []

    for frame_idx, (B0, N0, B1, N1, observed_count) in enumerate(route_sequence):
        if apply_top3_filter:
            error_info = compute_top3_route_choice_error(
                b_transition_dict,
                neural_state_dict,
                pair_transition_dict,
                B0,
                N0,
                B1,
                N1,
                top_k=top_k
            )
        else:
            error_info = compute_top3_route_choice_error(
                b_transition_dict,
                neural_state_dict,
                pair_transition_dict,
                B0,
                N0,
                B1,
                N1,
                top_k=None
            )

        error_history.append({
            "frame": frame_idx + 1,

            "B0": B0,
            "N0": N0,
            "B1": B1,
            "N1": N1,

            "raw_error": error_info["raw_error"],
            "normalized_error": error_info["normalized_error"],
            "actual_lookup_error": error_info["actual_lookup_error"],
            "normalized_actual_lookup_error": error_info["normalized_actual_lookup_error"],
            "C": error_info["C"],

            "p_route": error_info["p_route"],
            "p_best": error_info["p_best"],

            "p_b_route": error_info["p_b_route"],
            "p_n_route": error_info["p_n_route"],
            "p_b_best": error_info["p_b_best"],
            "p_n_best": error_info["p_n_best"],

            "actual_pair_prob": error_info["actual_pair_prob"],

            "pair_count": error_info["pair_count"],
            "pair_total": error_info["pair_total"],

            "b_route_count": error_info["b_route_count"],
            "b_total": error_info["b_total"],

            "n_route_count": error_info["n_route_count"],
            "n_total": error_info["n_total"],

            "b_in_top3": error_info["b_in_top3"],
            "n_in_top3": error_info["n_in_top3"],
            "b_rank_top3": error_info["b_rank_top3"],
            "n_rank_top3": error_info["n_rank_top3"],

            "num_b_options": error_info["num_b_options"],
            "num_n_options": error_info["num_n_options"],
            "num_pair_options": error_info["num_pair_options"]
        })

    return error_history

def plot_error_histories_over_time(error_history_50, error_history_100, title="Route-choice error over same paired route", use_normalized=True, show_points=True):
    def extract(error_history):
        frames = [item["frame"] for item in error_history]

        if use_normalized:
            errors = [item["normalized_error"] for item in error_history]
            ylabel = "Normalized route-choice error"
        else:
            errors = [item["raw_error"] for item in error_history]
            ylabel = "Raw route-choice error"

        return frames, errors, ylabel

    frames_50, errors_50, ylabel = extract(error_history_50)
    frames_100, errors_100, _ = extract(error_history_100)

    plt.figure(figsize=(11, 5.5))

    marker = "o" if show_points else None

    plt.plot(
        frames_50,
        errors_50,
        marker=marker,
        linewidth=1.8,
        label="min_attempts = 50"
    )

    plt.plot(
        frames_100,
        errors_100,
        marker=marker,
        linewidth=1.8,
        label="min_attempts = 100"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6
    )

    plt.xlabel("Frame along shared paired route")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("min_attempts = 50")
    print("Mean:", np.mean(errors_50))
    print("Median:", np.median(errors_50))
    print("Max:", np.max(errors_50))
    print("Min:", np.min(errors_50))

    print("\nmin_attempts = 100")
    print("Mean:", np.mean(errors_100))
    print("Median:", np.median(errors_100))
    print("Max:", np.max(errors_100))
    print("Min:", np.min(errors_100))




def summarize_fixed_route_coverage(
    route_sequence,
    b_transition_dict,
    neural_state_dict,
    pair_transition_dict,
    label="sweep"
):
    missing_B0 = 0
    missing_N0 = 0
    missing_pair0 = 0

    observed_B_transition_missing = 0
    observed_N_transition_missing = 0
    observed_pair_transition_missing = 0

    for B0, N0, B1, N1, _ in route_sequence:
        b_next = b_transition_dict.get(B0, {})
        n_next = neural_state_dict.get(N0, {})
        pair_next = pair_transition_dict.get((B0, N0), {})

        if len(b_next) == 0:
            missing_B0 += 1
        if len(n_next) == 0:
            missing_N0 += 1
        if len(pair_next) == 0:
            missing_pair0 += 1

        if B1 not in b_next:
            observed_B_transition_missing += 1
        if N1 not in n_next:
            observed_N_transition_missing += 1
        if (B1, N1) not in pair_next:
            observed_pair_transition_missing += 1

    total = len(route_sequence)

    print(f"\nCoverage summary for {label}")
    print("-" * 40)
    print(f"Total route transitions: {total}")
    print(f"Missing B0 states: {missing_B0}/{total}")
    print(f"Missing N0 states: {missing_N0}/{total}")
    print(f"Missing pair0 states: {missing_pair0}/{total}")
    print(f"Observed B transition missing: {observed_B_transition_missing}/{total}")
    print(f"Observed N transition missing: {observed_N_transition_missing}/{total}")
    print(f"Observed pair transition missing: {observed_pair_transition_missing}/{total}")

def summarize_error_history(error_history, label="sweep"):
    import numpy as np

    raw_errors = np.array([e["raw_error"] for e in error_history])
    norm_errors = np.array([e["normalized_error"] for e in error_history])

    b_top3 = np.array([e["b_in_top3"] for e in error_history])
    n_top3 = np.array([e["n_in_top3"] for e in error_history])

    both_top3 = b_top3 & n_top3

    print(f"\nError summary for {label}")
    print("-" * 40)
    print("Mean raw error:", raw_errors.mean())
    print("Median raw error:", np.median(raw_errors))
    print("Max raw error:", raw_errors.max())

    print("Mean normalized error:", norm_errors.mean())
    print("Median normalized error:", np.median(norm_errors))
    print("Max normalized error:", norm_errors.max())

    print("B route transition in top 3:", b_top3.sum(), "/", len(b_top3))
    print("N route transition in top 3:", n_top3.sum(), "/", len(n_top3))
    print("Both B and N in top 3:", both_top3.sum(), "/", len(both_top3))


if __name__ == "__main__":
    model = small_model.RNN().to(device)
    num_steps = 2
    
    #n_state_histogram = n_state_distribution_histograms_individual()
    #binned_overlay = n_state_histogram_overlay_binned(bin_size=100, reduce_to_same_bins=True, max_display_bin=1500)
    #summary = n_state_distribution_summary()
    #b_state_bar_graphs = b_state_distribution_barplots()

    #sweep_results = sweep_data_gen(model, step_size=5, min_attempts=50, max_attempts=101, sd=42)
    #print(sweep_results.keys())
    
    '''heatmap_pair_unique_edges_100 = paired_transition_density_heatmap(
        pair_transition_dict_100,
        grid_size=10,
        mode="unique_edges",
        title="Unique Paired Transition Density by Position, min_attempts=100"
    )'''
    pair_transition_dict_100, behavioral_transition_dict_100, neural_state_dict_100, all_visit_count_b_dict_100, all_visit_count_n_dict_100 = generate_dicts(model)
    #pair_transition_dict_50, behavioral_transition_dict_50, neural_state_dict_50, all_visit_count_b_dict_50, all_visit_count_n_dict_50 = generate_dicts(model, min_visits=50)
    #pair_transition_dict_75, behavioral_transistion_dict_75, neural_state_dict_75, _, _ = generate_dicts(model, min_visits=75)
    '''shared_route_sequence = generate_supported_observed_pair_route(
        pair_transition_dict_100,
        neural_state_dict_100,
        behavioral_transition_dict_100,
        route_len=50,
        seed=42,
        min_pair_total=3,
        min_pair_count=1,
        min_n_total=3,
        min_b_total=20
    )

    error_history_50 = evaluate_error_history_on_fixed_route(
        shared_route_sequence,
        behavioral_transition_dict_50,
        neural_state_dict_50,
        pair_transition_dict_50,
        top_k=3,
        apply_top3_filter=False
    )

    error_history_100 = evaluate_error_history_on_fixed_route(
        shared_route_sequence,
        behavioral_transition_dict_100,
        neural_state_dict_100,
        pair_transition_dict_100,
        top_k=3,
        apply_top3_filter=False
    )'''

    _, route100, error_history_100 = animate_json_lookup_transition_clean(pair_transition_dict_100, behavioral_transition_dict_100, neural_state_dict_100, max_steps=50, interval=1200, save_path="lookup_animation_min100.mp4")
    np.save(os.path.join(SCRIPT_DIR, "route_sequence_min100.npy"), np.array(route100, dtype=object), allow_pickle=True)

    #np.save(os.path.join(SCRIPT_DIR, "error_history_min100.npy"), np.array(error_history_100, dtype=object), allow_pickle=True)

    top3_route_history_100 = save_top3_route_transitions(route100, behavioral_transition_dict_100, neural_state_dict_100, pair_transition_dict_100, top_k=3)

    np.save(os.path.join(SCRIPT_DIR, "top3_route_transitions_min100.npy"), np.array(top3_route_history_100, dtype=object), allow_pickle=True)
    #error_plot_50_100 = plot_error_histories_over_time(error_history_50, error_history_100, title="Raw Error Plot Over Same Route: min50 vs min100", use_normalized=False, show_points=True)

    '''summarize_fixed_route_coverage(
        shared_route_sequence,
        behavioral_transition_dict_50,
        neural_state_dict_50,
        pair_transition_dict_50,
        label="min_attempts=50"
    )

    summarize_fixed_route_coverage(
        shared_route_sequence,
        behavioral_transition_dict_100,
        neural_state_dict_100,
        pair_transition_dict_100,
        label="min_attempts=100"
    )

    summarize_error_history(error_history_50, label="min_attempts=50")
    summarize_error_history(error_history_100, label="min_attempts=100")'''

    '''b_to_n_dict = build_b_to_n_map(pair_transition_dict_100)
    nheat = n_state_distribution_heatmap(b_to_n_dict, agg="union")
    bheat = b_state_distribution_heatmap(all_visit_count_b_dict_100, agg="mean")
    joint_bn_distribution_heatmap = joint_b_n_dist_heatmap(normalize=True, b_agg="mean", n_agg="union")

    #n_distrib_overlay = n_state_distribution_overlay_binned_barplots(top_k=100, bin_size=20, reduce="sum")

    json_path = os.path.join(SCRIPT_DIR, "behavioral_neural_state_table.json")
    out_ready = json_b_to_n_state(pair_transition_dict, 'count')
    with open(json_path, "w") as f:
        json.dump(out_ready, f, indent=2)

    converted_prob_n_transition_dict = convert_count_to_probability(neural_state_dict)
    json_prob_path = os.path.join(SCRIPT_DIR, "behavioral_to_neural_state_probabilities.json")
    out_ready_prob = json_b_to_n_state(converted_prob_n_transition_dict, 'probability')
    with open(json_prob_path, "w") as f:
        json.dump(out_ready_prob, f, indent=2)
    
    one_step_b_trans, one_step_n_trans, one_step_pair_trans = one_step_probability(
        behavioral_transition_dict, n_state_dict, pair_transition_dict
    )

    one_step_json_path = os.path.join(SCRIPT_DIR, "one_step_transition_probabilities.json")
    one_step_pair_out_ready = json_b_to_n_state(one_step_pair_trans, 'probability')
    with open(one_step_json_path, "w") as f:
        json.dump(one_step_pair_out_ready, f, indent=2)

    freq_b, freq_n, freq_pair, n_step_dict = n_step_sparse_probability(
        num_steps=num_steps,
        b_trans_dict=b_transition_dict,
        n_state_dict=n_state_dict,
        pair_dict=pair_transition_dict
    )

    n_step_json_path = os.path.join(SCRIPT_DIR, f"{num_steps}_step_transition_probabilities.json")
    n_step_out_ready = json_b_to_n_state(n_step_dict, 'probability')
    with open(n_step_json_path, "w") as f:
        json.dump(n_step_out_ready, f, indent=2)
    num_steps = 2
    #test_deg = test_degeneracy(pair_transition_dict)
    b_to_n_JSON, n_to_b_JSON = dynamic_degeneracy_probability(pair_transition_dict, num_steps=num_steps, gen_JSON=True)

    b_to_n_dynamic_json_path = os.path.join(SCRIPT_DIR, f"b_to_n_dynamic_degeneracy_{num_steps}_step.json")
    with open(b_to_n_dynamic_json_path, "w") as f:
        json.dump(b_to_n_JSON, f, indent=2)

    n_to_b_dynamic_json_path = os.path.join(SCRIPT_DIR, f"n_to_b_dynamic_degeneracy_{num_steps}_step.json")
    with open(n_to_b_dynamic_json_path, "w") as f:
        json.dump(n_to_b_JSON, f, indent=2)"""
        
    scores_b_to_n, scores_n_to_b = dynamic_degeneracy_score(pair_transition_dict)

    print("\nDynamic degeneracy scores (b to n):")
    print("Mean dynamic difference:", np.mean(scores_b_to_n))
    print("Max difference:", np.max(scores_b_to_n))
    print("num score comparisons:", len(scores_b_to_n))
    print("mean dynamic difference:", np.mean(scores_b_to_n))
    print("min dynamic difference:", np.min(scores_b_to_n))
    print("max dynamic difference:", np.max(scores_b_to_n))

    print("\nDynamic degeneracy scores (n to b):")
    print("Mean dynamic difference:", np.mean(scores_n_to_b))
    print("Max difference:", np.max(scores_n_to_b))
    print("num score comparisons:", len(scores_n_to_b))
    print("mean dynamic difference:", np.mean(scores_n_to_b))
    print("min dynamic difference:", np.min(scores_n_to_b))
    print("max dynamic difference:", np.max(scores_n_to_b))'''
    