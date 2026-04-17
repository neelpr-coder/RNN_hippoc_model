import torch 
import numpy as np
import os
import json
import random
import ast

from PIL import Image
from collections import defaultdict
from tqdm import tqdm

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

def neural_state_to_dict_key(neural_state, bin_size=0.2):
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
        sampled_x = np.clip(sampled_x, 0, 16)
        sampled_y = np.clip(sampled_y, 0, 16)
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
    
    cache_path = os.path.join(CACHE_DIR, f"behavioral_to_neural_state_table_min{min_visits}_max{max_visits}_sd{sd}.npz")
    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        loaded_pair_transition_dict = loaded["pair_dict"].item()
        loaded_trans_dict = loaded["b_transition_dict"].item()
        loaded_neural_state_dict = loaded["neural_state_dict"].item()
        pair_transition_dict = defaultdict(lambda: defaultdict(int))
        trans_dict = defaultdict(lambda: defaultdict(int))
        neural_state_dict = defaultdict(lambda: defaultdict(int))
        for b_state, freq_dict in loaded_pair_transition_dict.items():
            pair_transition_dict[b_state] = defaultdict(int, freq_dict)
        
        for cur_state, freq in loaded_trans_dict.items():
            trans_dict[cur_state] = defaultdict(int, freq)

        for cur_state, next_states in loaded_neural_state_dict.items():
            neural_state_dict[cur_state] = defaultdict(int, next_states)

        print("[Log] cache already exists")
        return pair_transition_dict, trans_dict, neural_state_dict
    else:
        print("[Log] creating the cache...")
    
        b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count
        np.random.seed(sd)
        all_b_states = list(all_visit_count_dict.keys())
        starting_point = all_b_states[np.random.randint(0, len(all_b_states))] # randomly select a behavioral state as a starting point
        
        pair_transition_dict = defaultdict(lambda: defaultdict(int))
        transition_dict = defaultdict(lambda: defaultdict(int))
        neural_state_dict = defaultdict(lambda: defaultdict(int))
        
        done = min(all_visit_count_dict.values()) >= min_visits
        is_first_visit = True
        model = net
        model.eval()

        total_targets = len(all_b_states) * min_visits
        current_progress = sum(min(v, min_visits) for v in all_visit_count_dict.values())
        progress_bar = tqdm(total=total_targets, initial=current_progress, desc="Generating the table")

        with torch.no_grad():
            h = None
            while not done:
                available_states = [s for s in all_b_states if all_visit_count_dict[s] < max_visits] # want the list of available states to keep updating each iteration
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

                    cur_count = all_visit_count_dict[cur_b_state]
                    all_visit_count_dict[cur_b_state] += 1

                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_key =  neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy())
                    cur_b_state_key = behavioral_state_to_key(cur_b_state)
                    next_neural_state_key = neural_state_to_dict_key(next_neural_state.detach().cpu().numpy())
                    next_b_state_key = behavioral_state_to_key(next_b_state)

                    # update the dictionaries
                    transition_dict[cur_b_state_key][next_b_state_key] += 1
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

                    cur_count = all_visit_count_dict[cur_b_state]
                    all_visit_count_dict[cur_b_state] += 1

                    if cur_count < min_visits:
                        progress_bar.update(1)
                    
                    cur_neural_state_key =  neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy())
                    cur_b_state_key = behavioral_state_to_key(cur_b_state)

                    next_neural_state_key = neural_state_to_dict_key(next_neural_state.detach().cpu().numpy())
                    next_b_state_key = behavioral_state_to_key(next_b_state)
                    
                    # update the dictionaries
                    transition_dict[cur_b_state_key][next_b_state_key] += 1
                    neural_state_dict[cur_neural_state_key][next_neural_state_key] += 1
                    pair_transition_dict[(cur_b_state_key, cur_neural_state_key)][(next_b_state_key, next_neural_state_key)] += 1

                done = min(all_visit_count_dict.values()) >= min_visits

        progress_bar.close()
        save_dict = {k: dict(v) for k, v in pair_transition_dict.items()}
        save_transition = {k: dict(v) for k, v in transition_dict.items()}
        save_neural_state = {k: dict(v) for k, v in neural_state_dict.items()}
        np.savez_compressed(cache_path, 
                            pair_dict=np.array(save_dict, dtype=object), 
                            b_transition_dict=np.array(save_transition, dtype=object),
                            neural_state_dict=np.array(save_neural_state, dtype=object)
                            )

        return pair_transition_dict, transition_dict, neural_state_dict

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

def dynamic_degeneracy_score(pair_transition_dict, num_steps=2, gen_JSON = False):
    freq_pair = convert_count_to_probability(pair_transition_dict)
    b_to_n = build_b_to_n_map(pair_transition_dict)
    n_to_b = build_n_to_b_map(pair_transition_dict)

    scores_b_to_n = []
    scores_n_to_b = []

    out_b_to_n = {}
    out_n_to_b = {}

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

                dist1 = n_step_from_start(state1, freq_pair, num_steps=num_steps)
                dist2 = n_step_from_start(state2, freq_pair, num_steps=num_steps)

                score = l1_distance(dist1, dist2)
                scores_b_to_n.append(score)

                
                b_to_n_comparisons.append({
                    "state_1": str(state1),
                    "state_2": str(state2),
                    "l1_distance": float(score),
                    "dist_1": {str(k): float(v) for k, v in dist1.items()},
                    "dist_2": {str(k): float(v) for k, v in dist2.items()},
                })

            if gen_JSON and b_to_n_comparisons:
                out_b_to_n[str(b)] = b_to_n_comparisons
                return out_b_to_n

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

                dist1 = n_step_from_start(state1, freq_pair, num_steps=num_steps)
                dist2 = n_step_from_start(state2, freq_pair, num_steps=num_steps)

                score_n_to_b = l1_distance(dist1, dist2)
                scores_n_to_b.append(score_n_to_b)

                n_to_b_comparisons.append({
                    "state_1": str(state1),
                    "state_2": str(state2),
                    "l1_distance": float(score),
                    "dist_1": {str(k): float(v) for k, v in dist1.items()},
                    "dist_2": {str(k): float(v) for k, v in dist2.items()},
                })
        
        if gen_JSON and n_to_b_comparisons:
            out_n_to_b[str(n)] = n_to_b_comparisons
            return out_n_to_b

    return scores_b_to_n, scores_n_to_b



if __name__ == "__main__":
    model = small_model.RNN().to(device)
    num_steps = 2

    pair_transition_dict, b_transition_dict, n_state_dict = generate_dicts(model)

    '''json_path = os.path.join(SCRIPT_DIR, "behavioral_neural_state_table.json")
    out_ready = json_b_to_n_state(pair_transition_dict, 'count')
    with open(json_path, "w") as f:
        json.dump(out_ready, f, indent=2)
    
    one_step_b_trans, one_step_n_trans, one_step_pair_trans = one_step_probability(
        b_transition_dict, n_state_dict, pair_transition_dict
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
        json.dump(n_step_out_ready, f, indent=2)'''

    test_deg = test_degeneracy(pair_transition_dict)
    dynamic_json = dynamic_degeneracy_json(pair_transition_dict, num_steps=2, gen_JSON=True)

    dynamic_json_path = os.path.join(SCRIPT_DIR, "dynamic_degeneracy_5_step.json")
    with open(dynamic_json_path, "w") as f:
        json.dump(dynamic_json, f, indent=2)
        
    '''scores_b_to_n, scores_n_to_b = dynamic_degeneracy_score(pair_transition_dict)

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
    