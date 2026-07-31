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
import matplotlib.pyplot as plt
import math

COPY_FOLDER = "Copy_Transition_Cache"
COPY_NAME_N = "copy_n_state_min100_max151_sd42_bin_size0.3.npz"
COPY_NAME_B = "copy_b_state_min100_max151_sd42_bin_size0.3.npz"
COPY_NAME_PAIR = "copy_pair_transition_min100_max151_sd42_bin_size0.3.npz"

ORIGINAL_N_PATH = "RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.3.npz"
ORIGINAL_B_PATH = "RNN_cache/b_state/b_state_min100_max151_sd42_bin_size0.3.npz"
ORIGINAL_PAIR_PATH = "RNN_cache/pair_transition/pair_transition_min100_max151_sd42_bin_size0.3.npz"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

n_path_copy = os.path.join(COPY_FOLDER, COPY_NAME_N)
b_path_copy = os.path.join(COPY_FOLDER, COPY_NAME_B)
pair_path_copy = os.path.join(COPY_FOLDER, COPY_NAME_PAIR)
os.makedirs(COPY_FOLDER, exist_ok=True)
# make a copy of the neural state transition .npz 
if not os.path.exists(n_path_copy):
    shutil.copy("RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.3.npz", n_path_copy)
if not os.path.exists(b_path_copy):
    shutil.copy("RNN_cache/b_state/b_state_min100_max151_sd42_bin_size0.3.npz", b_path_copy)
if not os.path.exists(pair_path_copy):
    shutil.copy("RNN_cache/pair_transition/pair_transition_min100_max151_sd42_bin_size0.3.npz", pair_path_copy)

def inner_dict():
    return defaultdict(int)
def outer_dict():
    return defaultdict(inner_dict)

def calc_stage2_error(frame, B0, N0, original_b_dict, original_n_dict, original_pair_dict, updated_b_dict, updated_n_dict, updated_pair_dict):
    original_b_next = original_b_dict.get(B0, {})
    original_n_next = original_n_dict.get(N0, {})
    original_pair_next = original_pair_dict.get((B0, N0), {})

    updated_b_next = updated_b_dict.get(B0, {})
    updated_n_next = updated_n_dict.get(N0, {})
    updated_pair_next = updated_pair_dict.get((B0, N0), {})

    # Top probabilities from original dictionaries.
    original_b_top_prob, original_b_top, _, _ = f2g.top_choice_prob(original_b_next)
    original_n_top_prob, original_n_top, _, _ = f2g.top_choice_prob(original_n_next)
    original_pair_top_prob, original_pair_top, _, _ = f2g.top_choice_prob(original_pair_next)

    # Top probabilities from updated dictionaries.
    updated_b_top_prob, updated_b_top, _, _ = f2g.top_choice_prob(updated_b_next)
    updated_n_top_prob, updated_n_top, _, _ = f2g.top_choice_prob(updated_n_next)
    updated_pair_top_prob, updated_pair_top, _, _ = f2g.top_choice_prob(updated_pair_next)
    
    # Independent probability
    original_best_probability = original_b_top_prob * original_n_top_prob
    updated_best_probability = updated_b_top_prob * updated_n_top_prob
    raw_error = original_best_probability - updated_best_probability

    # Direct paired-transition comparison.
    pair_raw_error = original_pair_top_prob - updated_pair_top_prob

    # Use the original choice space so the denominator stays fixed.
    C = len(original_b_next) + len(original_n_next) + len(original_pair_next)
    
    normalized_error = raw_error / C if C > 0 else 0.0
    normalized_pair_error = pair_raw_error / C if C > 0 else 0.0
    
    return {
        "frame": frame,
        "B0": B0,
        "N0": N0,

        "original_best_probability":
            original_best_probability,

        "updated_best_probability":
            updated_best_probability,

        "raw_error": raw_error,
        "normalized_error": normalized_error,

        "original_pair_top_probability":
            original_pair_top_prob,

        "updated_pair_top_probability":
            updated_pair_top_prob,

        "pair_raw_error": pair_raw_error,
        "normalized_pair_error": normalized_pair_error,

        "C": C,

        "original_b_top": original_b_top,
        "original_n_top": original_n_top,
        "original_pair_top": original_pair_top,

        "updated_b_top": updated_b_top,
        "updated_n_top": updated_n_top,
        "updated_pair_top": updated_pair_top
    }

def run_pertubation(model, total_num_pertubations, sd=42):
    '''Run pertubation such that one node is randomly knockout hooked at each step
        A minimum of 1000 total pertubations must take place
        Each node must be pertubed between 50 to a 100 times such that the sum of each node's pertubation count is 1000
        Update observed transitions to the copied transition dicts
        Keep track of stage 2 error original top transition - new top transition post pertubation'''
    stage2_error_history = []
    pertub_counter = 0
    neuron_visit_count_list = np.zeros(model.hidden_size, dtype=int)
    max_pertubs = total_num_pertubations 
    np.random.seed(sd)
    random.seed(sd)
    is_first_visit = True

    with np.load(ORIGINAL_N_PATH, allow_pickle=True) as original_n_file:
        original_n_transition_dict = (original_n_file["neural_state_dict"].item())

    with np.load(ORIGINAL_B_PATH, allow_pickle=True) as original_b_file:
        original_b_transition_dict = (original_b_file["b_transition_dict"].item())

    with np.load(ORIGINAL_PAIR_PATH, allow_pickle=True) as original_pair_file:
        original_pair_transition_dict = (original_pair_file["pair_dict"].item())
    
    # load and open the copied dicts for writing
    with np.load(n_path_copy, allow_pickle=True) as data_n:
        loaded_n = data_n["neural_state_dict"].item()

    with np.load(b_path_copy, allow_pickle=True) as data_b:
        loaded_b = data_b["b_transition_dict"].item()

    with np.load(pair_path_copy, allow_pickle=True) as data_pair:
            loaded_pair = data_pair["pair_dict"].item()

    updated_n_transition_dict = defaultdict(lambda: defaultdict(int))
    updated_b_transition_dict = defaultdict(lambda: defaultdict(int))
    updated_pair_transition_dict = defaultdict(lambda: defaultdict(int))

    for cur_state, next_states in loaded_n.items():
        updated_n_transition_dict[cur_state] = defaultdict(int, next_states)

    for cur_state, next_states in loaded_b.items():
            updated_b_transition_dict[cur_state] = defaultdict(int, next_states)

    for cur_pair, next_pair in loaded_pair.items():
        updated_pair_transition_dict[cur_pair] = defaultdict(int, next_pair)

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

                cur_neural_state_key = f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.3)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.3)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz, needs to be fixed
                updated_n_transition_dict[cur_neural_state_key][next_neural_state_key] += 1
                updated_b_transition_dict[cur_b_state_key][next_b_state_key] += 1
                updated_pair_transition_dict[(cur_b_state_key, cur_neural_state_key)][(next_b_state_key, next_neural_state_key)] += 1

                # update the visit and pertub count lists
                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1
                stage2_error_history.append(calc_stage2_error(
                                        frame=pertub_counter,
                                        B0=cur_b_state_key,
                                        N0=cur_neural_state_key,
                                
                                        original_b_dict=original_b_transition_dict,
                                        original_n_dict=original_n_transition_dict,
                                        original_pair_dict=original_pair_transition_dict,
                                
                                        updated_b_dict=updated_b_transition_dict,
                                        updated_n_dict=updated_n_transition_dict,
                                        updated_pair_dict=updated_pair_transition_dict)
                                    )
                
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
                                    
                cur_neural_state_key =  f2g.neural_state_to_dict_key(cur_neural_state.detach().cpu().numpy(), bin_size=0.3)
                cur_b_state_key = f2g.behavioral_state_to_key(cur_b_state)
                next_neural_state_key = f2g.neural_state_to_dict_key(next_neural_state.detach().cpu().numpy(), bin_size=0.3)
                next_b_state_key = f2g.behavioral_state_to_key(next_b_state)

                # update the copied n transition .npz, needs to be fixed
                updated_n_transition_dict[cur_neural_state_key][next_neural_state_key] += 1
                updated_b_transition_dict[cur_b_state_key][next_b_state_key] += 1
                updated_pair_transition_dict[(cur_b_state_key, cur_neural_state_key)][(next_b_state_key, next_neural_state_key)] += 1

                # update the visit and pertub count lists
                neuron_visit_count_list[gen_rand_neuron_index] += 1
                pertub_counter += 1
                stage2_error_history.append(calc_stage2_error(
                                        frame=pertub_counter,
                                        B0=cur_b_state_key,
                                        N0=cur_neural_state_key,
                
                                        original_b_dict=original_b_transition_dict,
                                        original_n_dict=original_n_transition_dict,
                                        original_pair_dict=original_pair_transition_dict,
                
                                        updated_b_dict=updated_b_transition_dict,
                                        updated_n_dict=updated_n_transition_dict,
                                        updated_pair_dict=updated_pair_transition_dict)
                                    )
                
    save_n = {state: dict(next_states) for state, next_states in updated_n_transition_dict.items()}

    save_b = {state: dict(next_states) for state, next_states in updated_b_transition_dict.items()}

    save_pair = {pair: dict(next_pair) for pair, next_pair in updated_pair_transition_dict.items()}

    np.savez_compressed(n_path_copy, neural_state_dict=np.array(save_n, dtype=object))

    np.savez_compressed(b_path_copy, b_transition_dict=np.array(save_b, dtype=object))

    np.savez_compressed(pair_path_copy, pair_dict=np.array(save_pair, dtype=object))

    print("Perturbations per neuron:", neuron_visit_count_list)
    print("Total perturbations:", pertub_counter)
    print("[Log] Perturbations complete")

    return updated_b_transition_dict, updated_n_transition_dict, updated_pair_transition_dict, stage2_error_history

def outgoing_transition_counts(transition_dict):
    '''Sum the outgoing transition counts from a given dictionary'''
    outgoing_count = {}
    for cur_key, next_keys in transition_dict.items():
        outgoing_count[cur_key] = sum(next_keys.values())

    return outgoing_count

def graph_neural_distribution(neural_dict, title="Neural-State Outgoing Transition Distribution", bin_size = 100, show = True, save_path = None):
    '''Create a neural distribution histogram'''
    outgoing_counts = outgoing_transition_counts(neural_dict)

    visit_counts = list(outgoing_counts.values())

    if not visit_counts:
        raise ValueError("The neural transition dictionary is empty.")

    max_count = max(visit_counts)
    upper_limit = max(bin_size, math.ceil(max_count / bin_size) * bin_size)
    bins = np.arange(0, upper_limit + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(visit_counts, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Number of outgoing transitions")
    ax.set_ylabel("Number of neural states")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outgoing_counts

def graph_behavioral_distribution(behavioral_dict, title, collapse_headings=True, show=True, save_path=None):
    '''Create a behavioral distribution bar plot'''
    outgoing_counts = outgoing_transition_counts(
        behavioral_dict
    )

    if not outgoing_counts:
        raise ValueError("The behavioral transition dictionary is empty.")

    if collapse_headings:
        plotted_counts = defaultdict(int)
        for behavioral_state, count in outgoing_counts.items():
            x, y, _ = behavioral_state
            plotted_counts[(x, y)] += count
    else:
        plotted_counts = outgoing_counts

    sorted_states = sorted(plotted_counts.keys())

    counts = [plotted_counts[state] for state in sorted_states]

    if collapse_headings:
        labels = [f"({state[0]}, {state[1]})" for state in sorted_states]
    else:
        labels = [f"({state[0]}, {state[1]}, {state[2] * 90}°)" for state in sorted_states]

    x_positions = np.arange(len(sorted_states))
    figure_width = max(12, len(sorted_states) * 0.18)

    fig, ax = plt.subplots(figsize=(figure_width, 6))

    ax.bar(x_positions, counts, width=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Behavioral position" if collapse_headings else "Behavioral state")
    ax.set_ylabel("Number of outgoing transitions")

    tick_step = max(1, math.ceil(len(labels) / 40))

    shown_positions = x_positions[::tick_step]
    shown_labels = labels[::tick_step]

    ax.set_xticks(shown_positions)
    ax.set_xticklabels(shown_labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(plotted_counts)

def calc_stage3_error(route_sequence, original_b_dict, original_n_dict, original_pair_dict, updated_b_dict, updated_n_dict, updated_pair_dict):
    """
    Compare original and updated transition probabilities along the
    same fixed 50-step route.
    """
    error_history = []

    for frame_idx, route_step in enumerate(route_sequence,start=1):
        B0, N0, B1, N1, observed_count = route_step

        original_b_next = original_b_dict.get(B0, {})
        original_n_next = original_n_dict.get(N0, {})
        original_pair_next = original_pair_dict.get((B0, N0), {})

        updated_b_next = updated_b_dict.get(B0, {})
        updated_n_next = updated_n_dict.get(N0, {})
        updated_pair_next = updated_pair_dict.get((B0, N0), {})

        original_b_prob, _, _ = (f2g.transition_prob_from_counts(original_b_next, B1))
        original_n_prob, _, _ = (f2g.transition_prob_from_counts(original_n_next, N1))
        original_pair_prob, _, _ = (f2g.transition_prob_from_counts(original_pair_next, (B1, N1)))
        updated_b_prob, _, _ = (f2g.transition_prob_from_counts(updated_b_next, B1))
        updated_n_prob, _, _ = (f2g.transition_prob_from_counts(updated_n_next, N1))
        updated_pair_prob, _, _ = (f2g.transition_prob_from_counts(updated_pair_next, (B1, N1)))

        original_independent_probability = (original_b_prob * original_n_prob)
        updated_independent_probability = (updated_b_prob * updated_n_prob)

        raw_error = (original_independent_probability - updated_independent_probability)

        pair_raw_error = (original_pair_prob - updated_pair_prob)

        # Same normalization structure as figure2_generation.py.
        C = (len(original_b_next) + len(original_n_next) + len(original_pair_next))

        normalized_error = (raw_error / C if C > 0 else 0.0)
        normalized_pair_error = (pair_raw_error / C if C > 0 else 0.0)

        error_history.append({
            "frame": frame_idx,

            "B0": B0,
            "N0": N0,
            "B1": B1,
            "N1": N1,

            "original_probability":
                original_independent_probability,

            "updated_probability":
                updated_independent_probability,

            "raw_error": raw_error,
            "normalized_error": normalized_error,

            "original_pair_probability":
                original_pair_prob,

            "updated_pair_probability":
                updated_pair_prob,

            "pair_raw_error": pair_raw_error,
            "normalized_pair_error":
                normalized_pair_error,

            "C": C
        })

    return error_history

def plot_normalized_error_over_time(error_history, title, x_label, include_pair_error=False, save_path=None, show_points=True):
    frames = [
        item["frame"]
        for item in error_history
    ]

    normalized_errors = [
        item["normalized_error"]
        for item in error_history
    ]

    marker = "o" if show_points else None

    plt.figure(figsize=(11, 5.5))

    plt.plot(
        frames,
        normalized_errors,
        marker=marker,
        markersize=3,
        linewidth=1.6,
        label="Independent transition error"
    )

    if include_pair_error:
        normalized_pair_errors = [
            item["normalized_pair_error"]
            for item in error_history
        ]

        plt.plot(
            frames,
            normalized_pair_errors,
            marker=marker,
            markersize=3,
            linewidth=1.6,
            label="Paired transition error"
        )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6
    )

    plt.xlabel(x_label)
    plt.ylabel("Normalized probability error")
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    print("Mean normalized error:", np.mean(normalized_errors))
    print("Median normalized error:", np.median(normalized_errors))
    print("Maximum normalized error:", np.max(normalized_errors))
    print("Minimum normalized error:", np.min(normalized_errors))

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = small_model.RNN().to(device)
    checkpoint = torch.load("post_stage_1_model.pt", map_location=device)

    model.load_state_dict(checkpoint)
    model.eval()

    route_seq = np.load("route_sequence_min100.npy", allow_pickle=True)
    with np.load(ORIGINAL_N_PATH, allow_pickle=True) as original_n_file:
        og_n_dict = (original_n_file["neural_state_dict"].item())

    with np.load(ORIGINAL_B_PATH, allow_pickle=True) as original_b_file:
        og_b_dict = (original_b_file["b_transition_dict"].item())

    with np.load(ORIGINAL_PAIR_PATH, allow_pickle=True) as original_pair_file:
        og_pair_dict = (original_pair_file["pair_dict"].item())

    new_b_t_dict, new_n_t_dict, new_pair_t_dict, stage_2_error_his = run_pertubation(model, total_num_pertubations=1000, sd=42)

    # gen figures for visualization
    graph_neural_distribution(new_n_t_dict, title="Updated Neural-State Outgoing Transition Distribution")
    graph_neural_distribution(og_n_dict, title="Original Neural-State Outgoing Transition Distribution")
    graph_behavioral_distribution(new_b_t_dict, title="Updated Behavioral-State Outgoing Transition Counts", collapse_headings=False)
    graph_behavioral_distribution(og_b_dict, title="Original Behavioral-State Outgoing Transition Counts", collapse_headings=False)
    #plot_normalized_error_over_time(stage_2_error_his, title="Stage 2 Error Plot", x_label="Pertubation Steps", include_pair_error=True, save_path="stage2_normalized_error_over_time.png")
    #stage_3_error_his = calc_stage3_error(route_seq, og_b_dict, og_n_dict, og_pair_dict, new_b_t_dict, new_n_t_dict, new_pair_t_dict)
    #plot_normalized_error_over_time(stage_3_error_his, title="Stage 3 Error Plot", x_label="Time Steps", include_pair_error=False, save_path="stage3_normalized_error_over_time.png")