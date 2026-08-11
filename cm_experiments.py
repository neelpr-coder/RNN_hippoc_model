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
            #raw_Na_seen = set()
            #raw_Nb_seen = set()
            diagnostic_steps = 0
            max_diagnostic_steps = 20
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
                    #raw_Na_seen.add(tuple(np.round(cur_neural_state_a.detach().cpu().numpy().reshape(-1), 6)))
                    #raw_Nb_seen.add(tuple(np.round(cur_neural_state_b.detach().cpu().numpy().reshape(-1), 6)))
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
                    #raw_Na_seen.add(tuple(np.round(next_neural_state_a.detach().cpu().numpy().reshape(-1), 6)))
                    #raw_Nb_seen.add(tuple(np.round(next_neural_state_b.detach().cpu().numpy().reshape(-1), 6)))

                    if diagnostic_steps < max_diagnostic_steps:
                        input_a = model.in_current_a(next_b_state_img_tensor.reshape(1, -1))
                        input_b = model.in_current_b(next_b_state_img_tensor.reshape(1, -1))
                    #print(f"[Step {diagnostic_steps}] A input={input_a.abs().mean().item():.4f}, recurrent={next_I_aa.abs().mean().item():.4f}, raw cross={next_I_ba.abs().mean().item():.4f}, scaled cross={(model.cross_strength * next_I_ba).abs().mean().item():.4f}")
                    #print(f"[Step {diagnostic_steps}] B input={input_b.abs().mean().item():.4f}, recurrent={next_I_bb.abs().mean().item():.4f}, raw cross={next_I_ab.abs().mean().item():.4f}, scaled cross={(model.cross_strength * next_I_ab).abs().mean().item():.4f}")                    
                    diagnostic_steps += 1
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
                    #raw_Na_seen.add(tuple(np.round(next_neural_state_a.detach().cpu().numpy().reshape(-1), 6)))
                    #raw_Nb_seen.add(tuple(np.round(next_neural_state_b.detach().cpu().numpy().reshape(-1), 6)))

                    if diagnostic_steps < max_diagnostic_steps:
                        input_a = model.in_current_a(next_b_state_img_tensor.reshape(1, -1))
                        input_b = model.in_current_b(next_b_state_img_tensor.reshape(1, -1))
                        #print(f"[Step {diagnostic_steps}] A input={input_a.abs().mean().item():.4f}, recurrent={next_I_aa.abs().mean().item():.4f}, cross={next_I_ba.abs().mean().item():.4f}")
                        #print(f"[Step {diagnostic_steps}] B input={input_b.abs().mean().item():.4f}, recurrent={next_I_bb.abs().mean().item():.4f}, cross={next_I_ab.abs().mean().item():.4f}")
                    diagnostic_steps += 1
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
        #print("Unique raw Na states:", len(raw_Na_seen))
        #print("Unique raw Nb states:", len(raw_Nb_seen))
        #print("Unique binned Na states:", len(set(Na_transition_dict.keys()) | {n for nexts in Na_transition_dict.values() for n in nexts}))
        #print("Unique binned Nb states:", len(set(Nb_transition_dict.keys()) | {n for nexts in Nb_transition_dict.values() for n in nexts}))
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

def copy_transition_dict(source):
    copied = defaultdict(lambda: defaultdict(int))
    for current_state, next_states in source.items():
        copied[current_state] = defaultdict(int, {next_state: int(count) for next_state, count in next_states.items()})
    return copied

def plain_transition_dict(source):
    return {current_state: dict(next_states) for current_state, next_states in source.items()}

def transition_count(source):
    return sum(sum(next_states.values()) for next_states in source.values())

def load_grayscale_tensor(image_path, tensor_device):
    image = Image.open(image_path).convert("L").resize((25, 25))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.tensor(image_array, dtype=torch.float32, device=tensor_device)

def _build_balanced_neuron_order(hidden_size, total_perturbations, sd=42):
    repeats, remainder = divmod(total_perturbations, hidden_size)
    schedule = np.concatenate([np.tile(np.arange(hidden_size), repeats), np.arange(remainder)]).astype(int)
    rng = np.random.default_rng(sd)
    rng.shuffle(schedule)
    return schedule

def _save_perturbed_result(result, region, total_perturbations, sd, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"connected_knockout_{region}_n{total_perturbations}_sd{sd}_bin{bin_size}.npz")
    dicts = result["dicts"]
    np.savez_compressed(
        save_path,
        Na_transition_dict=np.array(plain_transition_dict(dicts["Na"]), dtype=object),
        Nb_transition_dict=np.array(plain_transition_dict(dicts["Nb"]), dtype=object),
        Na_b_transition_dict=np.array(plain_transition_dict(dicts["Na_b"]), dtype=object),
        Nb_b_transition_dict=np.array(plain_transition_dict(dicts["Nb_b"]), dtype=object),
        Na_Nb_transition_dict=np.array(plain_transition_dict(dicts["Na_Nb"]), dtype=object),
        Na_Nb_b_transition_dict=np.array(plain_transition_dict(dicts["Na_Nb_b"]), dtype=object),
        b_transition_dict=np.array(plain_transition_dict(dicts["B"]), dtype=object),
        perturbed_route=np.array(result["route"], dtype=object),
        stage2_error_history=np.array(result["stage2_error_history"], dtype=object),
        neuron_counts=result["neuron_counts"],
        region=region,
        total_perturbations=total_perturbations,
        seed=sd,
        bin_size=bin_size
    )
    print(f"[Log] Saved {region} knockout dictionaries to {save_path}")

def run_connected_knockout_experiments(model, b_transition_dict, Na_transition_dict, Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_transition_dict, Na_Nb_b_transition_dict, total_perturbations=1000, sd=42, save_dir=None):
    if total_perturbations <= 0:
        raise ValueError("total_perturbations must be greater than zero.")
    if model.hidden_size1 != model.hidden_size2:
        raise ValueError("'both' knockout currently requires equal A and B hidden sizes.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        cache_paths = {
            region: os.path.join(
                save_dir,
                f"connected_knockout_{region}_n{total_perturbations}_sd{sd}_bin{bin_size}.npz"
            )
            for region in ("a", "b", "both")
        }

        if all(os.path.exists(path) for path in cache_paths.values()):
            results = {}
            cache_valid = True

            for region in ("a", "b", "both"):
                loaded = np.load(cache_paths[region], allow_pickle=True)

                if "stage2_error_history" not in loaded.files:
                    cache_valid = False
                    loaded.close()
                    break

                loaded_Na = loaded["Na_transition_dict"].item()
                loaded_Nb = loaded["Nb_transition_dict"].item()
                loaded_Na_b = loaded["Na_b_transition_dict"].item()
                loaded_Nb_b = loaded["Nb_b_transition_dict"].item()
                loaded_Na_Nb = loaded["Na_Nb_transition_dict"].item()
                loaded_Na_Nb_b = loaded["Na_Nb_b_transition_dict"].item()
                loaded_b = loaded["b_transition_dict"].item()

                Na_dict = defaultdict(lambda: defaultdict(int))
                Nb_dict = defaultdict(lambda: defaultdict(int))
                Na_b_dict = defaultdict(lambda: defaultdict(int))
                Nb_b_dict = defaultdict(lambda: defaultdict(int))
                Na_Nb_dict = defaultdict(lambda: defaultdict(int))
                Na_Nb_b_dict = defaultdict(lambda: defaultdict(int))
                b_transition_dict = defaultdict(lambda: defaultdict(int))

                for state, freq_dict in loaded_Na.items():
                    Na_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_Nb.items():
                    Nb_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_Na_b.items():
                    Na_b_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_Nb_b.items():
                    Nb_b_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_Na_Nb.items():
                    Na_Nb_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_Na_Nb_b.items():
                    Na_Nb_b_dict[state] = defaultdict(int, freq_dict)
                for state, freq_dict in loaded_b.items():
                    b_transition_dict[state] = defaultdict(int, freq_dict)

                results[region] = {
                    "dicts": {
                        "Na": Na_dict,
                        "Nb": Nb_dict,
                        "Na_b": Na_b_dict,
                        "Nb_b": Nb_b_dict,
                        "Na_Nb": Na_Nb_dict,
                        "Na_Nb_b": Na_Nb_b_dict,
                        "B": b_transition_dict
                    },
                    "route": loaded["perturbed_route"].tolist(),
                    "neuron_counts": loaded["neuron_counts"].copy(),
                    "stage2_error_history": loaded["stage2_error_history"].tolist()
                }

                loaded.close()

            if cache_valid:
                print(f"[Log] perturbation cache already exists for n={total_perturbations}")
                return results
            else:
                print("[Log] old perturbation cache missing Stage 2 history. Regenerating...")

    print("[Log] creating perturbation cache...")

    original_dicts = {
        "Na": Na_transition_dict,
        "Nb": Nb_transition_dict,
        "Na_b": Na_b_transition_dict,
        "Nb_b": Nb_b_transition_dict,
        "Na_Nb": Na_Nb_transition_dict,
        "Na_Nb_b": Na_Nb_b_transition_dict,
        "B": b_transition_dict
    }

    original_totals_before = {name: transition_count(dictionary) for name, dictionary in original_dicts.items()}
    model_device = next(model.parameters()).device
    model.eval()
    np.random.seed(sd)
    random.seed(sd)

    b_state_img_path_dict, all_visit_count_dict = data.image_preproccesing()
    all_b_states = list(all_visit_count_dict.keys())
    starting_b_state = all_b_states[np.random.randint(0, len(all_b_states))]

    starting_image_path = b_state_img_path_dict[starting_b_state][0]
    starting_image = load_grayscale_tensor(starting_image_path, model_device)
    neuron_schedule = _build_balanced_neuron_order(model.hidden_size1, total_perturbations, sd=sd)
    neuron_counts = np.bincount(neuron_schedule, minlength=model.hidden_size1)
    results = {}
    b_state_img_path_dict, _ = data.image_preproccesing() # list of tuples (b_state, img_path) and dict of all behavioral_states and their associated visit count

    original_Na_Nb_b_prob = f2g.convert_count_to_probability(original_dicts["Na_Nb_b"])
    original_B_prob = f2g.convert_count_to_probability(original_dicts["B"])

    for region in ("a", "b", "both"):
        updated_dicts = {name: copy_transition_dict(dictionary) for name, dictionary in original_dicts.items()}
        perturbed_route = []
        stage2_error_history = []

        h_a = torch.zeros(1, model.hidden_size1, device=model_device)
        h_b = torch.zeros(1, model.hidden_size2, device=model_device)

        with torch.no_grad():
            current_b_state, current_image = starting_b_state, starting_image
            current_Na, current_Nb, _, _, _, _ = model(current_image, current_image, h_a, h_b)
            h_a, h_b = current_Na, current_Nb

            for step, neuron_index in enumerate(neuron_schedule):
                current_B_key = f2g.behavioral_state_to_key(current_b_state)
                current_Na_key = f2g.neural_state_to_dict_key(current_Na.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                current_Nb_key = f2g.neural_state_to_dict_key(current_Nb.detach().cpu().numpy().reshape(-1), bin_size=bin_size)
                current_Na_Nb_b_key = (current_Na_key, current_Nb_key, current_B_key)
                transitions_from_cur = original_Na_Nb_b_prob.get(current_Na_Nb_b_key, {})
                frozen = len(transitions_from_cur) == 0 

                if not frozen:
                    all_next_states = list(transitions_from_cur.keys())
                    next_state_prob = list(transitions_from_cur.values())

                    _, _, next_b = random.choices(all_next_states, weights=next_state_prob, k=1)[0]
                   
                else:
                    next_B_transitions = original_B_prob.get(current_B_key, {})
                    possible_next_B = list(next_B_transitions.keys())
                    probabilities_B = list(next_B_transitions.values())

                    next_b = random.choices(possible_next_B, weights=probabilities_B, k=1)[0]


                next_b_state = next_b
                next_b_state_img_path = b_state_img_path_dict[next_b_state][0]
                next_b_state_img = Image.open(next_b_state_img_path).convert("L")
                next_b_state_img = next_b_state_img.resize((25,25))
                next_b_state_img_array = np.array(next_b_state_img) / 255.0
                next_b_state_img_tensor = torch.tensor(next_b_state_img_array, dtype=torch.float32, device=device)

                handle = model.knockout_neuron(int(neuron_index), region=region)
                try:
                    next_Na, next_Nb, _, _, _, _ = model(next_b_state_img_tensor, next_b_state_img_tensor, h_a, h_b)
                    next_Na_key = f2g.neural_state_to_dict_key(next_Na.detach().cpu().numpy().reshape(-1), bin_size)
                    next_Nb_key = f2g.neural_state_to_dict_key(next_Nb.detach().cpu().numpy().reshape(-1), bin_size)
                    next_B_key = f2g.behavioral_state_to_key(next_b_state)
                finally:
                    handle.remove()

                updated_dicts["Na"][current_Na_key][next_Na_key] += 1
                updated_dicts["Nb"][current_Nb_key][next_Nb_key] += 1
                updated_dicts["Na_b"][(current_Na_key, current_B_key)][(next_Na_key, next_B_key)] += 1
                updated_dicts["Nb_b"][(current_Nb_key, current_B_key)][(next_Nb_key, next_B_key)] += 1
                updated_dicts["Na_Nb"][(current_Na_key, current_Nb_key)][(next_Na_key, next_Nb_key)] += 1
                updated_dicts["Na_Nb_b"][(current_Na_key, current_Nb_key, current_B_key)][(next_Na_key, next_Nb_key, next_B_key)] += 1
                updated_dicts["B"][current_B_key][next_B_key] += 1

                stage2_error_history.append(pt.calc_stage2_error(frame=step + 1, B0=current_B_key, Na0=current_Na_key, Nb0=current_Nb_key, original_connected_dicts=original_dicts, updated_connected_dicts=updated_dicts))
                perturbed_route.append((current_B_key, current_Na_key, current_Nb_key, next_B_key, next_Na_key, next_Nb_key, 1))

                current_b_state = next_b_state
                current_Na, current_Nb = next_Na, next_Nb
                h_a, h_b = next_Na, next_Nb

        results[region] = {
            "dicts": updated_dicts,
            "route": perturbed_route,
            "neuron_counts": neuron_counts.copy(),
            "stage2_error_history": stage2_error_history
        }

        print(f"[Log] Completed {total_perturbations} knockouts for region={region}")
        print(f"[Log] Neuron knockout counts: {neuron_counts.tolist()}")
        print(f"[Log] Stage 2 error entries: {len(stage2_error_history)}")

        if save_dir is not None:
            _save_perturbed_result(results[region], region, total_perturbations, sd, save_dir)

    original_totals_after = {name: transition_count(dictionary) for name, dictionary in original_dicts.items()}

    if original_totals_before != original_totals_after:
        raise RuntimeError("An original transition dictionary was modified.")

    print("[Log] Original transition dictionaries were not modified.")
    return results

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
    perturbation_results = run_connected_knockout_experiments(model,
                                                                b_transition_dict,
                                                                Na_transition_dict,
                                                                Nb_transition_dict,
                                                                Na_b_transition_dict,
                                                                Nb_b_transition_dict,
                                                                Na_Nb_transition_dict,
                                                                Na_Nb_b_transition_dict,
                                                                total_perturbations=4000,
                                                                sd=42,
                                                                save_dir=os.path.join(SCRIPT_DIR, "Perturbation_Connected_Cache")
                                                            )