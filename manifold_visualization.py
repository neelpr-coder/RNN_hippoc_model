from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import os 
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import figure2_generation
from matplotlib.lines import Line2D


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.environ.get("RNN_CACHE_DIR", os.path.join(SCRIPT_DIR, "RNN_cache"))

def extract_neural_path_from_route(route_sequence):
    if len(route_sequence) == 0:
        return []

    neural_path = [route_sequence[0][1]]

    for B0, N0, B1, N1, observed_count in route_sequence:
        neural_path.append(N1)

    return neural_path

def extract_behavioral_path_from_route(route_sequence):
    if len(route_sequence) == 0:
        return []

    behavioral_path = [route_sequence[0][0]]

    for B0, N0, B1, N1, observed_count in route_sequence:
        behavioral_path.append(B1)

    return behavioral_path

def load_cache(min_visits=100, max_visits=151, sd=42, bin_size=0.3):
    """
    Checks if the cache directory exists and loads cached data for manifold visualization.  
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Cache directory at {CACHE_DIR} does not exist")
        return
    else:
        print(f"Cache directory already exists at {CACHE_DIR}")
        pair_dir = os.path.join(CACHE_DIR, "pair_transition")
        b_state_dir = os.path.join(CACHE_DIR, "b_state")
        n_state_dir = os.path.join(CACHE_DIR, "n_state")

        n_cache_path = os.path.join(n_state_dir, f"n_state_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz")
        b_cache_path = os.path.join(b_state_dir, f"b_state_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz")
        pair_cache_path = os.path.join(pair_dir, f"pair_transition_min{min_visits}_max{max_visits}_sd{sd}_bin_size{bin_size}.npz")
        
        if os.path.exists(n_cache_path) and os.path.exists(b_cache_path) and os.path.exists(pair_cache_path):
            n_state_dict = defaultdict(lambda: defaultdict(int))
            b_state_dict = defaultdict(lambda: defaultdict(int))
            pair_transition_dict = defaultdict(lambda: defaultdict(int))

            loaded_n = np.load(n_cache_path, allow_pickle=True)
            loaded_b = np.load(b_cache_path, allow_pickle=True)
            loaded_pair = np.load(pair_cache_path, allow_pickle=True)

            loaded_n_dict = loaded_n["neural_state_dict"].item()
            loaded_b_dict = loaded_b["b_transition_dict"].item()
            loaded_pair_dict = loaded_pair["pair_dict"].item()
            
            for state, freq_dict in loaded_n_dict.items():
                n_state_dict[state] = defaultdict(int, freq_dict)

            for state, freq_dict in loaded_b_dict.items():
                b_state_dict[state] = defaultdict(int, freq_dict)

            for state, freq_dict in loaded_pair_dict.items():
                pair_transition_dict[state] = defaultdict(int, freq_dict)

            return n_state_dict, b_state_dict, pair_transition_dict
        raise FileNotFoundError("One or more cache files are missing.")
    
def behavioral_state_to_vector(b_state):
        x, y, heading_idx = b_state
        theta = float(heading_idx) * (2 * np.pi / 4)
        return np.array([x, y, np.cos(theta), np.sin(theta)], dtype=float)

def convert_dict_to_matrix(cached_dict, dict_type=None):
    """
    Converts a dictionary of neural states into a matrix format suitable for manifold learning.
    """
    def paired_state_to_vector(pair_state):
        (B0, N0) = pair_state
        b_vector = behavioral_state_to_vector(B0)
        n_vector = np.array(N0, dtype=float)
        return np.concatenate([b_vector, n_vector])
    
    if dict_type == None:
        raise ValueError("dict_type cannot be None. Please specify 'neural_state', 'behavioral_state', or 'paired_transition'.")
    if dict_type == 'neural_state':
        neural_key_set = set()
        for key1, inner_dict in cached_dict.items():
            neural_key_set.add(key1)
            for key2 in inner_dict.keys():
                neural_key_set.add(key2)
        neural_keys = list(neural_key_set)
        matrix = np.array(neural_keys, dtype=float)
        return matrix, neural_keys
    elif dict_type == 'behavioral_state':
        b_set = set()
        for key1, inner_dict in cached_dict.items():
            b_set.add(key1)
            for key2 in inner_dict.keys():
                b_set.add(key2)
        
        b_list = list(b_set)
        b_matrix = np.array([behavioral_state_to_vector(b) for b in b_set], dtype=float)
        return b_matrix, b_list
    elif dict_type == 'paired_transition':
        paired_set = set()
        for key1, inner_dict in cached_dict.items():
            paired_set.add(key1)
            for key2 in inner_dict.keys():
                paired_set.add(key2)
            
        paired_list = list(paired_set)
        paired_matrix = np.array([paired_state_to_vector(p) for p in paired_set], dtype=float)
        return paired_matrix, paired_list
    else:
        raise ValueError("Unsupported dict_type. Please use 'neural_state', 'behavioral_state', or 'paired_transition'.")
    

def perform_manifold_learning(matrix, method='tsne', n_components=3, random_state=42):
    if method == 'tsne':
        manifold = TSNE(n_components=n_components, random_state=random_state)
    elif method == 'pca':
        manifold = PCA(n_components=n_components)
    elif method == 'umap':
        manifold = UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Unsupported method. Please use 'tsne', 'pca', or 'umap'.")

    return manifold.fit_transform(matrix)


def make_hashable_state(state):

    if isinstance(state, np.ndarray):
        return make_hashable_state(state.tolist())

    if isinstance(state, list):
        return tuple(make_hashable_state(x) for x in state)

    if isinstance(state, tuple):
        return tuple(make_hashable_state(x) for x in state)

    if isinstance(state, (np.integer, int)):
        return int(state)

    if isinstance(state, (np.floating, float)):
        return float(state)

    return state


def extract_state_path_from_route(route_sequence, state_type):
    """
    Extracts the actual animation route path for the selected state space.
    """

    if state_type == "neural states":
        return extract_neural_path_from_route(route_sequence)

    elif state_type == "behavioral states":
        return extract_behavioral_path_from_route(route_sequence)

    elif state_type == "joint states":
        if len(route_sequence) == 0:
            return []

        state_path = [(route_sequence[0][0], route_sequence[0][1])]

        for B0, N0, B1, N1, observed_count in route_sequence:
            state_path.append((B1, N1))

        return state_path

    else:
        raise ValueError(
            "Unsupported state_type. Please use 'neural states', "
            "'behavioral states', or 'joint states'."
        )


def get_top3_current_and_targets(frame_info, state_type):
    """
    Uses the saved top-3 transitions from figure2_generation.py.
    """

    if state_type == "neural states":
        current_state = frame_info["N0"]
        top3_items = frame_info["top3_n"]

    elif state_type == "behavioral states":
        current_state = frame_info["B0"]
        top3_items = frame_info["top3_b"]

    elif state_type == "joint states":
        current_state = (frame_info["B0"], frame_info["N0"])
        top3_items = frame_info["top3_pair"]

    else:
        raise ValueError(
            "Unsupported state_type. Please use 'neural states', "
            "'behavioral states', or 'joint states'."
        )

    target_states = [target_state for target_state, count in top3_items]

    return current_state, target_states

def get_actual_next_state_from_frame(frame_info, state_type):
    """
    Returns the actual next state from the predetermined route for this frame.
    """

    if state_type == "neural states":
        return frame_info["N1"]

    elif state_type == "behavioral states":
        return frame_info["B1"]

    elif state_type == "joint states":
        return (frame_info["B1"], frame_info["N1"])

    else:
        raise ValueError(
            "Unsupported state_type. Please use 'neural states', "
            "'behavioral states', or 'joint states'."
        )

def plot_manifold_raw_two_route(
    manifold,
    manifold_type,
    state_keys,
    state_type,
    route_sequence,
    top3_route_history,
    save_prefix=None,
    show_all_points=True,
    make_full_plot=True,
    make_zoomed_plot=True,
    zoom_padding_fraction=0.18,
    zoom_percentile=5,
    visual_offset_fraction=0.018,
    full_background_alpha=0.20,
    zoom_background_alpha=0.08,
    full_background_point_size=9,
    zoom_background_point_size=6
):
    """
    Plot actual route vs highest-probability predicted route.
    Zoom and normal view
    """

    actual_state_path = extract_state_path_from_route(route_sequence, state_type)

    if len(actual_state_path) == 0:
        raise ValueError("Actual route path is empty.")


    predicted_state_path = [actual_state_path[0]]

    frame_alignment_mismatches = 0

    for frame_idx, frame_info in enumerate(top3_route_history):
        current_state, target_states = get_top3_current_and_targets(
            frame_info,
            state_type
        )

        if len(target_states) == 0:
            raise ValueError(f"No top-3 targets found for frame {frame_idx}.")

        top1_target = target_states[0]
        predicted_state_path.append(top1_target)

        # Sanity check: saved current state should match actual route state
        if frame_idx < len(actual_state_path):
            actual_current_state = actual_state_path[frame_idx]

            if make_hashable_state(current_state) != make_hashable_state(actual_current_state):
                frame_alignment_mismatches += 1

    if frame_alignment_mismatches > 0:
        print(
            f"Warning: {frame_alignment_mismatches} frame(s) had current states in "
            f"top3_route_history that did not match the actual route state."
        )

    hashable_state_keys = [
        make_hashable_state(key)
        for key in state_keys
    ]

    key_to_idx = {
        key: i
        for i, key in enumerate(hashable_state_keys)
    }

    actual_indices = []
    missing_actual = 0

    for state in actual_state_path:
        h_state = make_hashable_state(state)

        if h_state in key_to_idx:
            actual_indices.append(key_to_idx[h_state])
        else:
            missing_actual += 1

    if len(actual_indices) == 0:
        raise ValueError(f"No actual {state_type} route states matched manifold keys.")


    predicted_indices = []
    missing_predicted = 0

    for state in predicted_state_path:
        h_state = make_hashable_state(state)

        if h_state in key_to_idx:
            predicted_indices.append(key_to_idx[h_state])
        else:
            missing_predicted += 1

    if len(predicted_indices) == 0:
        raise ValueError(f"No predicted {state_type} route states matched manifold keys.")

    common_len = min(len(actual_indices), len(predicted_indices))

    actual_indices = actual_indices[:common_len]
    predicted_indices = predicted_indices[:common_len]

    actual_state_path = actual_state_path[:common_len]
    predicted_state_path = predicted_state_path[:common_len]

    actual_coords = manifold[actual_indices]
    predicted_coords = manifold[predicted_indices]

    matching_frames = []
    actual_only_frames = []
    predicted_only_frames = []

    for frame_idx in range(1, common_len):
        actual_state = make_hashable_state(actual_state_path[frame_idx])
        predicted_state = make_hashable_state(predicted_state_path[frame_idx])

        if actual_state == predicted_state:
            matching_frames.append(frame_idx)
        else:
            actual_only_frames.append(frame_idx)
            predicted_only_frames.append(frame_idx)

    transition_count = common_len - 1

    actual_hashable = [
        make_hashable_state(state)
        for state in actual_state_path
    ]

    predicted_hashable = [
        make_hashable_state(state)
        for state in predicted_state_path
    ]

    actual_counter = Counter(actual_hashable)
    predicted_counter = Counter(predicted_hashable)

    print("\n===== Top-1 predicted route diagnostics =====")
    print(f"Manifold type: {manifold_type}")
    print(f"State type: {state_type}")
    print(f"Actual route states found: {len(actual_indices)}/{len(actual_state_path)}")
    print(f"Missing actual route states: {missing_actual}")
    print(f"Predicted route states found: {len(predicted_indices)}/{len(predicted_state_path)}")
    print(f"Missing predicted route states: {missing_predicted}")
    print(
    f"Transition-level matches between actual and top-1 predicted route: "
    f"{len(matching_frames)}/{transition_count}"
    )
    print(f"Unique actual route states: {len(set(actual_hashable))}")
    print(f"Unique predicted route states: {len(set(predicted_hashable))}")
    print("\nMost common predicted states:")
    for state, count in predicted_counter.most_common(10):
        print(f"  count={count}: {state}")
    print("\nTop-1 predicted state by transition frame:")
    for frame_idx in range(1, common_len):
        actual_state = make_hashable_state(actual_state_path[frame_idx])
        predicted_state = make_hashable_state(predicted_state_path[frame_idx])
        match_status = "MATCH" if actual_state == predicted_state else "NO MATCH"

        print(
            f"  transition {frame_idx:02d}: "
            f"predicted={predicted_state} | actual={actual_state} | {match_status}"
        )
    print("============================================\n")

    data_span = manifold.max(axis=0) - manifold.min(axis=0)
    offset_vector = np.array([
        visual_offset_fraction * data_span[0],
        visual_offset_fraction * data_span[1],
        0.0
    ])

    actual_plot_coords = actual_coords - offset_vector
    predicted_plot_coords = predicted_coords + offset_vector


    gold_coords = actual_coords[matching_frames] if len(matching_frames) > 0 else None

    def draw_route_plot(
        ax,
        zoom_to_cluster=False,
        background_alpha=0.20,
        background_point_size=9,
        title_suffix=""
    ):
        ax.set_proj_type("ortho")

        if show_all_points:
            ax.scatter(
                manifold[:, 0],
                manifold[:, 1],
                manifold[:, 2],
                s=background_point_size,
                color="black",
                alpha=background_alpha,
                label=f"All {state_type}",
                zorder=1,
                depthshade=False
            )

        ax.plot(
            actual_plot_coords[:, 0],
            actual_plot_coords[:, 1],
            actual_plot_coords[:, 2],
            color="crimson",
            linewidth=2.2,
            alpha=0.72,
            label="Actual predetermined route",
            zorder=6
        )

        ax.plot(
            predicted_plot_coords[:, 0],
            predicted_plot_coords[:, 1],
            predicted_plot_coords[:, 2],
            color="deepskyblue",
            linewidth=2.4,
            linestyle="--",
            alpha=1.0,
            label="Highest-probability predicted route",
            zorder=7
        )

        if len(actual_only_frames) > 0:
            actual_only_coords = actual_plot_coords[actual_only_frames]

            ax.scatter(
                actual_only_coords[:, 0],
                actual_only_coords[:, 1],
                actual_only_coords[:, 2],
                s=48,
                color="crimson",
                marker="o",
                alpha=0.98,
                label="Actual route only",
                zorder=10,
                depthshade=False
            )

        if len(predicted_only_frames) > 0:
            predicted_only_coords = predicted_plot_coords[predicted_only_frames]

            ax.scatter(
                predicted_only_coords[:, 0],
                predicted_only_coords[:, 1],
                predicted_only_coords[:, 2],
                s=70,
                color="deepskyblue",
                marker="^",
                alpha=1.0,
                label="Predicted route only",
                zorder=11,
                depthshade=False
            )

        if gold_coords is not None:
            ax.scatter(
                gold_coords[:, 0],
                gold_coords[:, 1],
                gold_coords[:, 2],
                s=105,
                color="gold",
                edgecolor="black",
                linewidth=0.9,
                alpha=1.0,
                label=f"Same next state ({len(matching_frames)}/{transition_count})",
                zorder=13,
                depthshade=False
            )

        ax.set_title(
            f"3D {manifold_type} Projection of {state_type}\n"
            f"Actual Route vs Highest-Probability Predicted Route{title_suffix}"
        )
        ax.set_xlabel(f"{manifold_type} 1")
        ax.set_ylabel(f"{manifold_type} 2")
        ax.set_zlabel(f"{manifold_type} 3")

        if zoom_to_cluster:
            zoom_coords = np.vstack([
                actual_plot_coords,
                predicted_plot_coords
            ])

            if gold_coords is not None:
                zoom_coords = np.vstack([
                    zoom_coords,
                    gold_coords
                ])

            x_min, y_min, z_min = zoom_coords.min(axis=0)
            x_max, y_max, z_max = zoom_coords.max(axis=0)

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            x_pad = zoom_padding_fraction * max(x_range, 1e-8)
            y_pad = zoom_padding_fraction * max(y_range, 1e-8)
            z_pad = zoom_padding_fraction * max(z_range, 1e-8)

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_zlim(z_min - z_pad, z_max + z_pad)

        if manifold_type == "PCA":
            ax.set_box_aspect((1.35, 1.15, 0.85))
        else:
            ax.set_box_aspect((1.25, 1.15, 0.95))

        ax.legend(loc="upper right")

    if make_full_plot:
        fig_full = plt.figure(figsize=(15, 13))
        ax_full = fig_full.add_subplot(111, projection="3d")

        draw_route_plot(
            ax_full,
            zoom_to_cluster=False,
            background_alpha=full_background_alpha,
            background_point_size=full_background_point_size,
            title_suffix=" (Full View)"
        )

        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92)

        if save_prefix is not None:
            plt.savefig(
                f"{save_prefix}_{manifold_type.lower().replace('-', '').replace(' ', '_')}_full.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    if make_zoomed_plot:
        fig_zoom = plt.figure(figsize=(15, 13))
        ax_zoom = fig_zoom.add_subplot(111, projection="3d")

        draw_route_plot(
            ax_zoom,
            zoom_to_cluster=True,
            background_alpha=zoom_background_alpha,
            background_point_size=zoom_background_point_size,
            title_suffix=" (Cluster Zoom)"
        )

        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92)

        if save_prefix is not None:
            plt.savefig(
                f"{save_prefix}_{manifold_type.lower().replace('-', '').replace(' ', '_')}_cluster_zoom.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()



def build_top1_predicted_path_from_route(route_sequence,b_transition_dict,neural_state_dict,pair_transition_dict,state_type):
    """
    Builds the top-1 highest-probability predicted route from the dictionary. Time frame aligned. 
    """

    actual_state_path = extract_state_path_from_route(route_sequence, state_type)

    if len(actual_state_path) == 0:
        raise ValueError("Actual state path is empty.")

    predicted_state_path = [actual_state_path[0]]

    for frame_idx, (B0, N0, B1, N1, observed_count) in enumerate(route_sequence):

        if state_type == "neural states":
            next_dict = neural_state_dict.get(N0, {})

        elif state_type == "behavioral states":
            next_dict = b_transition_dict.get(B0, {})

        elif state_type == "joint states":
            next_dict = pair_transition_dict.get((B0, N0), {})

        else:
            raise ValueError(
                "Unsupported state_type. Use 'neural states', "
                "'behavioral states', or 'joint states'."
            )

        if len(next_dict) == 0:
            raise ValueError(
                f"No predicted transitions found at frame {frame_idx + 1} "
                f"for state_type={state_type}."
            )

        # Highest count = highest transition probability.
        top1_target = max(next_dict.items(), key=lambda item: item[1])[0]

        predicted_state_path.append(top1_target)

    return predicted_state_path



REDUCTION_LABELS = {
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pca": "PCA"
}


def get_dictionary_plot_config(dictionary_name):
    """
    Maps a simple dictionary name to the state_type and dict_type
    """

    if dictionary_name == "neural":
        return {
            "state_type": "neural states",
            "dict_type": "neural_state",
            "display_name": "Neural"
        }

    elif dictionary_name == "behavioral":
        return {
            "state_type": "behavioral states",
            "dict_type": "behavioral_state",
            "display_name": "Behavioral"
        }

    elif dictionary_name == "paired":
        return {
            "state_type": "joint states",
            "dict_type": "paired_transition",
            "display_name": "Paired"
        }

    else:
        raise ValueError("dictionary_name must be 'neural', 'behavioral', or 'paired'.")


def select_transition_dict(dictionary_name,neural_state_dict,behavioral_state_dict,pair_transition_dict):
    """
    Selects the dictionary that should be embedded/plotted.
    """

    if dictionary_name == "neural":
        return neural_state_dict

    elif dictionary_name == "behavioral":
        return behavioral_state_dict

    elif dictionary_name == "paired":
        return pair_transition_dict

    else:
        raise ValueError("dictionary_name must be 'neural', 'behavioral', or 'paired'.")


def extract_error_values(error_history, error_key="raw_error"):
    """
    Converts error_history from evaluate_error_history_on_fixed_route(...) into an array
    """

    values = []

    for item in error_history:
        if isinstance(item, dict):
            if error_key in item:
                values.append(item[error_key])
            elif "error" in item:
                values.append(item["error"])
            else:
                raise KeyError(
                    f"Could not find '{error_key}' or 'error' in error_history item: {item}"
                )
        else:
            values.append(float(item))

    return np.array(values, dtype=float)


def path_to_coords(state_path, state_keys, manifold):
    """
    Converts a state path into manifold coordinates.

    Missing states are represented by NaN rows so matplotlib breaks the line cleanly.
    """

    key_to_idx = {
        make_hashable_state(key): idx
        for idx, key in enumerate(state_keys)
    }

    coords = np.full((len(state_path), 3), np.nan, dtype=float)
    missing_count = 0

    for i, state in enumerate(state_path):
        h_state = make_hashable_state(state)

        if h_state in key_to_idx:
            coords[i] = manifold[key_to_idx[h_state]]
        else:
            missing_count += 1

    return coords, missing_count


def plot_manifold_panel_on_axis(
    ax,
    manifold,
    state_keys,
    route_sequence,
    neural_state_dict,
    behavioral_state_dict,
    pair_transition_dict,
    dictionary_name,
    reduction_method,
    min_visits,
    elev=24,
    azim=-60
):
    """
    Draws one manifold panel inside a collage.
    """

    config = get_dictionary_plot_config(dictionary_name)
    state_type = config["state_type"]
    display_name = config["display_name"]
    reduction_label = REDUCTION_LABELS[reduction_method]

    actual_state_path = extract_state_path_from_route(route_sequence, state_type=state_type)

    predicted_state_path = build_top1_predicted_path_from_route(
        route_sequence=route_sequence,
        b_transition_dict=behavioral_state_dict,
        neural_state_dict=neural_state_dict,
        pair_transition_dict=pair_transition_dict,
        state_type=state_type
    )

    actual_coords, missing_actual = path_to_coords(actual_state_path, state_keys, manifold)
    predicted_coords, missing_predicted = path_to_coords(predicted_state_path, state_keys, manifold)
    n_frames = min(len(actual_state_path), len(predicted_state_path))
    actual_hashable = [make_hashable_state(state) for state in actual_state_path[:n_frames]]
    predicted_hashable = [make_hashable_state(state) for state in predicted_state_path[:n_frames]]

    actual_valid = ~np.isnan(actual_coords[:n_frames]).any(axis=1)
    predicted_valid = ~np.isnan(predicted_coords[:n_frames]).any(axis=1)

    same_frame_mask = np.array([actual_hashable[i] == predicted_hashable[i] for i in range(n_frames)], dtype=bool)

    same_frame_mask = same_frame_mask & actual_valid & predicted_valid

    actual_only_mask = actual_valid & (~same_frame_mask)
    predicted_only_mask = predicted_valid & (~same_frame_mask)

    same_count = int(np.sum(same_frame_mask))

    all_state_color = "gray"
    actual_line_color = "crimson"
    predicted_line_color = "deepskyblue"
    actual_point_color = "crimson"
    predicted_point_color = "deepskyblue"
    same_point_color = "gold"

    ax.set_proj_type("ortho")
    ax.scatter(manifold[:, 0],manifold[:, 1],manifold[:, 2],s=8,color=all_state_color,alpha=0.20,depthshade=False)
    ax.plot(actual_coords[:, 0],actual_coords[:, 1],actual_coords[:, 2],color=actual_line_color,linewidth=2.0,alpha=0.80)
    ax.plot(predicted_coords[:, 0],predicted_coords[:, 1],predicted_coords[:, 2],color=predicted_line_color,linewidth=2.0,linestyle="--",alpha=0.95)

    actual_only_coords = actual_coords[:n_frames][actual_only_mask]

    if len(actual_only_coords) > 0:
        ax.scatter(actual_only_coords[:, 0],actual_only_coords[:, 1],actual_only_coords[:, 2],s=42,color=actual_point_color,marker="o",alpha=0.95,depthshade=False,zorder=10)

    predicted_only_coords = predicted_coords[:n_frames][predicted_only_mask]

    if len(predicted_only_coords) > 0:
        ax.scatter(predicted_only_coords[:, 0],predicted_only_coords[:, 1],predicted_only_coords[:, 2],s=58,color=predicted_point_color,marker="^",alpha=0.95,depthshade=False,zorder=11)

    same_coords = actual_coords[:n_frames][same_frame_mask]

    if len(same_coords) > 0:
        ax.scatter(same_coords[:, 0],same_coords[:, 1],same_coords[:, 2],s=82,color=same_point_color,edgecolor="black",linewidth=0.8,marker="o",alpha=0.98,depthshade=False,zorder=12)

    ax.set_title(f"{reduction_label}, min visits = {min_visits}",fontsize=16,pad=16)

    ax.set_xlabel(f"{reduction_label} 1", fontsize=10, labelpad=7)
    ax.set_ylabel(f"{reduction_label} 2", fontsize=10, labelpad=7)
    ax.set_zlabel(f"{reduction_label} 3", fontsize=10, labelpad=7)

    ax.view_init(elev=elev, azim=azim)

    mins = manifold.min(axis=0)
    maxs = manifold.max(axis=0)
    ranges = maxs - mins
    pad = 0.08 * np.maximum(ranges, 1e-8)

    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    if reduction_label == "PCA":
        ax.set_box_aspect((1.35, 1.15, 0.85))
    else:
        ax.set_box_aspect((1.2, 1.1, 0.95))

    print(
        f"{display_name} | {reduction_label} | min visits {min_visits}: "
        f"same-frame matches = {same_count}/{n_frames}, "
        f"missing actual = {missing_actual}, missing predicted = {missing_predicted}"
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=all_state_color,
            alpha=0.35,
            markersize=5,
            label=f"All {display_name.lower()} states"
        ),
        Line2D(
            [0],
            [0],
            color=actual_line_color,
            linewidth=2.0,
            label="Actual predetermined route"
        ),
        Line2D(
            [0],
            [0],
            color=predicted_line_color,
            linewidth=2.0,
            linestyle="--",
            label="Highest-probability predicted route"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=actual_point_color,
            markeredgecolor=actual_point_color,
            markersize=7,
            label="Actual route only"
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=predicted_point_color,
            markeredgecolor=predicted_point_color,
            markersize=8,
            label="Predicted route only"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=same_point_color,
            markeredgecolor="black",
            markersize=9,
            label=f"Same state at same frame ({same_count}/{n_frames})"
        )
    ]

    return legend_handles


def plot_error_panel_on_axis(
    ax,
    error_values,
    min_visits,
    y_limits=None
):
    """
    Draws one  error-vs-time-frame panel below the manifold.
    """

    frames = np.arange(1, len(error_values) + 1)

    ax.plot(frames, error_values, color="black", linewidth=1.2, marker="o", markersize=2.4)
    ax.set_title(f"Error vs Time Frame, min visits = {min_visits}", fontsize=10)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("Error", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.25)

    if y_limits is not None:
        ax.set_ylim(y_limits)


def create_reduction_epoch_collage(
    dictionary_name,
    reduction_method,
    min_visits_list=(50, 75, 100),
    max_visits=151,
    sd=42,
    bin_size=0.5,
    route_min_visits=100,
    error_key="raw_error",
    save_path=None,
    show=True
):
    """
    Creates one collage.
    """

    config = get_dictionary_plot_config(dictionary_name)
    state_type = config["state_type"]
    dict_type = config["dict_type"]
    display_name = config["display_name"]
    reduction_label = REDUCTION_LABELS[reduction_method]

    route_sequence = np.load(os.path.join(SCRIPT_DIR, f"route_sequence_min{route_min_visits}.npy"), allow_pickle=True)

    panel_data = []
    all_error_values = []

    for min_visits in min_visits_list:
        neural_state_dict, behavioral_state_dict, pair_transition_dict = load_cache(min_visits=min_visits, max_visits=max_visits, sd=sd, bin_size=bin_size)
        transition_dict = select_transition_dict(dictionary_name, neural_state_dict, behavioral_state_dict, pair_transition_dict)
        matrix, state_keys = convert_dict_to_matrix(transition_dict, dict_type=dict_type)

        print(f"{display_name} {reduction_label}, min visits {min_visits}:vmatrix shape = {matrix.shape}")

        manifold = perform_manifold_learning(matrix, method=reduction_method, n_components=3, random_state=42)

        error_history = figure2_generation.evaluate_error_history_on_fixed_route(
            route_sequence=route_sequence,
            b_transition_dict=behavioral_state_dict,
            neural_state_dict=neural_state_dict,
            pair_transition_dict=pair_transition_dict,
            top_k=3,
            apply_top3_filter=False
        )

        error_values = extract_error_values(error_history, error_key=error_key)
        all_error_values.append(error_values)

        panel_data.append({
            "min_visits": min_visits,
            "neural_state_dict": neural_state_dict,
            "behavioral_state_dict": behavioral_state_dict,
            "pair_transition_dict": pair_transition_dict,
            "state_keys": state_keys,
            "manifold": manifold,
            "error_values": error_values
        })

    concatenated_errors = np.concatenate(all_error_values)

    y_min = float(np.min(concatenated_errors))
    y_max = float(np.max(concatenated_errors))

    if np.isclose(y_min, y_max):
        y_pad = 1e-3
    else:
        y_pad = 0.12 * (y_max - y_min)

    y_limits = (y_min - y_pad, y_max + y_pad)
    fig = plt.figure(figsize=(22, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[4.7, 1.35], hspace=0.18, wspace=0.16)

    shared_legend_handles = None

    for col, data in enumerate(panel_data):
        min_visits = data["min_visits"]

        ax_manifold = fig.add_subplot(grid[0, col], projection="3d")

        legend_handles = plot_manifold_panel_on_axis(
            ax=ax_manifold,
            manifold=data["manifold"],
            state_keys=data["state_keys"],
            route_sequence=route_sequence,
            neural_state_dict=data["neural_state_dict"],
            behavioral_state_dict=data["behavioral_state_dict"],
            pair_transition_dict=data["pair_transition_dict"],
            dictionary_name=dictionary_name,
            reduction_method=reduction_method,
            min_visits=min_visits
        )

        if shared_legend_handles is None:
            shared_legend_handles = legend_handles

        ax_error = fig.add_subplot(grid[1, col])

        plot_error_panel_on_axis(ax=ax_error, error_values=data["error_values"], min_visits=min_visits, y_limits=y_limits)

    fig.suptitle(f"{display_name} dictionary trajectories across epochs — {reduction_label}", fontsize=20, y=0.96)
    fig.legend(handles=shared_legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=6, fontsize=10, frameon=True)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.08)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved collage: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_all_dictionary_collages(
    dictionary_names=("neural", "behavioral", "paired"),
    reduction_methods=("tsne", "umap", "pca"),
    min_visits_list=(50, 75, 100),
    max_visits=151,
    sd=42,
    bin_size=0.5,
    route_min_visits=100,
    error_key="raw_error",
    save_dir=None,
    show=True
):
    """
    Generates all collages.
    3 dictionaries x 3 reduction techniques
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for dictionary_name in dictionary_names:
        for reduction_method in reduction_methods:
            save_path = None

            if save_dir is not None:
                save_path = os.path.join(
                    save_dir,
                    f"{dictionary_name}_{reduction_method}_epochs_50_75_100_clean_collage.png"
                )

            create_reduction_epoch_collage(
                dictionary_name=dictionary_name,
                reduction_method=reduction_method,
                min_visits_list=min_visits_list,
                max_visits=max_visits,
                sd=sd,
                bin_size=bin_size,
                route_min_visits=route_min_visits,
                error_key=error_key,
                save_path=save_path,
                show=show
            )



if __name__ == "__main__":
    RUN_COLLAGES = False
    RUN_SINGLE_PLOTS_WITHOUT_INSETS = True


    MIN_VISITS_LIST = (50, 75, 100)
    REDUCTION_METHODS = ("tsne", "umap", "pca")
    DICTIONARY_NAMES = ("neural", "behavioral", "paired")

    MAX_VISITS = 151
    SD = 42
    BIN_SIZE = 0.3

  
    ROUTE_MIN_VISITS = 100

    ERROR_KEY = "raw_error"

    COLLAGE_SAVE_DIR = None #os.path.join(SCRIPT_DIR, "clean_epoch_collages")

    if RUN_COLLAGES:
        generate_all_dictionary_collages(
            dictionary_names=DICTIONARY_NAMES,
            reduction_methods=REDUCTION_METHODS,
            min_visits_list=MIN_VISITS_LIST,
            max_visits=MAX_VISITS,
            sd=SD,
            bin_size=BIN_SIZE,
            route_min_visits=ROUTE_MIN_VISITS,
            error_key=ERROR_KEY,
            save_dir=COLLAGE_SAVE_DIR,
            show=True
        )

    if RUN_SINGLE_PLOTS_WITHOUT_INSETS:
        SINGLE_DICTIONARY_TO_PLOT = "neural"
        SINGLE_MIN_VISITS = 100

        config = get_dictionary_plot_config(SINGLE_DICTIONARY_TO_PLOT)

        state_type = config["state_type"]
        dict_type = config["dict_type"]

        neural_state_dict, behavioral_state_dict, pair_transition_dict = load_cache(
            min_visits=SINGLE_MIN_VISITS,
            max_visits=MAX_VISITS,
            sd=SD,
            bin_size=BIN_SIZE
        )

        transition_dict = select_transition_dict(
            SINGLE_DICTIONARY_TO_PLOT,
            neural_state_dict,
            behavioral_state_dict,
            pair_transition_dict
        )

        matrix, state_keys = convert_dict_to_matrix(
            transition_dict,
            dict_type=dict_type
        )

        route = np.load(
            os.path.join(SCRIPT_DIR, f"route_sequence_min{ROUTE_MIN_VISITS}.npy"),
            allow_pickle=True
        )

        top3_route_history = np.load(
            os.path.join(SCRIPT_DIR, f"top3_route_transitions_min{ROUTE_MIN_VISITS}.npy"),
            allow_pickle=True
        ).tolist()

        for method in REDUCTION_METHODS:
            method_label = REDUCTION_LABELS[method]

            manifold = perform_manifold_learning(
                matrix,
                method=method,
                n_components=3,
                random_state=42
            )

            plot_manifold_raw_two_route(
                manifold,
                manifold_type=method_label,
                state_keys=state_keys,
                state_type=state_type,
                route_sequence=route,
                top3_route_history=top3_route_history,
                save_prefix=None,
                show_all_points=True,
                make_full_plot=True,
                make_zoomed_plot=False,
                zoom_padding_fraction=0.12,
                zoom_percentile=0,
                visual_offset_fraction=0.018,
                full_background_alpha=0.20,
                zoom_background_alpha=0.08,
                full_background_point_size=9,
                zoom_background_point_size=6
            )
