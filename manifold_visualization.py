from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import os 
from collections import defaultdict
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, label
#from skimage.measure import marching_cubes
#from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
import matplotlib.tri as mtri


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

def load_cache(min_visits=100, max_visits=151, sd=42, bin_size=0.5):
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
    

def convert_dict_to_matrix(cached_dict, dict_type=None):
    """
    Converts a dictionary of neural states into a matrix format suitable for manifold learning.
    """
    def behavioral_state_to_vector(b_state):
        x, y, heading_idx = b_state
        theta = float(heading_idx) * (2 * np.pi / 4)
        return np.array([x, y, np.cos(theta), np.sin(theta)], dtype=float)

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

    These are the exact top 3 shown in the animation panels:
        top3_b    = behavioral transition lookup panel
        top3_n    = neural transition lookup panel
        top3_pair = actual pair transition lookup panel
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

    # top3_items format: [(target_state, count), ...]
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

def plot_manifold_raw(
    manifold,
    manifold_type,
    state_keys,
    state_type,
    route_sequence,
    top3_route_history=None,
    save_path=None,
    show_route_line=True,
    show_all_points=True,
    show_top3=True
):
    """
    Raw 3D manifold plot.

    Blue   = all states
    Red    = actual route states
    Green  = top-3 predicted target states that are NOT the actual next state
             for that specific frame
    Orange = top-3 predicted target states that ARE the actual next state
             for that specific frame

    Important:
        Orange is now frame-by-frame.
        It does NOT mean the state appears somewhere in both the route and predictions.
        It means that at a given frame, the actual next state was one of the top-3
        predicted targets from the dictionary.
    """

    # -------------------------
    # Build actual route path
    # -------------------------
    state_path = extract_state_path_from_route(route_sequence, state_type)

    hashable_state_keys = [
        make_hashable_state(key)
        for key in state_keys
    ]

    key_to_idx = {
        key: i
        for i, key in enumerate(hashable_state_keys)
    }

    # -------------------------
    # Match actual route states to manifold indices
    # -------------------------
    route_indices = []
    missing_route = 0

    for state in state_path:
        h_state = make_hashable_state(state)

        if h_state in key_to_idx:
            route_indices.append(key_to_idx[h_state])
        else:
            missing_route += 1

    if len(route_indices) == 0:
        raise ValueError(f"No route {state_type} matched manifold keys.")

    route_coords = manifold[route_indices]
    route_index_set = set(route_indices)

    print(f"Exact route {state_type} found: {len(route_indices)}/{len(state_path)}")
    print(f"Missing route {state_type}: {missing_route}")
    print(f"Unique route {state_type}: {len(route_index_set)}")

    # -------------------------
    # Match top-3 predicted targets frame-by-frame
    # -------------------------
    top3_match_indices = set()
    top3_nonmatch_indices = set()

    missing_top3_targets = 0
    total_top3_targets = 0

    frames_actual_next_in_top3 = 0
    total_frames_checked = 0

    if show_top3 and top3_route_history is not None:
        for frame_info in top3_route_history:
            _, target_states = get_top3_current_and_targets(
                frame_info,
                state_type
            )

            actual_next_state = get_actual_next_state_from_frame(
                frame_info,
                state_type
            )

            h_actual_next = make_hashable_state(actual_next_state)

            actual_next_found_this_frame = False
            total_frames_checked += 1

            for target_state in target_states:
                total_top3_targets += 1
                h_target = make_hashable_state(target_state)

                if h_target in key_to_idx:
                    target_idx = key_to_idx[h_target]

                    if h_target == h_actual_next:
                        top3_match_indices.add(target_idx)
                        actual_next_found_this_frame = True
                    else:
                        top3_nonmatch_indices.add(target_idx)
                else:
                    missing_top3_targets += 1

            if actual_next_found_this_frame:
                frames_actual_next_in_top3 += 1

        print(f"Total top-3 predicted targets checked: {total_top3_targets}")
        print(f"Missing top-3 predicted target states: {missing_top3_targets}")
        print(
            f"Frames where actual next state is in top 3: "
            f"{frames_actual_next_in_top3}/{total_frames_checked}"
        )
        print(f"Unique same-frame matching top-3 states: {len(top3_match_indices)}")
        print(f"Unique nonmatching top-3 states: {len(top3_nonmatch_indices)}")

    # -------------------------
    # Split plotted groups
    # -------------------------
    # Red = actual route states, except states that are orange same-frame matches
    route_only_indices = route_index_set - top3_match_indices

    # Green = top-3 predictions that did not match the actual next state
    # If a state is ever a same-frame match, show it as orange instead of green.
    top3_only_nonmatch_indices = top3_nonmatch_indices - top3_match_indices

    # Orange = predictions that matched the actual next state for that frame
    orange_match_indices = top3_match_indices

    print(f"Route-only states plotted red: {len(route_only_indices)}")
    print(f"Top-3 nonmatching states plotted green: {len(top3_only_nonmatch_indices)}")
    print(f"Same-frame matching states plotted orange: {len(orange_match_indices)}")

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # All manifold states
    if show_all_points:
        ax.scatter(
            manifold[:, 0],
            manifold[:, 1],
            manifold[:, 2],
            s=12,
            color="gray",
            alpha=0.4,
            label=f"All {state_type}",
            zorder=1
        )

    # Actual route line, using ordered route coordinates
    if show_route_line and len(route_coords) > 1:
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            route_coords[:, 2],
            color="black",
            linewidth=1.8,
            alpha=0.55,
            label="Actual route path",
            zorder=6
        )

    # Red route-only states
    if len(route_only_indices) > 0:
        route_only_coords = manifold[list(route_only_indices)]

        ax.scatter(
            route_only_coords[:, 0],
            route_only_coords[:, 1],
            route_only_coords[:, 2],
            s=40,
            color="crimson",
            marker="o",
            alpha=1.0,
            label="Actual route states only",
            zorder=9
        )

    # Green top-3 predictions that did NOT match actual next state for that frame
    if show_top3 and len(top3_only_nonmatch_indices) > 0:
        green_coords = manifold[list(top3_only_nonmatch_indices)]

        ax.scatter(
            green_coords[:, 0],
            green_coords[:, 1],
            green_coords[:, 2],
            s=85,
            color="deepskyblue",
            marker="x",
            alpha=1.0,
            label="Top-3 predictions not actual next state",
            zorder=12
        )

    # Orange top-3 predictions that DID match actual next state for that frame
    if show_top3 and len(orange_match_indices) > 0:
        orange_coords = manifold[list(orange_match_indices)]

        ax.scatter(
            orange_coords[:, 0],
            orange_coords[:, 1],
            orange_coords[:, 2],
            s=55,
            color="gold",
            marker="o",
            alpha=1.0,
            label="Top-3 prediction = actual next state",
            zorder=13
        )

    ax.set_title(
        f"3D {manifold_type} Projection of {state_type}\n"
        f"Actual Route vs Top-3 Dictionary Predictions"
    )
    ax.set_xlabel(f"{manifold_type} 1")
    ax.set_ylabel(f"{manifold_type} 2")
    ax.set_zlabel(f"{manifold_type} 3")

    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def filter_long_triangles(triangulation, points_2d, max_edge_length=None, scale=3.0):
    """
    Masks triangles with overly long edges.

    This does not move points or smooth data.
    It only removes triangle connections that span large gaps.
    """
    triangles = triangulation.triangles
    keep = []

    if max_edge_length is None:
        tree = cKDTree(points_2d)
        dists, _ = tree.query(points_2d, k=2)
        nearest_neighbor_dists = dists[:, 1]

        max_edge_length = np.percentile(nearest_neighbor_dists, 90) * scale

    for tri in triangles:
        p0, p1, p2 = points_2d[tri]

        e01 = np.linalg.norm(p0 - p1)
        e12 = np.linalg.norm(p1 - p2)
        e20 = np.linalg.norm(p2 - p0)

        if max(e01, e12, e20) <= max_edge_length:
            keep.append(True)
        else:
            keep.append(False)

    triangulation.set_mask(~np.array(keep))
    return triangulation

def plot_manifold_trisurf_raw(
    manifold,
    manifold_type,
    state_keys,
    state_type,
    route_sequence,
    top3_route_history=None,
    save_path=None,
    show_points=True,
    show_route_line=True,
    show_top3=True,
    all_points_color="blue",
    route_color="red",
    top3_color="green",
    surface_color="lightgray",
    point_size=10,
    route_point_size=10,
    top3_point_size=16,
    all_points_alpha=0.12,
    route_points_alpha=0.95,
    top3_points_alpha=0.80,
    top3_line_alpha=0.55,
    surface_alpha=0.28,
    surface_linewidth=0.15,
    route_linewidth=1.2,
    top3_linewidth=0.9,
    max_edge_length=None
):
    """
    Raw triangulated 3D manifold plot with filtered triangles.

    Blue = all states
    Red = actual animation route
    Green = exact top-3 transitions saved from the animation
    """

    state_path = extract_state_path_from_route(route_sequence, state_type)

    hashable_state_keys = [
        make_hashable_state(key)
        for key in state_keys
    ]

    key_to_idx = {
        key: i
        for i, key in enumerate(hashable_state_keys)
    }

    # -------------------------
    # Actual route lookup
    # -------------------------
    route_indices = []
    missing_route = 0

    for state in state_path:
        h_state = make_hashable_state(state)

        if h_state in key_to_idx:
            route_indices.append(key_to_idx[h_state])
        else:
            missing_route += 1

    if len(route_indices) == 0:
        raise ValueError(f"No route {state_type} matched manifold keys.")

    route_coords = manifold[route_indices]

    print(f"Exact route {state_type} found: {len(route_indices)}/{len(state_path)}")
    print(f"Missing route {state_type}: {missing_route}")
    print(
        f"Unique route {state_type}:",
        len(set(make_hashable_state(s) for s in state_path))
    )

    # -------------------------
    # Top-3 animation transitions
    # -------------------------
    top3_edges = []
    top3_target_indices = set()
    missing_top3_current = 0
    missing_top3_targets = 0

    if show_top3 and top3_route_history is not None:
        for frame_info in top3_route_history:
            current_state, target_states = get_top3_current_and_targets(
                frame_info,
                state_type
            )

            h_current = make_hashable_state(current_state)

            if h_current not in key_to_idx:
                missing_top3_current += 1
                continue

            current_idx = key_to_idx[h_current]
            current_coord = manifold[current_idx]

            for target_state in target_states:
                h_target = make_hashable_state(target_state)

                if h_target in key_to_idx:
                    target_idx = key_to_idx[h_target]
                    target_coord = manifold[target_idx]

                    top3_edges.append((current_coord, target_coord))
                    top3_target_indices.add(target_idx)
                else:
                    missing_top3_targets += 1

        print(f"Top-3 animation transition edges found: {len(top3_edges)}")
        print(f"Missing top-3 current states: {missing_top3_current}")
        print(f"Missing top-3 target states: {missing_top3_targets}")

    # -------------------------
    # Triangulation
    # -------------------------
    x = manifold[:, 0]
    y = manifold[:, 1]
    z = manifold[:, 2]

    points_2d = manifold[:, :2]

    triangulation = mtri.Triangulation(x, y)
    triangulation = filter_long_triangles(
        triangulation,
        points_2d,
        max_edge_length=max_edge_length
    )

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Raw triangulated manifold surface
    ax.plot_trisurf(
        triangulation,
        z,
        color=surface_color,
        alpha=surface_alpha,
        linewidth=surface_linewidth,
        edgecolor="gray"
    )

    # Optional raw manifold points
    if show_points:
        ax.scatter(
            x,
            y,
            z,
            s=point_size,
            color=all_points_color,
            alpha=all_points_alpha,
            label=f"All {state_type}"
        )

    # Green top-3 transitions first
    if show_top3 and top3_route_history is not None:
        for start_coord, end_coord in top3_edges:
            ax.plot(
                [start_coord[0], end_coord[0]],
                [start_coord[1], end_coord[1]],
                [start_coord[2], end_coord[2]],
                color=top3_color,
                linewidth=top3_linewidth,
                alpha=top3_line_alpha,
                zorder=7
            )

        if len(top3_target_indices) > 0:
            top3_coords = manifold[list(top3_target_indices)]

            ax.scatter(
                top3_coords[:, 0],
                top3_coords[:, 1],
                top3_coords[:, 2],
                s=top3_point_size,
                color=top3_color,
                alpha=top3_points_alpha,
                label="Top 3 animation transition targets",
                zorder=8
            )

    # Exact route points
    ax.scatter(
        route_coords[:, 0],
        route_coords[:, 1],
        route_coords[:, 2],
        s=route_point_size,
        color=route_color,
        alpha=route_points_alpha,
        label=f"Animation-route {state_type}",
        zorder=10
    )

    # Exact route line
    if show_route_line:
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            route_coords[:, 2],
            color=route_color,
            linewidth=route_linewidth,
            alpha=0.85,
            zorder=9
        )

    ax.set_title(
        f"Raw 3D {manifold_type} {state_type} surface\n"
        f"Actual Route vs Top 3 Animation Transitions"
    )
    ax.set_xlabel(f"{manifold_type} 1")
    ax.set_ylabel(f"{manifold_type} 2")
    ax.set_zlabel(f"{manifold_type} 3")

    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == '__main__':
    neural_state_dict, behavioral_state_dict, pair_transition_dict = load_cache(min_visits=100, max_visits=151, sd=42, bin_size=0.5)

    route = np.load(os.path.join(SCRIPT_DIR, "route_sequence_min100.npy"), allow_pickle=True)

    top3_route_history = np.load(os.path.join(SCRIPT_DIR, "top3_route_transitions_min100.npy"), allow_pickle=True).tolist()

    '''paired_matrix, paired_keys = convert_dict_to_matrix(pair_transition_dict, dict_type='paired_transition')
    print("Paired transition matrix shape:", paired_matrix.shape)

    paired_tsne_manifold = perform_manifold_learning(paired_matrix, method='tsne', n_components=3, random_state=42)
    print("Paired transition t-SNE manifold shape:", paired_tsne_manifold.shape)

    paired_umap_manifold = perform_manifold_learning(paired_matrix, method='umap', n_components=3, random_state=42)
    print("Paired transition UMAP manifold shape:", paired_umap_manifold.shape)

    paired_pca_manifold = perform_manifold_learning(paired_matrix, method='pca', n_components=3, random_state=42)
    print("Paired transition PCA manifold shape:", paired_pca_manifold.shape)

    plot_manifold_raw(
        paired_tsne_manifold,
        manifold_type="t-SNE",
        state_keys=paired_keys,
        state_type="joint states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        paired_umap_manifold,
        manifold_type="UMAP",
        state_keys=paired_keys,
        state_type="joint states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        paired_pca_manifold,
        manifold_type="PCA",
        state_keys=paired_keys,
        state_type="joint states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )

    b_state_matrix, behavioral_keys = convert_dict_to_matrix(behavioral_state_dict, dict_type='behavioral_state')
    print("Behavioral state matrix shape:", b_state_matrix.shape)

    b_tsne_manifold = perform_manifold_learning(b_state_matrix, method='tsne', n_components=3, random_state=42)
    print("Behavioral t-SNE manifold shape:", b_tsne_manifold.shape)

    b_umap_manifold = perform_manifold_learning(b_state_matrix, method='umap', n_components=3, random_state=42)
    print("Behavioral UMAP manifold shape:", b_umap_manifold.shape)

    b_pca_manifold = perform_manifold_learning(b_state_matrix, method='pca', n_components=3, random_state=42)
    print("Behavioral PCA manifold shape:", b_pca_manifold.shape)

    plot_manifold_raw(
        b_tsne_manifold,
        manifold_type="t-SNE",
        state_keys=behavioral_keys,
        state_type="behavioral states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        b_umap_manifold,
        manifold_type="UMAP",
        state_keys=behavioral_keys,
        state_type="behavioral states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        b_pca_manifold,
        manifold_type="PCA",
        state_keys=behavioral_keys,
        state_type="behavioral states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=True,
        show_all_points=True,
        show_top3=True
    )'''
 
    n_state_matrix, neural_keys = convert_dict_to_matrix(neural_state_dict, dict_type='neural_state')
    print("Neural state matrix shape:", n_state_matrix.shape)


    n_tsne_manifold = perform_manifold_learning(n_state_matrix, method='tsne', n_components=3, random_state=42)
    print("Neural t-SNE manifold shape:", n_tsne_manifold.shape)

    n_umap_manifold = perform_manifold_learning(n_state_matrix, method='umap', n_components=3, random_state=42)
    print("Neural UMAP manifold shape:", n_umap_manifold.shape)

    n_pca_manifold = perform_manifold_learning(n_state_matrix, method='pca', n_components=3, random_state=42)
    print("Neural PCA manifold shape:", n_pca_manifold.shape)
    
    plot_manifold_raw(
        n_tsne_manifold,
        manifold_type="t-SNE",
        state_keys=neural_keys,
        state_type="neural states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=False,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        n_umap_manifold,
        manifold_type="UMAP",
        state_keys=neural_keys,
        state_type="neural states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=False,
        show_all_points=True,
        show_top3=True
    )

    plot_manifold_raw(
        n_pca_manifold,
        manifold_type="PCA",
        state_keys=neural_keys,
        state_type="neural states",
        route_sequence=route,
        top3_route_history=top3_route_history,
        save_path=None,
        show_route_line=False,
        show_all_points=True,
        show_top3=True
    )
    
    '''
    plot_manifold_trisurf_raw(
        n_tsne_manifold,
        neural_keys,
        route_sequence=route,
        save_path=None,
        show_points=True,
        show_route_line=True,
        all_points_color="blue",
        route_color="red",
        surface_color="blue",
        point_size=10,
        route_point_size=30,
        all_points_alpha=0.65,
        route_points_alpha=0.95,
        surface_alpha=0.65,
        surface_linewidth=0.3,
        route_linewidth=1.0,
        max_edge_length=1
    )'''
