from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import os 
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.tri as mtri
from collections import Counter


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

    # --------------------------------------------------
    # 1. Actual ordered route path
    # --------------------------------------------------
    actual_state_path = extract_state_path_from_route(route_sequence, state_type)

    if len(actual_state_path) == 0:
        raise ValueError("Actual route path is empty.")

    # --------------------------------------------------
    # 2. Build top-1 predicted route path
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 3. Map manifold keys -> indices
    # --------------------------------------------------
    hashable_state_keys = [
        make_hashable_state(key)
        for key in state_keys
    ]

    key_to_idx = {
        key: i
        for i, key in enumerate(hashable_state_keys)
    }

    # --------------------------------------------------
    # 4. Actual route indices
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 5. Predicted route indices
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 6. Trim to common frame length
    # --------------------------------------------------
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

    # Start at 1 because frame 0 is the shared initial state.
    # The real transition comparison is actual next state vs predicted next state.
    for frame_idx in range(1, common_len):
        actual_state = make_hashable_state(actual_state_path[frame_idx])
        predicted_state = make_hashable_state(predicted_state_path[frame_idx])

        if actual_state == predicted_state:
            matching_frames.append(frame_idx)
        else:
            actual_only_frames.append(frame_idx)
            predicted_only_frames.append(frame_idx)

    transition_count = common_len - 1

    # --------------------------------------------------
    # 8. Debug counts
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 9. Visual coordinate offsets
    # --------------------------------------------------
    data_span = manifold.max(axis=0) - manifold.min(axis=0)
    offset_vector = np.array([
        visual_offset_fraction * data_span[0],
        visual_offset_fraction * data_span[1],
        0.0
    ])

    # These are display-only coordinates.
    # The underlying manifold coordinates and matching logic are unchanged.
    actual_plot_coords = actual_coords - offset_vector
    predicted_plot_coords = predicted_coords + offset_vector

    # Gold points stay at true coordinates
    gold_coords = actual_coords[matching_frames] if len(matching_frames) > 0 else None

    # --------------------------------------------------
    # 10. Helper for drawing a plot
    # --------------------------------------------------
    def draw_route_plot(
        ax,
        zoom_to_cluster=False,
        background_alpha=0.20,
        background_point_size=9,
        title_suffix=""
    ):
        ax.set_proj_type("ortho")

        # Background manifold
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

        # Actual route line, visually offset
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

        # Predicted route line, visually offset
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

        # Actual-only points
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

        # Predicted-only points
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

        # Same-frame matches
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

        # Cluster zoom based on actual + predicted route coordinates only.
        if zoom_to_cluster:
            # Use the plotted/offset route coordinates so the zoom contains
            # exactly what is visually drawn.
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

            # Add padding separately for each axis
            x_pad = zoom_padding_fraction * max(x_range, 1e-8)
            y_pad = zoom_padding_fraction * max(y_range, 1e-8)
            z_pad = zoom_padding_fraction * max(z_range, 1e-8)

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_zlim(z_min - z_pad, z_max + z_pad)

        # Do not force equal box aspect for PCA because PCA has much smaller z-range.
        # This prevents the PCA view from looking overly squashed.
        if manifold_type == "PCA":
            ax.set_box_aspect((1.35, 1.15, 0.85))
        else:
            ax.set_box_aspect((1.25, 1.15, 0.95))

        ax.legend(loc="upper right")

    # --------------------------------------------------
    # 11. Full view
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 12. Cluster zoom view
    # --------------------------------------------------
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

    test_dictionary_manifold = "neural"

    if test_dictionary_manifold == "joint":
        paired_matrix, paired_keys = convert_dict_to_matrix(pair_transition_dict, dict_type='paired_transition')
        print("Paired transition matrix shape:", paired_matrix.shape)

        paired_tsne_manifold = perform_manifold_learning(paired_matrix, method='tsne', n_components=3, random_state=42)
        print("Paired transition t-SNE manifold shape:", paired_tsne_manifold.shape)

        paired_umap_manifold = perform_manifold_learning(paired_matrix, method='umap', n_components=3, random_state=42)
        print("Paired transition UMAP manifold shape:", paired_umap_manifold.shape)

        paired_pca_manifold = perform_manifold_learning(paired_matrix, method='pca', n_components=3, random_state=42)
        print("Paired transition PCA manifold shape:", paired_pca_manifold.shape)

        plot_manifold_raw_two_route(
            paired_tsne_manifold,
            manifold_type="t-SNE",
            state_keys=paired_keys,
            state_type="joint states",
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

        plot_manifold_raw_two_route(
            paired_umap_manifold,
            manifold_type="UMAP",
            state_keys=paired_keys,
            state_type="joint states",
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

        plot_manifold_raw_two_route(
            paired_pca_manifold,
            manifold_type="PCA",
            state_keys=paired_keys,
            state_type="joint states",
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
    elif test_dictionary_manifold == 'behavioral':
        b_state_matrix, behavioral_keys = convert_dict_to_matrix(behavioral_state_dict, dict_type='behavioral_state')
        print("Behavioral state matrix shape:", b_state_matrix.shape)

        b_tsne_manifold = perform_manifold_learning(b_state_matrix, method='tsne', n_components=3, random_state=42)
        print("Behavioral t-SNE manifold shape:", b_tsne_manifold.shape)

        b_umap_manifold = perform_manifold_learning(b_state_matrix, method='umap', n_components=3, random_state=42)
        print("Behavioral UMAP manifold shape:", b_umap_manifold.shape)

        b_pca_manifold = perform_manifold_learning(b_state_matrix, method='pca', n_components=3, random_state=42)
        print("Behavioral PCA manifold shape:", b_pca_manifold.shape)

        plot_manifold_raw_two_route(
            b_tsne_manifold,
            manifold_type="t-SNE",
            state_keys=behavioral_keys,
            state_type="behavioral states",
            route_sequence=route,
            top3_route_history=top3_route_history,
            save_prefix=None,
            show_all_points=True,
            make_full_plot=True,
            make_zoomed_plot=False,
            zoom_padding_fraction=0.18,
            zoom_percentile=5,
            visual_offset_fraction=0.018,
            full_background_alpha=0.20,
            zoom_background_alpha=0.08,
            full_background_point_size=9,
            zoom_background_point_size=6
        )

        plot_manifold_raw_two_route(
            b_umap_manifold,
            manifold_type="UMAP",
            state_keys=behavioral_keys,
            state_type="behavioral states",
            route_sequence=route,
            top3_route_history=top3_route_history,
            save_prefix=None,
            show_all_points=True,
            make_full_plot=True,
            make_zoomed_plot=False,
            zoom_padding_fraction=0.18,
            zoom_percentile=5,
            visual_offset_fraction=0.018,
            full_background_alpha=0.20,
            zoom_background_alpha=0.08,
            full_background_point_size=9,
            zoom_background_point_size=6
        )

        plot_manifold_raw_two_route(
            b_pca_manifold,
            manifold_type="PCA",
            state_keys=behavioral_keys,
            state_type="behavioral states",
            route_sequence=route,
            top3_route_history=top3_route_history,
            save_prefix=None,
            show_all_points=True,
            make_full_plot=True,
            make_zoomed_plot=False,
            zoom_padding_fraction=0.18,
            zoom_percentile=5,
            visual_offset_fraction=0.018,
            full_background_alpha=0.20,
            zoom_background_alpha=0.08,
            full_background_point_size=9,
            zoom_background_point_size=6
        )
    elif test_dictionary_manifold == "neural":
        n_state_matrix, neural_keys = convert_dict_to_matrix(neural_state_dict, dict_type='neural_state')
        print("Neural state matrix shape:", n_state_matrix.shape)


        n_tsne_manifold = perform_manifold_learning(n_state_matrix, method='tsne', n_components=3, random_state=42)
        print("Neural t-SNE manifold shape:", n_tsne_manifold.shape)

        n_umap_manifold = perform_manifold_learning(n_state_matrix, method='umap', n_components=3, random_state=42)
        print("Neural UMAP manifold shape:", n_umap_manifold.shape)

        n_pca_manifold = perform_manifold_learning(n_state_matrix, method='pca', n_components=3, random_state=42)
        print("Neural PCA manifold shape:", n_pca_manifold.shape)
        
        plot_manifold_raw_two_route(
            n_tsne_manifold,
            manifold_type="t-SNE",
            state_keys=neural_keys,
            state_type="neural states",
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

        plot_manifold_raw_two_route(
            n_umap_manifold,
            manifold_type="UMAP",
            state_keys=neural_keys,
            state_type="neural states",
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

        plot_manifold_raw_two_route(
            n_pca_manifold,
            manifold_type="PCA",
            state_keys=neural_keys,
            state_type="neural states",
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
    else: raise ValueError("test value not an accepted dictionary")


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
