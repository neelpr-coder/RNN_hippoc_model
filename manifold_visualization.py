from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import os 
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, label
from skimage.measure import marching_cubes
from matplotlib.lines import Line2D
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


def plot_manifold_raw(
    manifold,
    neural_keys,
    route_sequence,
    save_path=None,
    show_route_line=True,
    show_all_points=True
):
    """
    Raw 3D manifold plot.
    """

    neural_path = extract_neural_path_from_route(route_sequence)
    key_to_idx = {key: i for i, key in enumerate(neural_keys)}

    route_indices = []

    for n_state in neural_path:
        if n_state in key_to_idx:
            route_indices.append(key_to_idx[n_state])

    if len(route_indices) == 0:
        raise ValueError("No route neural states matched manifold neural keys.")

    route_coords = manifold[route_indices]

    print(f"Exact route neural states found: {len(route_indices)}/{len(neural_path)}")
    print(f"Unique route neural states: {len(set(neural_path))}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    if show_all_points:
        ax.scatter(
            manifold[:, 0],
            manifold[:, 1],
            manifold[:, 2],
            s=12,
            color="blue",
            alpha=0.75,
            label="All neural states"
        )

    ax.scatter(
        route_coords[:, 0],
        route_coords[:, 1],
        route_coords[:, 2],
        s=12,
        color="red",
        alpha=0.95,
        label="Animation-route neural states",
        zorder=10
    )

    if show_route_line:
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            route_coords[:, 2],
            color="red",
            linewidth=1.8,
            alpha=0.85,
            zorder=9
        )

    ax.set_title("3D t-SNE Projection of Neural States with Animation Route Highlighted")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

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
    neural_keys,
    route_sequence,
    save_path=None,
    show_points=True,
    show_route_line=True,
    all_points_color="blue",
    route_color="red",
    surface_color="lightgray",
    point_size=10,
    route_point_size=10,
    all_points_alpha=0.12,
    route_points_alpha=0.95,
    surface_alpha=0.28,
    surface_linewidth=0.15,
    route_linewidth=1.2,
    max_edge_length=None
):
    """
    Raw triangulated 3D manifold plot with filtered triangles.
    """

    neural_path = extract_neural_path_from_route(route_sequence)
    key_to_idx = {key: i for i, key in enumerate(neural_keys)}

    route_indices = []
    for n_state in neural_path:
        if n_state in key_to_idx:
            route_indices.append(key_to_idx[n_state])

    if len(route_indices) == 0:
        raise ValueError("No route neural states matched manifold neural keys.")

    route_coords = manifold[route_indices]

    print(f"Exact route neural states found: {len(route_indices)}/{len(neural_path)}")
    print(f"Unique route neural states: {len(set(neural_path))}")

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
            label="All neural states"
        )

    # Exact route points
    ax.scatter(
        route_coords[:, 0],
        route_coords[:, 1],
        route_coords[:, 2],
        s=route_point_size,
        color=route_color,
        alpha=route_points_alpha,
        label="Animation-route neural states",
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

    ax.set_title("Raw 3D t-SNE Neural-State Surface with Animation Route")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == '__main__':
    neural_state_dict, behavioral_state_dict, pair_transition_dict = load_cache(min_visits=100, max_visits=151, sd=42, bin_size=0.5)
    
    n_state_matrix, neural_keys = convert_dict_to_matrix(neural_state_dict, dict_type='neural_state')
    print("Neural state matrix shape:", n_state_matrix.shape)

    n_tsne_manifold = perform_manifold_learning(n_state_matrix, method='tsne', n_components=3, random_state=42)
    print("t-SNE manifold shape:", n_tsne_manifold.shape)

    neural_route = np.load(os.path.join(SCRIPT_DIR, "route_sequence_min100.npy"), allow_pickle=True)

    np.save("n_tsne_manifold_min100_sd42.npy", n_tsne_manifold)
    np.save("neural_keys_min100_sd42.npy", np.array(neural_keys, dtype=object))

    '''plot_manifold_trisurf_raw(
        n_tsne_manifold,
        neural_keys,
        route_sequence=neural_route,
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

    '''plot_manifold_raw(
        n_tsne_manifold,
        neural_keys,
        route_sequence=neural_route,
        save_path=None,
        show_route_line=True,
        show_all_points=True
    )'''
