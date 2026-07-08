from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import os 
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        for key1 in cached_dict:
            for key2 in cached_dict[key1]:
                cached_dict[key1][key2] = np.array(cached_dict[key1][key2])
        return np.array([item for sublist in cached_dict.values() for item in sublist])
    elif dict_type == 'paired_transition':
        for key1 in cached_dict:
            for key2 in cached_dict[key1]:
                cached_dict[key1][key2] = np.array(cached_dict[key1][key2])
        return np.array([item for sublist in cached_dict.values() for item in sublist]) # check if right
    else:
        return ValueError("Unsupported dict_type. Please use 'neural_state', 'behavioral_state', or 'paired_transition'.")
    

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

def plot_manifold(manifold_data, neural_keys, route_sequence=None, save_path=None):


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all neural states
    ax.scatter(
        manifold_data[:, 0],
        manifold_data[:, 1],
        manifold_data[:, 2],
        s=8,
        alpha=0.25,
        label="All neural states"
    )

    # Convex hull surface
    hull = ConvexHull(manifold_data)
    faces = [manifold_data[simplex] for simplex in hull.simplices]

    surface = Poly3DCollection(
        faces,
        alpha=0.12,
        linewidths=0.3
    )
    surface.set_edgecolor("gray")
    ax.add_collection3d(surface)

    # Highlight predetermined path
    if route_sequence is not None:
        neural_path = extract_neural_path_from_route(route_sequence)

        key_to_idx = {key: idx for idx, key in enumerate(neural_keys)}

        path_indices = []
        missing = 0

        for n_state in neural_path:
            if n_state in key_to_idx:
                path_indices.append(key_to_idx[n_state])
            else:
                missing += 1

        if len(path_indices) > 0:
            path_coords = manifold_data[path_indices]

            ax.plot(
                path_coords[:, 0],
                path_coords[:, 1],
                path_coords[:, 2],
                linewidth=3.0,
                color="red",
                label="Predetermined neural path"
            )

            ax.scatter(
                path_coords[:, 0],
                path_coords[:, 1],
                path_coords[:, 2],
                s=55,
                color="red",
                alpha=1.0,
                zorder=10
            )

            # Mark start and end
            ax.scatter(
                path_coords[0, 0],
                path_coords[0, 1],
                path_coords[0, 2],
                s=95,
                color="green",
                alpha=1.0,
                label="Path start",
                zorder=11
            )

            ax.scatter(
                path_coords[-1, 0],
                path_coords[-1, 1],
                path_coords[-1, 2],
                s=95,
                color="black",
                alpha=1.0,
                label="Path end",
                zorder=11
            )

        print(f"Path states found in manifold: {len(path_indices)}/{len(neural_path)}")
        print(f"Missing path states: {missing}")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.legend()

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
    
    plot_manifold(n_tsne_manifold, neural_keys, route_sequence=neural_route, save_path=None)
