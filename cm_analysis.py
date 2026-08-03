import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import perturbation_testing as pt
import cm_experiments as cme
import connected_models as cm

import figure2_generation as f2g
import manifold_visualization as mv

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def transition_prob_from_counts(next_counts, next_key):
    total = sum(next_counts.values())
    if total == 0:
        return 0.0, 0
    return next_counts.get(next_key, 0) / total, total

def neural_state_to_vector(state):
    return np.asarray(state, dtype=float)

def joint_Na_Nb_state_to_vector(state):
    Na, Nb = state
    return np.concatenate([np.asarray(Na, dtype=float), np.asarray(Nb, dtype=float)])

def joint_Na_Nb_b_state_to_vector(state):
    Na, Nb, B = state
    b_vec = mv.behavioral_state_to_vector(B)
    return np.concatenate([b_vec, np.asarray(Na, dtype=float), np.asarray(Nb, dtype=float)])

def generate_connected_route(Na_Nb_b_transition_dict, route_length=50, seed=42):
    rng = random.Random(seed)

    available_current_states = [
        state for state, next_states in Na_Nb_b_transition_dict.items()
        if len(next_states) > 0
    ]

    if not available_current_states:
        raise ValueError("Na_Nb_b_transition_dict has no transitions.")

    cur_state = rng.choice(available_current_states)

    route = []

    for _ in range(route_length):
        next_counts = Na_Nb_b_transition_dict.get(cur_state, {})
        if len(next_counts) == 0:
            break

        next_states = list(next_counts.keys())
        weights = list(next_counts.values())

        next_state = rng.choices(next_states, weights=weights, k=1)[0]
        observed_count = next_counts[next_state]

        cur_Na, cur_Nb, cur_B = cur_state
        next_Na, next_Nb, next_B = next_state

        route.append((cur_B, cur_Na, cur_Nb, next_B, next_Na, next_Nb, observed_count))

        cur_state = next_state

    return route

def calc_connected_stage3_error_history(
    route,
    og_Na_dict,
    og_Nb_dict,
    og_Na_b_dict,
    og_Nb_b_dict,
    og_Na_Nb_dict,
    og_Na_Nb_b_dict,
    new_Na_dict,
    new_Nb_dict,
    new_Na_b_dict,
    new_Nb_b_dict,
    new_Na_Nb_dict,
    new_Na_Nb_b_dict
):
    error_history = []

    for frame, step in enumerate(route, start=1):
        B0, Na0, Nb0, B1, Na1, Nb1, observed_count = step

        cur_Na = Na0
        next_Na = Na1

        cur_Nb = Nb0
        next_Nb = Nb1

        cur_Na_b = (Na0, B0)
        next_Na_b = (Na1, B1)

        cur_Nb_b = (Nb0, B0)
        next_Nb_b = (Nb1, B1)

        cur_Na_Nb = (Na0, Nb0)
        next_Na_Nb = (Na1, Nb1)

        cur_Na_Nb_b = (Na0, Nb0, B0)
        next_Na_Nb_b = (Na1, Nb1, B1)

        specs = [
            ("Na", og_Na_dict, new_Na_dict, cur_Na, next_Na),
            ("Nb", og_Nb_dict, new_Nb_dict, cur_Nb, next_Nb),
            ("Na_b", og_Na_b_dict, new_Na_b_dict, cur_Na_b, next_Na_b),
            ("Nb_b", og_Nb_b_dict, new_Nb_b_dict, cur_Nb_b, next_Nb_b),
            ("Na_Nb", og_Na_Nb_dict, new_Na_Nb_dict, cur_Na_Nb, next_Na_Nb),
            ("Na_Nb_b", og_Na_Nb_b_dict, new_Na_Nb_b_dict, cur_Na_Nb_b, next_Na_Nb_b)
            ]

        row = {"frame": frame}

        for name, orig_dict, upd_dict, cur_key, next_key in specs:
            orig_next_counts = orig_dict.get(cur_key, {})
            upd_next_counts = upd_dict.get(cur_key, {})

            orig_prob, _ = transition_prob_from_counts(orig_next_counts, next_key)
            upd_prob, _ = transition_prob_from_counts(upd_next_counts, next_key)

            C = len(orig_next_counts)
            normalized_error = (orig_prob - upd_prob) / C if C > 0 else np.nan

            row[name] = normalized_error

        error_history.append(row)

    return error_history

def plot_connected_error_overlay(error_history, title="Connected RNN Error Over Time", save_path=None):
    frames = [row["frame"] for row in error_history]

    labels = ["Na", "Nb", "Na_b", "Nb_b", "Na_Nb", "Na_Nb_b"]

    plt.figure(figsize=(12, 7))

    for label in labels:
        values = [row[label] for row in error_history]
        plt.plot(frames, values, marker="o", linewidth=1.5, label=label)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time step")
    plt.ylabel("Normalized error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def pearson_corr_between_vectors(vec_a, vec_b):
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")

    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return np.nan

    return np.corrcoef(vec_a, vec_b)[0, 1]

def euclidean_distance_between_vectors(vec_a, vec_b):
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")

    return np.linalg.norm(vec_a - vec_b)

def get_top_states_from_transition_dict(transition_dict, top_k=100):
    state_counts = {state: sum(next_counts.values()) for state, next_counts in transition_dict.items()}
    sorted_items = sorted(state_counts.items(), key=lambda item: item[1], reverse=True)[:top_k]

    states = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    return states, counts

def build_similarity_matrix(transition_dict_a, transition_dict_b, top_k=100, metric="pearson"):
    states_a, counts_a = get_top_states_from_transition_dict(transition_dict_a,top_k=top_k)
    states_b, counts_b = get_top_states_from_transition_dict(transition_dict_b,top_k=top_k)

    if len(states_a) == 0 or len(states_b) == 0:
        raise ValueError("One or both transition dictionaries are empty.")

    matrix = np.full((len(states_b), len(states_a)), np.nan)

    for i, state_b in enumerate(states_b):
        for j, state_a in enumerate(states_a):
            if metric == "pearson":
                matrix[i, j] = pearson_corr_between_vectors(state_a, state_b)
            elif metric == "euclidean":
                matrix[i, j] = euclidean_distance_between_vectors(state_a, state_b)
            else:
                raise ValueError("metric must be 'pearson' or 'euclidean'")

    x_labels = [f"A:N{j + 1}" for j in range(len(states_a))]
    y_labels = [f"B:N{i + 1}" for i in range(len(states_b))]

    return matrix, x_labels, y_labels, counts_a, counts_b

def plot_pearson_matrix(matrix, x_labels, y_labels, title, tick_step=10):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, origin="lower", aspect="auto", vmin=-1, vmax=1, cmap="viridis")

    plt.title(title)
    plt.xlabel("States from first dictionary")
    plt.ylabel("States from second dictionary")

    x_ticks = np.arange(0, len(x_labels), tick_step)
    y_ticks = np.arange(0, len(y_labels), tick_step)

    plt.xticks(x_ticks, [x_labels[i] for i in x_ticks], rotation=45, ha="right", fontsize=8)
    plt.yticks(y_ticks, [y_labels[i] for i in y_ticks], fontsize=8)
    plt.colorbar(im, label="Pearson correlation")
    plt.tight_layout()
    plt.show()

def plot_euclidean_matrix(matrix, x_labels, y_labels, title, tick_step=10, cmap="viridis_r"):
    plt.figure(figsize=(10, 8))

    im = plt.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)

    plt.title(title)
    plt.xlabel("States from first dictionary")
    plt.ylabel("States from second dictionary")

    x_ticks = np.arange(0, len(x_labels), tick_step)
    y_ticks = np.arange(0, len(y_labels), tick_step)

    plt.xticks(x_ticks, [x_labels[i] for i in x_ticks], rotation=45, ha="right", fontsize=8)
    plt.yticks(y_ticks, [y_labels[i] for i in y_ticks], fontsize=8)
    plt.colorbar(im, label="Euclidean distance")
    plt.tight_layout()
    plt.show()

def single_neural_dict_to_matrix(transition_dict):
    state_set = set()

    for cur_state, next_states in transition_dict.items():
        state_set.add(cur_state)
        state_set.update(next_states.keys())

    state_keys = list(state_set)
    matrix = np.array([np.asarray(state, dtype=float) for state in state_keys])

    return matrix, state_keys

def joint_Na_Nb_dict_to_matrix(transition_dict):
    state_set = set()

    for cur_state, next_states in transition_dict.items():
        state_set.add(cur_state)
        state_set.update(next_states.keys())

    state_keys = list(state_set)
    matrix = np.array([joint_Na_Nb_state_to_vector(state) for state in state_keys])

    return matrix, state_keys

def joint_Na_Nb_b_dict_to_matrix(transition_dict):
    state_set = set()

    for cur_state, next_states in transition_dict.items():
        state_set.add(cur_state)
        state_set.update(next_states.keys())

    state_keys = list(state_set)
    matrix = np.array([joint_Na_Nb_b_state_to_vector(state) for state in state_keys])

    return matrix, state_keys

def plot_3d_embedding(embedding, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=10, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    plt.tight_layout()
    plt.show()

def generate_manifold_from_dict(transition_dict, dict_kind, method="tsne", title=None):
    if dict_kind == "Na":
        matrix, state_keys = single_neural_dict_to_matrix(transition_dict)
    elif dict_kind == "Nb":
        matrix, state_keys = single_neural_dict_to_matrix(transition_dict)
    elif dict_kind == "Na_Nb":
        matrix, state_keys = joint_Na_Nb_dict_to_matrix(transition_dict)
    elif dict_kind == "Na_Nb_b":
        matrix, state_keys = joint_Na_Nb_b_dict_to_matrix(transition_dict)
    else:
        raise ValueError("Unknown dict_kind")

    embedding = mv.perform_manifold_learning(matrix, method=method, n_components=3)

    if title is None:
        title = f"{dict_kind} manifold ({method})"

    plot_3d_embedding(embedding, title)
    return embedding, state_keys

def connect_b_to_n(n_b_transition_dict):
    b_to_n_connected = defaultdict(set)
    for n_state, b_state in n_b_transition_dict.keys():
        b_to_n_connected[b_state].add(n_state)
    return b_to_n_connected

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = cm.connected_models().to(device)
    state_dict = torch.load("post_stage1_connected_model_sd42.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    b_transition_dict, all_visit_b_count_dict, Na_transition_dict, Nb_transition_dict, Na_Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_b_transition_dict = cme.generate_dicts(model)
    Na_matrix, Na_keys = single_neural_dict_to_matrix(Na_transition_dict)
    Nb_matrix, Nb_keys = single_neural_dict_to_matrix(Nb_transition_dict)
    print("Unique Na states:", len(Na_keys), "matrix:", Na_matrix.shape)
    print("Unique Nb states:", len(Nb_keys), "matrix:", Nb_matrix.shape)
    print("Unique Na-Nb states:", len(single_neural_dict_to_matrix(Na_Nb_transition_dict)[1]))
    print("Na binned values:", np.unique(Na_matrix))
    print("Nb binned values:", np.unique(Nb_matrix))
    _, _, _ = f2g.b_state_distribution_heatmap(all_visit_b_count_dict)
    rearranged_b_Na = connect_b_to_n(Na_b_transition_dict)
    rearranged_b_Nb = connect_b_to_n(Nb_b_transition_dict)
    _, _, _ = f2g.n_state_distribution_heatmap(rearranged_b_Na)
    _, _, _ = f2g.n_state_distribution_heatmap(rearranged_b_Nb)

    route = generate_connected_route(Na_Nb_b_transition_dict, route_length=50, seed=42)
    pearson_matrix, x_labels, y_labels, _, _ = build_similarity_matrix(Na_transition_dict, Nb_transition_dict, top_k=100, metric="pearson")
    plot_pearson_matrix(pearson_matrix, x_labels, y_labels, title="Pearson Correlation: RNN A vs RNN B", tick_step=10)
    euclidean_matrix, x_labels, y_labels, _, _ = build_similarity_matrix(Na_transition_dict, Nb_transition_dict, top_k=100, metric="euclidean")
    plot_euclidean_matrix(euclidean_matrix, x_labels, y_labels, title="Euclidean Distance: RNN A vs RNN B", tick_step=10)

    generate_manifold_from_dict(Na_transition_dict, dict_kind="Na", method="tsne", title="RNN A Transition Neural Manifold")
    generate_manifold_from_dict(Nb_transition_dict, dict_kind="Nb", method="tsne", title="RNN B Neural Transition Manifold")
    generate_manifold_from_dict(Na_Nb_transition_dict, dict_kind="Na_Nb", method="tsne", title="Joint A-B Neural Transition Manifold")