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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

TRANSITION_LABELS = ["Na", "Nb", "Na_b", "Nb_b", "Na_Nb", "Na_Nb_b"]

def get_connected_transition_keys(step):
    B0, Na0, Nb0, B1, Na1, Nb1, _ = step
    return {
        "Na": (Na0, Na1),
        "Nb": (Nb0, Nb1),
        "Na_b": ((Na0, B0), (Na1, B1)),
        "Nb_b": ((Nb0, B0), (Nb1, B1)),
        "Na_Nb": ((Na0, Nb0), (Na1, Nb1)),
        "Na_Nb_b": ((Na0, Nb0, B0), (Na1, Nb1, B1))
    }

def transition_probability_or_nan(next_counts, next_key):
    total = sum(next_counts.values())
    return np.nan if total == 0 else next_counts.get(next_key, 0) / total

def calc_connected_stage3_error_history(route, updated_dicts):
    error_history = []

    for frame, step in enumerate(route, start=1):
        row = {"frame": frame}
        transition_keys = get_connected_transition_keys(step)

        for name in TRANSITION_LABELS:
            current_state, route_next_state = transition_keys[name]
            next_counts = updated_dicts[name].get(current_state, {})
            total = sum(next_counts.values())

            if total == 0:
                row[name] = np.nan
                row[f"{name}_top1_prob"] = np.nan
                row[f"{name}_route_prob"] = np.nan
                row[f"{name}_route_is_top1"] = False
                continue

            top1_state, top1_count = max(next_counts.items(), key=lambda item: item[1])
            top1_prob = top1_count / total
            route_prob = next_counts.get(route_next_state, 0) / total

            row[name] = top1_prob - route_prob
            row[f"{name}_top1_prob"] = top1_prob
            row[f"{name}_route_prob"] = route_prob
            row[f"{name}_top1_state"] = top1_state
            row[f"{name}_route_is_top1"] = np.isclose(route_prob, top1_prob)

        error_history.append(row)

    return error_history

def plot_connected_stage3_error(error_history, title="Connected RNN Stage 3 Error", save_path=None, ax=None):
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(12, 7))

    frames = [row["frame"] for row in error_history]
    for label in TRANSITION_LABELS:
        ax.plot(frames, [row[label] for row in error_history], marker="o", markersize=3, linewidth=1.5, label=label)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time step along original route")
    ax.set_ylabel("Stage 3 error\n(top-1 probability − route probability)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if created_figure:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

def plot_all_stage3_conditions(stage3_histories, save_path=None):
    titles = {"a": "Region A Knockout", "b": "Region B Knockout", "both": "Regions A and B Knockout"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 16), sharex=True)

    for ax, region in zip(axes, ("a", "b", "both")):
        plot_connected_stage3_error(stage3_histories[region], title=titles[region], ax=ax)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    fig.suptitle("Connected RNN Stage 3 Error", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_connected_stage2_error_overlay(error_history, title="Connected RNN Stage 2 Error", save_path=None, ax=None):
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(12, 7))

    frames = [row["frame"] for row in error_history]

    for label in TRANSITION_LABELS:
        ax.plot(frames, [row[label] for row in error_history], linewidth=1.5, label=label)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Perturbation step")
    ax.set_ylabel("Stage 2 error\n(original top-1 − updated top-1)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if created_figure:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

def plot_all_stage2_conditions(stage2_histories, save_path=None):
    titles = {"a": "Region A Knockout", "b": "Region B Knockout", "both": "Regions A and B Knockout"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 16), sharex=True)

    for ax, region in zip(axes, ("a", "b", "both")):
        plot_connected_stage2_error_overlay(stage2_histories[region], title=titles[region], ax=ax)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    fig.suptitle("Connected RNN Stage 2 Error", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

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

def connected_state_to_vector(state, dict_kind):
    if dict_kind in ("Na", "Nb"):
        return np.asarray(state, dtype=float)
    if dict_kind == "B":
        return mv.behavioral_state_to_vector(state)
    if dict_kind in ("Na_b", "Nb_b"):
        neural_state, behavioral_state = state
        neural_vector = np.asarray(neural_state, dtype=float)
        behavioral_vector = mv.behavioral_state_to_vector(behavioral_state)
        return np.concatenate([neural_vector, behavioral_vector])
    if dict_kind == "Na_Nb":
        Na, Nb = state
        return np.concatenate([np.asarray(Na, dtype=float), np.asarray(Nb, dtype=float)])
    if dict_kind == "Na_Nb_b":
        Na, Nb, B = state
        return np.concatenate([np.asarray(Na, dtype=float), np.asarray(Nb, dtype=float), mv.behavioral_state_to_vector(B)])

    raise ValueError("dict_kind must be 'Na', 'Nb', 'B', 'Na_b', 'Nb_b', 'Na_Nb', or 'Na_Nb_b'")

def connected_dict_to_matrix(transition_dict, dict_kind):
    state_set = set()

    for current_state, next_states in transition_dict.items():
        state_set.add(current_state)
        state_set.update(next_states.keys())

    state_keys = list(state_set)
    matrix = np.asarray([connected_state_to_vector(state, dict_kind) for state in state_keys], dtype=float)

    return matrix, state_keys


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

def extract_connected_state_path(route_sequence, dict_kind):
    if len(route_sequence) == 0:
        return []

    B0, Na0, Nb0, _, _, _, _ = route_sequence[0]

    if dict_kind == "B":
        return [B0] + [B1 for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Na":
        return [Na0] + [Na1 for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Nb":
        return [Nb0] + [Nb1 for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Na_b":
        return [(Na0, B0)] + [(Na1, B1) for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Nb_b":
        return [(Nb0, B0)] + [(Nb1, B1) for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Na_Nb":
        return [(Na0, Nb0)] + [(Na1, Nb1) for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]
    if dict_kind == "Na_Nb_b":
        return [(Na0, Nb0, B0)] + [(Na1, Nb1, B1) for B0, Na0, Nb0, B1, Na1, Nb1, count in route_sequence]

    raise ValueError( "dict_kind must be 'Na', 'Nb', 'B', 'Na_b','Nb_b', 'Na_Nb', or 'Na_Nb_b'")

def plot_3d_embedding(embedding, state_keys, transition_dict, route_sequence, dict_kind, title, method="tsne", save_path=None, visual_offset_fraction=0.015):
    actual_state_path = extract_connected_state_path(route_sequence, dict_kind)
    if len(actual_state_path) < 2:
        raise ValueError("Route must contain at least one transition.")

    predicted_state_path = [actual_state_path[0]]
    for frame, current_state in enumerate(actual_state_path[:-1], start=1):
        next_counts = transition_dict.get(current_state, {})
        if len(next_counts) == 0:
            raise ValueError(f"No outgoing {dict_kind} transitions for route frame {frame}: {current_state}")
        predicted_state_path.append(max(next_counts.items(), key=lambda item: item[1])[0])

    key_to_idx = {make_hashable_state(key): i for i, key in enumerate(state_keys)}
    missing_actual = [state for state in actual_state_path if make_hashable_state(state) not in key_to_idx]
    missing_predicted = [state for state in predicted_state_path if make_hashable_state(state) not in key_to_idx]
    if missing_actual:
        raise ValueError(f"{len(missing_actual)} actual route states were not found in the {dict_kind} manifold keys.")
    if missing_predicted:
        raise ValueError(f"{len(missing_predicted)} top-1 states were not found in the {dict_kind} manifold keys.")

    actual_indices = [key_to_idx[make_hashable_state(state)] for state in actual_state_path]
    predicted_indices = [key_to_idx[make_hashable_state(state)] for state in predicted_state_path]
    actual_coords = embedding[actual_indices]
    predicted_coords = embedding[predicted_indices]

    matching_frames = [i for i in range(1, len(actual_state_path)) if make_hashable_state(actual_state_path[i]) == make_hashable_state(predicted_state_path[i])]
    mismatch_frames = [i for i in range(1, len(actual_state_path)) if i not in matching_frames]
    transition_count = len(actual_state_path) - 1
    print(f"{dict_kind}: top-1 matches = {len(matching_frames)}/{transition_count}")

    data_span = embedding.max(axis=0) - embedding.min(axis=0)
    offset = np.array([visual_offset_fraction * data_span[0], visual_offset_fraction * data_span[1], 0.0])
    actual_plot_coords = actual_coords - offset
    predicted_plot_coords = predicted_coords + offset

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=10, alpha=0.18, color="black", label=f"All {dict_kind} states", depthshade=False)
    ax.plot(actual_plot_coords[:, 0], actual_plot_coords[:, 1], actual_plot_coords[:, 2], linewidth=2.2, color="crimson", label="Actual route", zorder=6)
    ax.plot(predicted_plot_coords[:, 0], predicted_plot_coords[:, 1], predicted_plot_coords[:, 2], linewidth=2.2, linestyle="--", color="deepskyblue", label="Top-1 transition route", zorder=7)

    if mismatch_frames:
        actual_mismatch = actual_plot_coords[mismatch_frames]
        predicted_mismatch = predicted_plot_coords[mismatch_frames]
        ax.scatter(actual_mismatch[:, 0], actual_mismatch[:, 1], actual_mismatch[:, 2], s=45, color="crimson", marker="o", label="Actual next state", zorder=10, depthshade=False)
        ax.scatter(predicted_mismatch[:, 0], predicted_mismatch[:, 1], predicted_mismatch[:, 2], s=60, color="deepskyblue", marker="^", label="Top-1 next state", zorder=11, depthshade=False)

    if matching_frames:
        matching_coords = actual_coords[matching_frames]
        ax.scatter(matching_coords[:, 0], matching_coords[:, 1], matching_coords[:, 2], s=85, color="gold", edgecolor="black", linewidth=0.8, label=f"Same next state ({len(matching_frames)}/{transition_count})", zorder=12, depthshade=False)

    method_label = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}.get(method.lower(), method.upper())
    ax.set_title(title)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.set_zlabel(f"{method_label} 3")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def generate_manifold_from_dict(transition_dict, dict_kind, route_sequence, method="tsne", title=None, save_path=None):
    valid_kinds = {"Na", "Nb", "B", "Na_b", "Nb_b", "Na_Nb", "Na_Nb_b"}

    if dict_kind not in valid_kinds:
        raise ValueError("dict_kind must be 'Na', 'Nb', 'B', 'Na_b', 'Nb_b', 'Na_Nb', or 'Na_Nb_b'")

    matrix, state_keys = connected_dict_to_matrix(transition_dict, dict_kind)

    if len(state_keys) < 4:
        raise ValueError(
            f"{dict_kind} contains only {len(state_keys)} unique states; "
            "at least 4 are needed for a 3D embedding."
        )

    embedding = mv.perform_manifold_learning(matrix, method=method, n_components=3)

    if title is None:
        title = f"{dict_kind} manifold ({method})"

    plot_3d_embedding(embedding, state_keys, transition_dict, route_sequence, dict_kind, title, method=method, save_path=save_path)

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
    route = generate_connected_route(Na_Nb_b_transition_dict, route_length=50, seed=42)
    '''original_dicts = {
        "Na": Na_transition_dict,
        "Nb": Nb_transition_dict,
        "Na_b": Na_b_transition_dict,
        "Nb_b": Nb_b_transition_dict,
        "Na_Nb": Na_Nb_transition_dict,
        "Na_Nb_b": Na_Nb_b_transition_dict
    }
    perturbation_results = cme.run_connected_knockout_experiments(
                                                                    model,
                                                                    Na_transition_dict,
                                                                    Nb_transition_dict,
                                                                    Na_b_transition_dict,
                                                                    Nb_b_transition_dict,
                                                                    Na_Nb_transition_dict,
                                                                    Na_Nb_b_transition_dict,
                                                                    total_perturbations=1000,
                                                                    sd=42,
                                                                    save_dir=os.path.join(SCRIPT_DIR, "Perturbation_Connected_Cache")
                                                                )
    stage2_histories = {region: perturbation_results[region]["stage2_error_history"] for region in ("a", "b", "both")}
    for region in ("a", "b", "both"):
        plot_connected_stage2_error_overlay(
            stage2_histories[region],
            title=f"Stage 2 Error During {region.upper()} Knockout",
            save_path=None
        )
    plot_all_stage2_conditions(stage2_histories, save_path="connected_stage2_error_all_conditions.png")

    stage3_histories = {}
    for region in ("a", "b", "both"):
        updated_dicts = perturbation_results[region]["dicts"]
        stage3_histories[region] = calc_connected_stage3_error_history(route, updated_dicts)
        plot_connected_stage3_error(
            stage3_histories[region],
            title=f"Stage 3 Error After {region.upper()} Knockout",
            save_path=None
        )
    plot_all_stage3_conditions(stage3_histories, save_path="connected_stage3_error_all_conditions.png")'''

    #Na_matrix, Na_keys = single_neural_dict_to_matrix(Na_transition_dict)
    #Nb_matrix, Nb_keys = single_neural_dict_to_matrix(Nb_transition_dict)
    '''print("Unique Na states:", len(Na_keys), "matrix:", Na_matrix.shape)
    print("Unique Nb states:", len(Nb_keys), "matrix:", Nb_matrix.shape)
    print("Unique Na-Nb states:", len(single_neural_dict_to_matrix(Na_Nb_transition_dict)[1]))
    print("Na binned values:", np.unique(Na_matrix))
    print("Nb binned values:", np.unique(Nb_matrix))'''
    #_, _, _ = f2g.b_state_distribution_heatmap(all_visit_b_count_dict)
    #rearranged_b_Na = connect_b_to_n(Na_b_transition_dict)
    #rearranged_b_Nb = connect_b_to_n(Nb_b_transition_dict)
    #_, _, _ = f2g.n_state_distribution_heatmap(rearranged_b_Na)
    #_, _, _ = f2g.n_state_distribution_heatmap(rearranged_b_Nb)

    #stage2_error_history = calc_connected_stage2_error_history(route, original_dicts)
    #plot_connected_stage_2_error_overlay(stage2_error_history, title="Stage 2 Error Before Perturbation", save_path=None)

    #pearson_matrix, x_labels, y_labels, _, _ = build_similarity_matrix(Na_transition_dict, Nb_transition_dict, top_k=100, metric="pearson")
    #plot_pearson_matrix(pearson_matrix, x_labels, y_labels, title="Pearson Correlation: RNN A vs RNN B", tick_step=10)
    #euclidean_matrix, x_labels, y_labels, _, _ = build_similarity_matrix(Na_transition_dict, Nb_transition_dict, top_k=100, metric="euclidean")
    #plot_euclidean_matrix(euclidean_matrix, x_labels, y_labels, title="Euclidean Distance: RNN A vs RNN B", tick_step=10)

    '''generate_manifold_from_dict(Na_transition_dict, dict_kind="Na", route_sequence=route, method="tsne", title="RNN A Transition Neural Manifold")
    generate_manifold_from_dict(Nb_transition_dict, dict_kind="Nb", route_sequence=route, method="tsne", title="RNN B Neural Transition Manifold")
    generate_manifold_from_dict(Na_Nb_transition_dict, dict_kind="Na_Nb", route_sequence=route, method="tsne", title="Joint A-B Neural Transition Manifold")
    generate_manifold_from_dict(Na_transition_dict, dict_kind="Na", route_sequence=route, method="umap", title="RNN A Transition Neural Manifold")
    generate_manifold_from_dict(Nb_transition_dict, dict_kind="Nb", route_sequence=route, method="umap", title="RNN B Neural Transition Manifold")
    generate_manifold_from_dict(Na_Nb_transition_dict, dict_kind="Na_Nb", route_sequence=route, method="umap", title="Joint A-B Neural Transition Manifold")
    generate_manifold_from_dict(Na_transition_dict, dict_kind="Na", route_sequence=route, method="pca", title="RNN A Transition Neural Manifold")
    generate_manifold_from_dict(Nb_transition_dict, dict_kind="Nb", route_sequence=route, method="pca", title="RNN B Neural Transition Manifold")
    generate_manifold_from_dict(Na_Nb_transition_dict, dict_kind="Na_Nb", route_sequence=route, method="pca", title="Joint A-B Neural Transition Manifold")
    generate_manifold_from_dict(Na_b_transition_dict, dict_kind="Na_b", route_sequence=route, method="tsne", title="RNN A & Behavioral State Transition t-SNE Manifold")
    generate_manifold_from_dict(Na_b_transition_dict, dict_kind="Na_b", route_sequence=route, method="umap", title="RNN A & Behavioral State Transition UMAP Manifold")
    generate_manifold_from_dict(Na_b_transition_dict, dict_kind="Na_b", route_sequence=route, method="pca", title="RNN A & Behavioral State Transition PCA Manifold")
    generate_manifold_from_dict(Nb_b_transition_dict, dict_kind="Nb_b", route_sequence=route, method="tsne", title="RNN B & Behavioral State Transition t-SNE Manifold")
    generate_manifold_from_dict(Nb_b_transition_dict, dict_kind="Nb_b", route_sequence=route, method="umap", title="RNN B & Behavioral State Transition UMAP Manifold")
    generate_manifold_from_dict(Nb_b_transition_dict, dict_kind="Nb_b", route_sequence=route, method="pca", title="RNN B & Behavioral State Transition PCA Manifold")
    generate_manifold_from_dict(Na_Nb_b_transition_dict, dict_kind="Na_Nb_b", route_sequence=route, method="tsne", title="RNN A, RNN B, & Behavioral State Transition t-SNE Manifold")
    generate_manifold_from_dict(Na_Nb_b_transition_dict, dict_kind="Na_Nb_b", route_sequence=route, method="umap", title="RNN A, RNN B, & Behavioral State Transition UMAP Manifold")
    generate_manifold_from_dict(Na_Nb_b_transition_dict, dict_kind="Na_Nb_b", route_sequence=route, method="pca", title="RNN A, RNN B, & Behavioral State Transition PCA Manifold")'''