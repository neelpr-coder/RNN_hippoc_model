import numpy as np
import matplotlib.pyplot as plt
import figure2_generation as fig2
import small_model


def pearson_corr_between_vectors(n_state_a, n_state_b):
    """
    Compute Pearson correlation between two neural-state vectors.
    """

    vec_a = np.array(n_state_a, dtype=float)
    vec_b = np.array(n_state_b, dtype=float)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector shapes do not match: {vec_a.shape} vs {vec_b.shape}")

    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return np.nan

    return np.corrcoef(vec_a, vec_b)[0, 1]


def get_top_n_states(sweep_results, min_attempts, top_k=100):
    """
    Return top_k neural states by visit count for one min_attempts condition.
    """

    n_count_dict = sweep_results[min_attempts]["all_visit_count_n_dict"]

    sorted_items = sorted(
        n_count_dict.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_items = sorted_items[:top_k]

    n_states = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    return n_states, counts


def neural_state_pearson_confusion_matrix(
    sweep_results,
    min_attempts_a,
    min_attempts_b,
    top_k=100
):

    n_states_a, counts_a = get_top_n_states(
        sweep_results,
        min_attempts=min_attempts_a,
        top_k=top_k
    )

    n_states_b, counts_b = get_top_n_states(
        sweep_results,
        min_attempts=min_attempts_b,
        top_k=top_k
    )

    matrix = np.full((len(n_states_b), len(n_states_a)), np.nan)

    for i, n_b in enumerate(n_states_b):
        for j, n_a in enumerate(n_states_a):
            matrix[i, j] = pearson_corr_between_vectors(n_a, n_b)

    x_labels = [f"{min_attempts_a}:N{j + 1}" for j in range(len(n_states_a))]
    y_labels = [f"{min_attempts_b}:N{i + 1}" for i in range(len(n_states_b))]

    return matrix, x_labels, y_labels, counts_a, counts_b


def plot_neural_state_pearson_confusion_matrix(
    matrix,
    x_labels,
    y_labels,
    title,
    tick_step=10
):
    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="viridis"
    )

    plt.title(title)
    plt.xlabel("Neural states from first distribution")
    plt.ylabel("Neural states from second distribution")

    x_tick_positions = np.arange(0, len(x_labels), tick_step)
    y_tick_positions = np.arange(0, len(y_labels), tick_step)

    plt.xticks(
        x_tick_positions,
        [x_labels[i] for i in x_tick_positions],
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.yticks(
        y_tick_positions,
        [y_labels[i] for i in y_tick_positions],
        fontsize=8
    )

    plt.colorbar(im, label="Pearson correlation")
    plt.tight_layout()
    plt.show()


def neural_state_vector_similarity_heatmaps(top_k=100):
    model = fig2.small_model.RNN().to(fig2.device)

    sweep_results = fig2.sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )

    matrix_50_55, x_labels, y_labels, counts_50, counts_55 = neural_state_pearson_confusion_matrix(
        sweep_results,
        min_attempts_a=50,
        min_attempts_b=55,
        top_k=top_k
    )

    plot_neural_state_pearson_confusion_matrix(
        matrix_50_55,
        x_labels,
        y_labels,
        title=f"Neural-State Vector Similarity: min_attempts 50 vs 55, top {top_k}",
        tick_step=5
    )

    matrix_50_100, x_labels, y_labels, counts_50, counts_100 = neural_state_pearson_confusion_matrix(
        sweep_results,
        min_attempts_a=50,
        min_attempts_b=100,
        top_k=top_k
    )

    plot_neural_state_pearson_confusion_matrix(
        matrix_50_100,
        x_labels,
        y_labels,
        title=f"Neural-State Vector Similarity: min_attempts 50 vs 100, top {top_k}",
        tick_step=5
    )


def euclidean_distance_between_vectors(n_state_a, n_state_b):
    vec_a = np.array(n_state_a, dtype=float)
    vec_b = np.array(n_state_b, dtype=float)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")

    return np.linalg.norm(vec_a - vec_b)

def neural_state_euclidean_confusion_matrix(
    sweep_results,
    min_attempts_a,
    min_attempts_b,
    top_k=100
):
    """
    Creates a Euclidean distance matrix comparing top neural states from two min_attempts values.

    Rows = neural states from min_attempts_b
    Columns = neural states from min_attempts_a

    Lower distance = more similar neural-state vectors.
    """

    n_states_a, counts_a = get_top_n_states(
        sweep_results,
        min_attempts=min_attempts_a,
        top_k=top_k
    )

    n_states_b, counts_b = get_top_n_states(
        sweep_results,
        min_attempts=min_attempts_b,
        top_k=top_k
    )

    matrix = np.full((len(n_states_b), len(n_states_a)), np.nan)

    for i, n_b in enumerate(n_states_b):
        for j, n_a in enumerate(n_states_a):
            matrix[i, j] = euclidean_distance_between_vectors(n_a, n_b)

    x_labels = [f"{min_attempts_a}:N{j + 1}" for j in range(len(n_states_a))]
    y_labels = [f"{min_attempts_b}:N{i + 1}" for i in range(len(n_states_b))]

    return matrix, x_labels, y_labels, counts_a, counts_b

def plot_neural_state_euclidean_confusion_matrix(
    matrix,
    x_labels,
    y_labels,
    title,
    tick_step=10,
    cmap="viridis_r",
    vmin=0,
    vmax=4
):
    """
    Heatmap for Euclidean distance between neural-state vectors.

    Uses viridis_r so smaller distances appear brighter / more highlighted.
    """

    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.title(title)
    plt.xlabel("Neural states from first distribution")
    plt.ylabel("Neural states from second distribution")

    x_tick_positions = np.arange(0, len(x_labels), tick_step)
    y_tick_positions = np.arange(0, len(y_labels), tick_step)

    plt.xticks(
        x_tick_positions,
        [x_labels[i] for i in x_tick_positions],
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.yticks(
        y_tick_positions,
        [y_labels[i] for i in y_tick_positions],
        fontsize=8
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Euclidean distance")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    neural_state_vector_similarity_heatmaps(top_k=100)
    '''model = small_model.RNN().to(fig2.device)
    sweep_results = fig2.sweep_data_gen(
        model,
        step_size=5,
        min_attempts=50,
        max_attempts=101,
        sd=42
    )


    matrix_50_55_euc, x_labels_50, y_labels_55, counts_50, counts_55 = neural_state_euclidean_confusion_matrix(
            sweep_results,
            min_attempts_a=50,
            min_attempts_b=55,
            top_k=100
        )

    plot_neural_state_euclidean_confusion_matrix(
            matrix_50_55_euc,
            x_labels_50,
            y_labels_55,
            title="Euclidean Distance Between Top Neural States: min50 vs min55",
            tick_step=10
        )
    
    matrix_50_100_euc, x_labels_50, y_labels_100, counts_50, counts_100 = neural_state_euclidean_confusion_matrix(
            sweep_results,
            min_attempts_a=50,
            min_attempts_b=100,
            top_k=100,
        )

    plot_neural_state_euclidean_confusion_matrix(
            matrix_50_100_euc,
            x_labels_50,
            y_labels_100,
            title="Euclidean Distance Between Top Neural States: min50 vs min100",
            tick_step=10
        )'''