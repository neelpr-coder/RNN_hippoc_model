import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import perturbation_testing as pt
import cm_experiments as cme

import figure2_generation as f2g
import manifold_visualization as mv

if __name__ == "__main__":
    model = torch.load()
    b_transition_dict, all_visit_b_count_dict, Na_transition_dict, Nb_transition_dict, Na_Nb_transition_dict, Na_b_transition_dict, Nb_b_transition_dict, Na_Nb_b_transition_dict = cme,generate_dicts(model)
    rearranged_Na_b = f2g.build_b_to_n_map(Na_b_transition_dict)
    rearranged_Nb_b = f2g.build_b_to_n_map(Nb_b_transition_dict)
    _, _, _ = f2g.b_state_distribution_heatmap(b_transition_dict)
    _, _, _ = f2g.n_state_distribution_heatmap(rearranged_Na_b)
    _, _, _ = f2g.n_state_distribution_heatmap(rearranged_Nb_b)
    
    f2g.graph_neural_distribution(Na_transition_dict)
    f2g.graph_neural_distribution(Nb_transition_dict)
    pt.graph_behavioral_distribution(b_transition_dict, title="Connected RNN Behavioral Transition Distribution")
    route_seq = np.load("route_sequence_min100.npy", allow_pickle=True)