import figure2_generation as f2g
import math
import random
import os
import torch
import numpy as np
import small_model

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def rand_gen_start_goal(dictionary):
    all_keys = set()
    all_possible_starts = set()
    for key1 in dictionary:
        all_keys.add(key1)
        if dictionary[key1]: 
            all_possible_starts.add(key1)
        else: continue 
        for key2 in dictionary[key1]:
            all_keys.add(key2)
    start = random.choice(list(all_possible_starts))
    goal = random.choice(list(all_keys))
    return start, goal

def get_lowest_cost_node(open_list):
    return open_list[0] if open_list else None

def convert_probability_to_cost(prob):
    if prob <= 0:
        return float('inf')
    return -math.log(prob)

def get_neighbors(prob_dictionary, key1):
    return [(convert_probability_to_cost(probability), next_node) for next_node, probability in prob_dictionary[key1].items()]


def a_star_search(dictionary, start, goal):
    # cost eval by -log(probability) where higher probability edges have lower cost and vice versa
    prob_dictionary = f2g.convert_count_to_probability(dictionary)

    if start == goal:
        print("Start is the same as goal. Total Steps Taken: 1")
        return 1

    came_from = {start: None} # track taken path via list of (cost, node) tuples
    open_list = []
    closed_list = set() # store nodes only

    node = (0, start) # (cost, node) tuple

    g_scores = {start: 0}  # Dictionary to store g scores for each node
    open_list.append(node) 
    cur_g = g_scores[start]

    while open_list:
        # Get the node with the lowest cost
        cheapest_node = get_lowest_cost_node(open_list)
        open_list.remove(cheapest_node)
        cur_g = g_scores[cheapest_node[1]]
        if cheapest_node[1] == goal:
            # Reconstruct the path
            total_path = []
            current = goal
            while current is not None:
                total_path.append(current)
                current = came_from[current]
            total_path.reverse()
            print("Path found:", total_path)
            print("Total Steps Taken:", len(total_path))
            return len(total_path)
        # get next nodes from cheapest nodes
        neighbors = get_neighbors(prob_dictionary, cheapest_node[1]) # gives list of sucessor cost and keys from the cheapest node
        for next_node in neighbors:
            temp_g = cur_g + next_node[0]  # Calculate the temp cur g score for the neighbor
            h = 0  # Heuristic temp set to 0 for uniform cost search
            f = temp_g + h
            if next_node[1] in closed_list:
                continue  # Skip if the neighbor is already evaluated
            if next_node[1] not in [n[1] for n in open_list]:
                # If the neighbor is not in the open list, add it
                open_list.append((f, next_node[1]))
                came_from[next_node[1]] = cheapest_node[1]  # Track the path
                g_scores[next_node[1]] = temp_g  # Update g score for the neighbor
            else:
                # If neighbor in open list, check if this path is better
                existing_node = next(n for n in open_list if n[1] == next_node[1])
                if temp_g < g_scores[next_node[1]]:
                    # Update the g score and path if this path is better
                    g_scores[next_node[1]] = temp_g
                    came_from[next_node[1]] = cheapest_node[1]
                    # Update the cost in the open list
                    open_list.remove(existing_node)
                    open_list.append((temp_g, next_node[1]))

        closed_list.add(cheapest_node[1])
        open_list.sort(key=lambda x: x[0]) # sort by cost
    print("no path found")
    return 0

def perturbed_search(dictionary, start, stop):
    # find chepaest node at each time step out of all the nodes and then 
    ...

if __name__ == "__main__":
    SEED = 42
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    model = small_model.RNN().to(device)
    model.eval()
    
    BASELINE_MODEL_PATH = os.path.join(SCRIPT_DIR, "post_stage1_model_sd42.pt")
    
    torch.save(model.state_dict(), BASELINE_MODEL_PATH)
    
    print(f"[Log] Baseline model saved to {BASELINE_MODEL_PATH}")

    pair_transition_dict, behavioral_transition_dict, neural_transition_dict, all_visit_count_b_dict, all_visit_count_n_dict = f2g.generate_dicts(model)

    start, stop = rand_gen_start_goal(neural_transition_dict)
    result_a_star = a_star_search(neural_transition_dict, start, stop)

