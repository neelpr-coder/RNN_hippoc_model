import os
import pandas as pd
from collections import defaultdict

def image_preproccesing(file_path = os.path.expanduser("~/Desktop/unity/RNN_images/image_data.csv")): 
    """preprocess the csv file to create tuple of 
    (x,y,heading) behavioral states, associated image, isValid, and visit_count that can be unpacked in initial training"""
    if not os.path.exists(file_path):
        raise ValueError("not a valid file")

    df = pd.read_csv(file_path, sep=',', engine='python')
    #print(df.columns)
    b_state_image_path_dict = defaultdict(list)
    all_visit_counts_dict = defaultdict(int)
    for row in df.itertuples(index=False):
        b_state = (row.x, row.y, row.heading)
        b_state_image_path_dict[b_state].append(row.image_path)

        if b_state not in all_visit_counts_dict:
            all_visit_counts_dict[b_state] = row.visit_count
    
    return b_state_image_path_dict, all_visit_counts_dict

'''test_a, test_b = image_preproccesing()
print(len(test_a))
print(test_a)'''

'''test, visit_dict = image_preproccesing()
print(test[10])
print(visit_dict[(0,11,0)])
print(len(test))'''