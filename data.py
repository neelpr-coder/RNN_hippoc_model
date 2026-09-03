import os
import pandas as pd
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.environ.get("RNN_IMAGE_DIR", "/Users/neelprabhakar/Desktop/unity/10x10")

def image_preproccesing(file_path = None): 
    """preprocess the csv file to create tuple of 
    (x,y,heading) behavioral states, associated image, isValid, and visit_count that can be unpacked in initial training"""
    if file_path is None:
        data_dir = os.environ.get("RNN_DATA_DIR", os.path.join(SCRIPT_DIR, "10x10_dataset"))
        file_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(file_path):
        raise ValueError(
            f"Metadata file not found: {file_path}\n"
            "Place the Unity dataset in the repository's data/ folder "
            "or set the RNN_DATA_DIR environment variable."
        )

    df = pd.read_csv(file_path, sep=',', engine='python')
    #print(df.columns)
    b_state_image_path_dict = defaultdict(list)
    all_visit_counts_dict = defaultdict(int)

    image_dir = IMAGE_DIR
    for row in df.itertuples(index=False):
        b_state = (int(row.x + 4.5), int(row.z + 4.5), int(row.rotation_index))

        img_path = os.path.join(image_dir, row.filename)
        b_state_image_path_dict[b_state].append(img_path)

        if b_state not in all_visit_counts_dict:
            all_visit_counts_dict[b_state] = 0
    
    return b_state_image_path_dict, all_visit_counts_dict

'''test_a, test_b = image_preproccesing()
print(len(test_a))
print(test_a)'''

'''test, visit_dict = image_preproccesing()
print(test[10])
print(visit_dict[(0,11,0)])
print(len(test))'''