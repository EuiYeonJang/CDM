import pickle as pkl
import numpy as np
from sklearn.linear_model import LogisticRegression
  
# create subsets
def create_subsets(filename="subset_dict.pkl"):

    # load parsed file
    with open("Corpora/ICSI/adjacency_list.pkl", "rb") as f:
        adj_list = pkl.load(f)
    
    subset = dict(mm=list(),
                    mf=list(),
                    fm=list(),
                    ff=list())

    for item in adj_list:
        gender_a = item["a"]["gender"] 
        gender_b = item["b"]["gender"] 
        if gender_a == "m" and gender_b == "m":
            subset_name = "mm"
        elif gender_a == "m" and gender_b == "f":
            subset_name = "mf"
        elif gender_a == "f" and gender_b == "m":
            subset_name = "fm"
        else: # gneder_a == "f" and gender_b == "f"
            subset_name = "ff"

        subset[subset_name].append((item["a"]["counter"], item["b"]["counter"]))

    with open(f"{filename}", "wb") as f:
        pkl.dump(subset, f)
