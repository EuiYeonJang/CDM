import numpy as np
import statsmodels.api as sm
import pickle as pkl
from collections import Counter


def prime_lists(target_gender, dataset):
    with open(f"../Corpora/{dataset}/in_between_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf


# def prep_target_apl(target_gender, dataset):
#     pf_l, pm_l = prime_lists(target_gender, dataset)

#     pf_apl = list()
#     pm_apl = list()

#     for pair in pf_l:
#         if sum(pair["a"].values()) > 0 and sum(pair["b"].values()) > 0:
#             pf_apl.append((pair["a"], pair["b"]))

#     for pair in pm_l:
#         if sum(pair["a"].values()) > 0 and sum(pair["b"].values()) > 0:
#             pm_apl.append((pair["a"], pair["b"]))
        

# def prep_all_apl(dataset):
#     fm, mm = prep_target_apl("m", dataset)


    


# def preprocess_split(dataset):
    
#     return


def create_vocab(adj_pair_list):
    """
    Parameters: 
        adj_pair_list -list of tuples (gender of prime, prime counter, target counter)
    
    Returns:
        set of vocab
    """
    vocab = set()

    for _, prime, target in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab