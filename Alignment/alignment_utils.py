import numpy as np
import statsmodels.api as sm
import pickle as pkl
from collections import Counter

def prime_lists(target_gender, dataset="ICSI"):
    """
    Parameters
        target_gender (str): "m" or "f"

    Returns
        tuple of original splits (split with female prime, split with male prime)
    """
    
    with open(f"../Corpora/{dataset}/in_between_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf


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