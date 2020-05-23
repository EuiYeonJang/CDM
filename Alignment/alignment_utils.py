import pickle as pkl
import os


#################################################
########## ONE TIME PREPROCESSING STEP ##########
#################################################
def __expand_split_list(split_list, prime_gender):
    expanded_split = list()
    b_str = f"list_{prime_gender}b"

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0:
            if sum(adj_pair["a"].values()) > 0:
                expanded_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

            for ab in adj_pair[b_str]:
                if sum(ab.values()) > 0:
                    expanded_split.append((prime_gender, ab, adj_pair["b"], sum(ab.values())))

    return expanded_split


def __clean_split_list(split_list, prime_gender):
    clean_split = list()
    b_str = f"list_{prime_gender}b"

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0 and sum(adj_pair["a"].values()) > 0:
            clean_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

    return clean_split


def prep_in_between_split_list(dataset):
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    e_mf = __expand_split_list(mf, "m")
    e_mm = __expand_split_list(mm, "m")
    e_fm = __expand_split_list(fm, "f")
    e_ff = __expand_split_list(ff, "f")
    
    with open(f"../Corpora/{dataset}/processed_in_between_split.pkl", "wb") as f:
        pkl.dump((e_mf, e_mm, e_fm, e_ff), f)


def prep_clean_split_list(dataset):
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    c_mf = __clean_split_list(mf, "m")
    c_mm = __clean_split_list(mm, "m")
    c_fm = __clean_split_list(fm, "f")
    c_ff = __clean_split_list(ff, "f")
    
    with open(f"../Corpora/{dataset}/processed_clean_split.pkl", "wb") as f:
        pkl.dump((c_mf, c_mm, c_fm, c_ff), f)


def prep(dataset):
    c_filename = f"../Corpora/{dataset}/processed_clean_split.pkl"
    e_filename = f"../Corpora/{dataset}/processed_in_between_split.pkl"
    
    if not os.path.exists(c_filename):
        prep_clean_split_list(dataset)

    if not os.path.exists(e_filename):
        prep_in_between_split_list(dataset)

    
#################################################
#################################################


def create_vocab(adj_pair_list):
    vocab = set()

    for _, prime, target, _ in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab


def prime_lists(target_gender, dataset, between):
    split_type = "in_between" if between else "clean"

    with open(f"../Corpora/{dataset}/processed_{split_type}_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf
