import pickle as pkl
import os


######################################################
########## PREPROCESSING STEP FOR Xu et al. ##########
######################################################
def __expand_split_list(split_list, prime_gender, lexical=False):
    expanded_split = list()
    b_str = f"list_{prime_gender}b"

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0:
            if sum(adj_pair["a"].values()) > 0:
                if lexical:
                    for k in adj_pair['a'].keys():
                        expanded_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                else:
                    expanded_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

            for ab in adj_pair[b_str]:
                if sum(ab.values()) > 0:
                    if lexical:
                        for k in ab.keys():
                            expanded_split.append((prime_gender, ab[k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                    else:
                        expanded_split.append((prime_gender, ab, adj_pair["b"], sum(ab.values())))

    return expanded_split


def __clean_split_list(split_list, prime_gender, lexical=False):
    clean_split = list()

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0 and sum(adj_pair["a"].values()) > 0:
            if lexical:
                for k in adj_pair['a'].keys():
                    clean_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
            else:
                clean_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

    return clean_split


def prep_in_between_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I prepare expanded split list...")
    if dataset == "ICSI":
        with open(f"../Corpora/{dataset}/f_f_short.pkl", "rb") as f:
            ff = pkl.load(f)

        with open(f"../Corpora/{dataset}/f_m_short.pkl", "rb") as f:
            mf = pkl.load(f)

        with open(f"../Corpora/{dataset}/m_m_short.pkl", "rb") as f:
            mm = pkl.load(f)

        with open(f"../Corpora/{dataset}/m_f_short.pkl", "rb") as f:
            fm = pkl.load(f)

    else:
        with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
            mf, mm, fm, ff = pkl.load(f)

    e_mf = __expand_split_list(mf, "m", lexical)
    e_mm = __expand_split_list(mm, "m", lexical)
    e_fm = __expand_split_list(fm, "f", lexical)
    e_ff = __expand_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_in_between_split.pkl", "wb") as f:
        pkl.dump((e_mf, e_mm, e_fm, e_ff), f)


def prep_clean_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I prepare clean split list...")
    if dataset == "ICSI":
        with open(f"../Corpora/{dataset}/f_f_short.pkl", "rb") as f:
            ff = pkl.load(f)

        with open(f"../Corpora/{dataset}/f_m_short.pkl", "rb") as f:
            mf = pkl.load(f)

        with open(f"../Corpora/{dataset}/m_m_short.pkl", "rb") as f:
            mm = pkl.load(f)

        with open(f"../Corpora/{dataset}/m_f_short.pkl", "rb") as f:
            fm = pkl.load(f)

    else:
        with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
            mf, mm, fm, ff = pkl.load(f)

    c_mf = __clean_split_list(mf, "m", lexical)
    c_mm = __clean_split_list(mm, "m", lexical)
    c_fm = __clean_split_list(fm, "f", lexical)
    c_ff = __clean_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_clean_split.pkl", "wb") as f:
        pkl.dump((c_mf, c_mm, c_fm, c_ff), f)


def prep(dataset, lexical=False):
    print(f"Prepping the dataset for lexical alignment,... at least I'm supposed to be? Is that True? {lexical}")
    alignment_type = "lexical" if lexical else "stylistic"

    c_filename = f"../Corpora/{dataset}/{alignment_type}_processed_clean_split.pkl"
    e_filename = f"../Corpora/{dataset}/{alignment_type}_processed_in_between_split.pkl"
    
    if not os.path.exists(c_filename):
        prep_clean_split_list(dataset, lexical)

    if not os.path.exists(e_filename):
        prep_in_between_split_list(dataset, lexical)

###############################################################
###############################################################

###########################################################
########## PREPROCESSING STEP FOR Danescu et al. ##########
###########################################################
# def __clean_split_list_danescu(split_list, prime_gender):
#     clean_split = list()

#     for adj_pair in split_list:
#         if sum(adj_pair["b"].values()) > 0 and sum(adj_pair["a"].values()) > 0:
#             clean_split.append((set(adj_pair["a"].keys()), set(adj_pair["b"].keys())))

#     return clean_split

# def prep_clean_split_list_danescu(dataset):
#     print("Please wait while I prepare clean split list...")
#     if dataset == "ICSI":
#         with open(f"../Corpora/{dataset}/f_f.pkl", "rb") as f:
#             ff = pkl.load(f)

#         with open(f"../Corpora/{dataset}/f_m.pkl", "rb") as f:
#             mf = pkl.load(f)

#         with open(f"../Corpora/{dataset}/m_m.pkl", "rb") as f:
#             mm = pkl.load(f)

#         with open(f"../Corpora/{dataset}/m_f.pkl", "rb") as f:
#             fm = pkl.load(f)

#     else:
#         with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
#             mf, mm, fm, ff = pkl.load(f)

#     c_mf = __clean_split_list_danescu(mf, "m")
#     c_mm = __clean_split_list_danescu(mm, "m")
#     c_fm = __clean_split_list_danescu(fm, "f")
#     c_ff = __clean_split_list_danescu(ff, "f")
    
#     with open(f"../Corpora/{dataset}/danescu_processed_clean_split.pkl", "wb") as f:
#         pkl.dump((c_mf, c_mm, c_fm, c_ff), f)


# def prep_danescu(dataset):
#     c_filename = f"../Corpora/{dataset}/danescu_processed_clean_split.pkl"
#     e_filename = f"../Corpora/{dataset}/danescu_processed_in_between_split.pkl"
    
#     if not os.path.exists(c_filename):
#         prep_clean_split_list(dataset)

#     # if not os.path.exists(e_filename):
#         # prep_in_between_split_list(dataset)

###########################################################
###########################################################


def create_vocab(adj_pair_list):
    print("Creating vocab...")
    vocab = set()

    for _, prime, target, _ in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab


def prime_lists(target_gender, dataset, between, lexical=False):
    split_type = "in_between" if between else "clean"
    alignment_type = "lexical" if lexical else "stylistic"

    with open(f"../Corpora/{dataset}/{alignment_type}_processed_{split_type}_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf
