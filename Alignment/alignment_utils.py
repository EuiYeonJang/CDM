import pickle as pkl
import os


######################################################
########## PREPROCESSING STEP FOR Xu et al. ##########
######################################################
def __between_split_list(split_list, prime_gender, lexical=False):
    expanded_split = list()
    b_str = f"list_{prime_gender}b"

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0:
            if sum(adj_pair["a"].values()) > 0:
                if lexical:
                    keyz = list(adj_pair['a'].keys()) + list(adj_pair['b'].keys())
                    for k in keyz:
                        expanded_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                else:
                    expanded_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

            for ab in adj_pair[b_str]:
                if sum(ab.values()) > 0:
                    if lexical:
                        keyz = list(ab.keys()) + list(adj_pair['b'].keys())
                        for k in keyz:
                            expanded_split.append((prime_gender, ab[k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                    else:
                        expanded_split.append((prime_gender, ab, adj_pair["b"], sum(ab.values())))

    return expanded_split


def __plain_split_list(split_list, prime_gender, lexical=False):
    clean_split = list()

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0 and sum(adj_pair["a"].values()) > 0:
            if lexical:
                keyz = list(adj_pair['a'].keys()) + list(adj_pair['b'].keys())
                for k in keyz:
                    clean_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
            else:
                clean_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

    return clean_split


def __prep_between_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I first prepare between split list...")
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    e_mf = __between_split_list(mf, "m", lexical)
    e_mm = __between_split_list(mm, "m", lexical)
    e_fm = __between_split_list(fm, "f", lexical)
    e_ff = __between_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_between_split.pkl", "wb") as f:
        pkl.dump((e_mf, e_mm, e_fm, e_ff), f)


def __prep_plain_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I first prepare plain split list...")
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    c_mf = __plain_split_list(mf, "m", lexical)
    c_mm = __plain_split_list(mm, "m", lexical)
    c_fm = __plain_split_list(fm, "f", lexical)
    c_ff = __plain_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_plain_split.pkl", "wb") as f:
        pkl.dump((c_mf, c_mm, c_fm, c_ff), f)


def prep_dataset(dataset, between, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"
    adj_pair_type = "between" if between else "plain"

    c_filename = f"../Corpora/{dataset}/{alignment_type}_processed_plain_split.pkl"
    e_filename = f"../Corpora/{dataset}/{alignment_type}_processed_between_split.pkl"

    if between and not os.path.exists(e_filename):
        __prep_between_split_list(dataset, lexical)
        
    elif not os.path.exists(c_filename):
        __prep_plain_split_list(dataset, lexical)


#############################################################################

def print_results(z, p, b, dir_name, between, analysis, target_gender, prime_gender):
    betas = [0, 1] if analysis == 2 else list(range(8))

    adj_pair_type = "between" if between else "plain"

    results_filename = f"{dir_name}/results_{adj_pair_type}.txt"

    with open(results_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        f.write("BETA:\tZ-SCORES\tP-VALUES\n")
        f.write("----------------------------------\n")

        for bb in betas:
            if p[bb] < 0.001: 
                f.write(">>> ")
            elif p[bb] < 0.01:
                f.write(">> ")
            elif p[bb] < 0.05:
                f.write("> ")

            f.write(f"{bb}\t{z[bb]:.3f}\t{p[bb]:.3f}\n")


    betas_filename = f"{dir_name}/betas_{adj_pair_type}.txt"

    with open(betas_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        for bb in betas:
            f.write(f"\tBETA_{bb}")

        f.write("\n----------------------------------\n")

        for bb in betas:
            f.write(f"\t{b[bb]:.3f}")
        f.write(f"\n")


def create_vocab(adj_pair_list):
    print("Creating vocab...")
    vocab = set()

    for _, prime, target, _ in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab


def prime_lists(target_gender, dataset, between, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"
    adj_pair_type = "between" if between else "plain"

    with open(f"../Corpora/{dataset}/{alignment_type}_processed_{adj_pair_type}_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf
