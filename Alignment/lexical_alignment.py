import pickle as pkl
import numpy as np
import os
import statsmodels.api as sm
import argparse
import itertools

from collections import Counter
import alignment_utils as au

VOCAB = set()

def create_vocab(adj_pair_list):
    """
    Parameters: 
        adj_pair_list -list of tuples (gender of prime, prime counter, target counter)
    
    Returns:
        set of vocab
    """
    vocab = set()

    for _, prime, target, _ in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab

def prep_in_between_apl(target_gender):
    global VOCAB
    female_prime_list, male_prime_list = au.prime_lists(target_gender)

    apl = list()

    for pair in female_prime_list:
        combined = pair["a"] + pair["fb"]
        if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
            apl.append(("f", combined , pair["b"], sum(combined.values())))


    for pair in male_prime_list:
        combined = pair["a"] + pair["mb"]
        if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
            apl.append(("m",  combined, pair["b"], sum(combined.values())))


    VOCAB = create_vocab(apl)

    with open(f"./lexical/t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(apl, f)

    with open(f"./lexical/t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(VOCAB, f)

    return apl


def prep_in_between_apl_two(target_gender):
    global VOCAB

    female_prime_list, male_prime_list = au.prime_lists(target_gender)
    female_prime_apl = list()

    for pair in female_prime_list:
        combined = pair["a"] + pair["fb"]
        if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
            female_prime_apl.append(("f", combined , pair["b"], sum(combined.values())))

    male_prime_apl = list()
    for pair in male_prime_list:
        combined = pair["a"] + pair["mb"]
        if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
            male_prime_apl.append(("m",  combined, pair["b"], sum(combined.values())))

    female_prime_vocab = create_vocab(female_prime_apl)
    male_prime_vocab = create_vocab(male_prime_apl)

    with open(f"./stylistic/p_f_t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(female_prime_apl, f)

    with open(f"./stylistic/p_m_t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(male_prime_apl, f)

    with open(f"./stylistic/p_f_t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(female_prime_vocab, f)

    with open(f"./stylistic/p_m_t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(male_prime_vocab, f)


    return female_prime_apl, male_prime_apl, female_prime_vocab, male_prime_vocab


def partition_vocab(n_partitions=2):
    global VOCAB

    print(len(VOCAB))
    partition_size = int(len(VOCAB)/n_partitions)
    print(partition_size)
    for partition in itertools.islice(VOCAB, 0, None, partition_size):
        yield partition


def create_predictors(apl, vocab, eq, normalise=False):
    y = {w: [ 1 if target[w] > 0 else 0 for _, _, target, _ in apl] for w in vocab}

    if eq == 1:
        c_count = {w: [prime[w]/plen if normalise else prime[w] for _, prime, _, plen in apl] for w in vocab}
        c_gender = {w: [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in vocab}

        return c_count, c_gender, y
    
    elif eq == 2:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in vocab}

        return c_count, y

    elif eq == 3: 
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in vocab}
        c_gender = {w: [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in vocab}
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y


def calculate_beta_one(apl, vocab, normalise=False):
    c_count, c_gender, y = create_predictors(apl, vocab, 1, normalise)

    pvalue_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}
    zscores_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}

    for w in vocab:
        c_w = c_count[w]
        g_w = c_gender[w]
        y_w = y[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            g_w = np.array(g_w)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_w*g_w]).T

            y_w = np.array(y_w)

            res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            
            p_values = res.pvalues
            for i, p in enumerate(p_values):
                pvalue_dict[w][i] = p

            z_scores = res.tvalues
            for i, z in enumerate(z_scores):
                zscores_dict[w][i] = z
    
    return zscores_dict, pvalue_dict


def calculate_beta_two(apl, vocab):
    c_count, y = create_predictors(apl, vocab, 2)
    
    pvalue_dict = {w: {i: "undefined" for i in range(2)} for w in vocab}
    zscores_dict = {w: {i: "undefined" for i in range(2)} for w in vocab}

    for w in vocab:
        c_w = c_count[w]
        y_w = y[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            X = np.array([np.ones(len(c_w)), c_w]).T

            y_w = np.array(y_w)

            res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            
            p_values = res.pvalues
            for i, p in enumerate(p_values):
                pvalue_dict[w][i] = p

            z_scores = res.tvalues
            for i, z in enumerate(z_scores):
                zscores_dict[w][i] = z
    
    return zscores_dict, pvalue_dict


def calculate_beta_three(apl, vocab):
    c_count, c_gender, c_plen, y = create_predictors(apl, vocab, 3)

    pvalue_dict = {w: {i: "undefined" for i in range(8)} for w in vocab}
    zscores_dict = {w: {i: "undefined" for i in range(8)} for w in vocab}

    for w in vocab:
        c_w = c_count[w]
        g_w = c_gender[w]
        y_w = y[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            g_w = np.array(g_w)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_plen, c_w*g_w, c_w*c_plen, g_w*c_plen, c_w*g_w*c_plen]).T
    
            y_w = np.array(y_w)

            res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            
            p_values = res.pvalues
            for i, p in enumerate(p_values):
                pvalue_dict[w][i] = p

            z_scores = res.tvalues
            for i, z in enumerate(z_scores):
                zscores_dict[w][i] = z

    return zscores_dict, pvalue_dict


def calculate_alignment(apl, eq, prime="f", normalise=False):
    betas = [3] if eq == 1 else [1] if eq == 2 else [4, 5, 6, 7]
    
    beta_track = {b: {"triple_count": 0, "triple_list": set(), "double_count": 0, "double_list": set(), "single_count": 0, "single_list": set()} for b in betas}

    for p_vocab in partition_vocab():
        if eq == 1:
            z, p = calculate_beta_one(apl, p_vocab, normalise)
        elif eq == 2:
            z, p = calculate_beta_two(apl, p_vocab)
        else:
            z, p = calculate_beta_three(apl, p_vocab)

        for b in betas:
            for w in z:
                print(z[w][b])
                if p[w][b] < 0.001: 
                    beta_track[b]["triple_count"] += 1
                    beta_track[b]["triple_list"].add(w)
                elif p[w][b] < 0.01: 
                    beta_track[b]["double_count"] += 1
                    beta_track[b]["double_list"].add(w)
                elif p[w][b] < 0.05: 
                    beta_track[b]["single_count"] += 1
                    beta_track[b]["single_list"].add(w)


    with open("./lexical/results.txt", "a") as f:
        f.write("\n==================================\n")
        f.write(f"TARGET GENDER: {args.target_gender}\nEQUATION: {args.analysis}\n")
        
        if eq == 1 : f.write(f"NORMALISE: {args.normalise}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")
        betas = [3] if eq == 1 else [1] if eq == 2 else [4, 5, 6, 7]

        for b in betas:
            f.write(f"\nBETA: {b}\n")

            f.write("P-VALUES \tNR OF WORDS\n")
            f.write("----------------------------------\n")

            f.write("0.001:\t{}\n{}\n".format(beta_track[b]["triple_count"], beta_track[b]["triple_list"] if len(beta_track[b]["triple_list"]) > 0 else {}))
            f.write("0.01:\t{}\n{}\n".format(beta_track[b]["double_count"], beta_track[b]["double_list"] if len(beta_track[b]["double_list"]) > 0 else {}))
            f.write("0.05:\t{}\n{}\n".format(beta_track[b]["single_count"], beta_track[b]["single_list"] if len(beta_track[b]["single_list"]) > 0 else {}))


def main():
    global VOCAB

    os.makedirs("./lexical/", exist_ok=True)

    if args.analysis == 1 or args.analysis == 3:
        filename = f"./lexical/t_{args.target_gender}_prepped_apl.pkl"

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                adj_pair_list = pkl.load(f)

            with open(f"./lexical/t_{args.target_gender}_vocab.pkl", "rb") as f:
                VOCAB = pkl.load(f)
        else:
            adj_pair_list = prep_in_between_apl(args.target_gender)

        calculate_alignment(adj_pair_list, args.analysis, normalise=args.normalise)

    elif args.analysis == 2:
        filename = f"./lexical/p_f_t_{args.target_gender}_prepped_apl.pkl"
        
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                female_prime_apl = pkl.load(f)

            with open(f"./lexical/p_m_t_{args.target_gender}_prepped_apl.pkl", "rb") as f:
                male_prime_apl = pkl.load(f)

            with open(f"./lexical/p_f_t_{args.target_gender}_vocab.pkl", "rb") as f:
                VOCAB = pkl.load(f)

            with open(f"./lexical/p_m_t_{args.target_gender}_vocab.pkl", "rb") as f:
                male_prime_vocab = pkl.load(f)
            
        else:
            female_prime_apl, male_prime_apl, VOCAB, male_prime_vocab = prep_in_between_apl_two(args.target_gender)

        calculate_alignment(female_prime_apl, args.analysis)

        VOCAB = male_prime_vocab

        calculate_alignment(male_prime_apl, args.analysis, prime="m")



args = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for lexical alignment.')
    parser.add_argument('--target_gender', type=str, default="m",
						help="\"m\" or \"f\", gender of target speaker.")
    parser.add_argument('--analysis', type=int, default=1,
						help="1, 2 or 3, the type of analysis to perform (equation number)")
    parser.add_argument('--normalise', type=bool, default=False,
                        help="to normalise c_count for equation 1")

    args = parser.parse_args()

    print("Attention: Make sure you're in the 'Alignment' directory before running code!")

    main()