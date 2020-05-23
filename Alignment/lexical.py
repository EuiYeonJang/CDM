import pickle as pkl
import numpy as np
import os
import statsmodels.api as sm
import argparse
import more_itertools.more

from collections import Counter
import alignment_utils as au


def prep_apl(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)

    apl = pf_apl + pm_apl

    vocab = au.create_vocab(apl)

    with open(f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(apl, f)

    with open(f"{SAVEDIR}/t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(vocab, f)

    return apl, vocab


def prep_apl_two(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)

    female_prime_vocab = au.create_vocab(pf_apl)
    male_prime_vocab = au.create_vocab(pm_apl)

    with open(f"{SAVEDIR}/p_f_t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(female_prime_vocab, f)

    with open(f"{SAVEDIR}/p_m_t_{target_gender}_vocab.pkl", "wb") as f:
        pkl.dump(male_prime_vocab, f)


    return pf_apl, pm_apl, female_prime_vocab, male_prime_vocab


def create_predictors(apl, vocab, eq):
    print("Creating predictors...")
    y = {w: [ 1 if target[w] > 0 else 0 for _, _, target, _ in apl] for w in vocab}

    if eq == 1:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in vocab}
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


def calculate_beta_one(apl, vocab):
    c_count, c_gender, y = create_predictors(apl, vocab, 1)

    print("Calculating betas...")
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

            try:
                res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
                
                p_values = res.pvalues
                for i, p in enumerate(p_values):
                    pvalue_dict[w][i] = p

                z_scores = res.tvalues
                for i, z in enumerate(z_scores):
                    zscores_dict[w][i] = z
            except:
                pass
    
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
            
            try:
                res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
                
                p_values = res.pvalues
                for i, p in enumerate(p_values):
                    pvalue_dict[w][i] = p

                z_scores = res.tvalues
                for i, z in enumerate(z_scores):
                    zscores_dict[w][i] = z
            except:
                pass
    
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

            try:
                res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
                
                p_values = res.pvalues
                for i, p in enumerate(p_values):
                    pvalue_dict[w][i] = p

                z_scores = res.tvalues
                for i, z in enumerate(z_scores):
                    zscores_dict[w][i] = z
            except:
                pass

    return zscores_dict, pvalue_dict


def calculate_alignment(apl, vocab, eq, target_gender, prime="f"):
    print(f"Calculating alignment for eq {eq}, p_{prime}_t_{target_gender}")
    betas = [0, 1, 2, 3] if eq == 1 else [0, 1] if eq == 2 else [0, 1, 2, 3, 4, 5, 6, 7]
    
    beta_track = {b: {"triple_count": 0, "triple_list": set(), "double_count": 0, "double_list": set(), "single_count": 0, "single_list": set()} for b in betas}


    if eq == 1:
        z, p = calculate_beta_one(apl, vocab)
    elif eq == 2:
        z, p = calculate_beta_two(apl, vocab)
    else:
        z, p = calculate_beta_three(apl, vocab)

    for b in betas:
        for w in z:
            if not z[w][b] == "undefined":
                if p[w][b] < 0.001: 
                    beta_track[b]["triple_count"] += 1
                    beta_track[b]["triple_list"].add(w)
                elif p[w][b] < 0.01: 
                    beta_track[b]["double_count"] += 1
                    beta_track[b]["double_list"].add(w)
                elif p[w][b] < 0.05: 
                    beta_track[b]["single_count"] += 1
                    beta_track[b]["single_list"].add(w)

    results_filename = f"./lexical_{ARGS.dataset}/results_between.txt" if ARGS.between else f"./lexical_{ARGS.dataset}/results_orig.txt"

    with open(results_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EQUATION: {ARGS.analysis}\nTARGET GENDER: {target_gender}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")

        for b in betas:
            f.write(f"\nBETA: {b}\n")

            f.write("P-VALUES \tNR OF WORDS\n")
            f.write("----------------------------------\n")

            f.write("0.001:\t{}\n{}\n".format(beta_track[b]["triple_count"], beta_track[b]["triple_list"] if len(beta_track[b]["triple_list"]) > 0 else {}))
            f.write("0.01:\t{}\n{}\n".format(beta_track[b]["double_count"], beta_track[b]["double_list"] if len(beta_track[b]["double_list"]) > 0 else {}))
            f.write("0.05:\t{}\n{}\n".format(beta_track[b]["single_count"], beta_track[b]["single_list"] if len(beta_track[b]["single_list"]) > 0 else {}))


def main():
    au.prep(ARGS.dataset)

    os.makedirs(SAVEDIR, exist_ok=True)
    for target_gender in ["m", "f"]:

        if ARGS.analysis == 1 or ARGS.analysis == 3:
            filename = f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl"

            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    adj_pair_list = pkl.load(f)

                with open(f"{SAVEDIR}/t_{target_gender}_vocab.pkl", "rb") as f:
                    vocab = pkl.load(f)
            else:
                adj_pair_list, vocab = prep_apl(target_gender)

            # calculate_alignment(adj_pair_list, vocab, ARGS.analysis, target_gender)
            print(f"Analysis {ARGS.analysis} target {target_gender} vocab length = {len(vocab)}")

        elif ARGS.analysis == 2:
            filename_f = f"{SAVEDIR}/p_f_t_{target_gender}_vocab.pkl"
            filename_m = f"{SAVEDIR}/p_m_t_{target_gender}_vocab.pkl"
            
            if os.path.exists(filename_f) and os.path.exists(filename_m):
                with open(f"{SAVEDIR}/p_f_t_{target_gender}_vocab.pkl", "rb") as f:
                    female_prime_vocab = pkl.load(f)

                with open(f"{SAVEDIR}/p_m_t_{target_gender}_vocab.pkl", "rb") as f:
                    male_prime_vocab = pkl.load(f)

                female_prime_apl, male_prime_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
                
            else:
                female_prime_apl, male_prime_apl, female_prime_vocab, male_prime_vocab = prep_apl_two(target_gender)

            print(f"Analysis {ARGS.analysis} target {target_gender} female prime vocab length = {len(female_prime_vocab)}")
            print(f"Analysis {ARGS.analysis} target {target_gender} male prime vocab length = {len(male_prime_vocab)}")

            # calculate_alignment(female_prime_apl, female_prime_vocab, ARGS.analysis, target_gender)

            # calculate_alignment(male_prime_apl, male_prime_vocab, ARGS.analysis, target_gender, prime="m")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for lexical alignment.')
    parser.add_argument('--dataset', type=str, default="AMI",
						help="\"AMI\" or \"ICSI\", name of the dataset being analysed.")
    parser.add_argument('--analysis', type=int, default=1,
						help="1, 2 or 3, the type of analysis to perform (equation number)")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate utterances or not, default True")


    ARGS = parser.parse_args()

    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SAVEDIR = f"./lexical_{ARGS.dataset}/between" if ARGS.between else f"./lexical_{ARGS.dataset}/orig"

    main()