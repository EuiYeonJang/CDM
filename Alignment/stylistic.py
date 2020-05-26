import os
import argparse
import pickle as pkl
import numpy as np
import statsmodels.api as sm
from collections import Counter
import alignment_utils as au


CATEGORIES = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']


def create_liwc(vocab):
    """
    Parameters:
        vocab - set of vocab

    Returns:
        dictionary of liwc categories (key) and list of its words (value)
    """
    with open("liwc.pkl", "rb") as f:
        liwc = pkl.load(f)

    for cat in liwc:
        extension = list()
        asterisk = set()
        
        for w in liwc[cat]:
            if "*" in w:
                asterisk.add(w)
                extension.extend([vw for vw in vocab if (w == vw or vw.startswith(w[:-1]))])
        liwc[cat] |= (set(extension))
        liwc[cat] -= asterisk     

    return liwc


def create_apl(liwc, orig_apl):
    print("Creating LIWC adjacency pairs...")
    # inv_ind = {w: cat for w in liwc[cat] for cat in liwc}
    inv_ind = {w: cat for cat in liwc for w in liwc[cat]}

    liwc_apl = list()

    for g, p, t, plen in orig_apl:
        new_p = Counter([inv_ind[w] for w in p.elements() if w in inv_ind])
        new_t = Counter([inv_ind[w] for w in t.elements() if w in inv_ind])
        liwc_apl.append((g, new_p, new_t, plen))

    return liwc_apl


def prep_apl(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
    
    apl = pf_apl + pm_apl
    
    vocab = au.create_vocab(apl)

    liwc = create_liwc(vocab)

    liwc_apl = create_apl(liwc, apl)

    with open(f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(liwc_apl, f)

    return liwc_apl


def prep_apl_two(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
    
    female_prime_vocab = au.create_vocab(pf_apl)
    male_prime_vocab = au.create_vocab(pm_apl)

    female_prime_liwc = create_liwc(female_prime_vocab)
    male_prime_liwc = create_liwc(male_prime_vocab)


    female_prime_stylistic_apl = create_apl(female_prime_liwc, pf_apl)
    male_prime_stylistic_apl = create_apl(male_prime_liwc, pm_apl)


    with open(f"{SAVEDIR}/p_f_t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(female_prime_stylistic_apl, f)

    with open(f"{SAVEDIR}/p_m_t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(male_prime_stylistic_apl, f)


    return female_prime_stylistic_apl, male_prime_stylistic_apl


def create_predictors(apl, eq):
    print("Creating predictors...")
    y = {w: [ 1 if target[w] > 0 else 0 for _, _, target, _ in apl] for w in CATEGORIES}

    if eq == 1:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}
        c_gender = {w: [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in CATEGORIES}

        return c_count, c_gender, y
    
    elif eq == 2:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}

        return c_count, y

    elif eq == 3: 
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}
        c_gender = {w: [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in CATEGORIES}
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y


def calculate_beta_one(apl):
    c_count, c_gender, y = create_predictors(apl, 1)

    pvalue_dict = {w: {i: "undefined" for i in range(4)} for w in CATEGORIES}
    zscores_dict = {w: {i: "undefined" for i in range(4)} for w in CATEGORIES}
    betas_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}


    for w in CATEGORIES:
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

            betas = res.params
            for i, b in enumerate(betas):
                betas_dict[w][i] = b

    return zscores_dict, pvalue_dict, betas_dict
       

def calculate_beta_two(apl):
    c_count, y = create_predictors(apl, 2)
    
    pvalue_dict = {w: {i: "undefined" for i in range(2)} for w in CATEGORIES}
    zscores_dict = {w: {i: "undefined" for i in range(2)} for w in CATEGORIES}
    betas_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}


    for w in CATEGORIES:
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

            betas = res.params
            for i, b in enumerate(betas):
                betas_dict[w][i] = b
    
    return zscores_dict, pvalue_dict, betas_dict


def calculate_beta_three(apl):
    c_count, c_gender, c_plen, y = create_predictors(apl, 3)

    pvalue_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}
    zscores_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}
    betas_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}


    for w in CATEGORIES:
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

            betas = res.params
            for i, b in enumerate(betas):
                betas_dict[w][i] = b
    return zscores_dict, pvalue_dict, betas_dict


def calculate_alignment(apl, eq, target_gender, prime="f"):
    print(f"Calculating alignment for eq {eq}, p_{prime}_t_{target_gender}")
    if eq == 1:
        z, p, b = calculate_beta_one(apl)
    elif eq == 2:
        z, p, b = calculate_beta_two(apl)
    else:
        z, p, b = calculate_beta_three(apl)

    results_filename = f"./stylistic_{ARGS.dataset}/results_between.txt" if ARGS.between else f"./stylistic_{ARGS.dataset}/results_orig.txt"

    with open(results_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EQUATION: {ARGS.analysis}\nTARGET GENDER: {target_gender}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")
        betas = [0, 1, 2, 3] if eq == 1 else [0, 1] if eq == 2 else [0, 1, 2, 3, 4, 5, 6, 7]

        for b in betas:
            f.write(f"\nBETA: {b}\n")

            f.write("CATEGORIES:\tZ-SCORES\tP-VALUES\n")
            f.write("----------------------------------\n")

            for w in z:
                if p[w][b] < 0.05: f.write(">> ")
                f.write(f"{w}:\t{z[w][b]:.3f}\t{p[w][b]:.3f}\n")\

    betas_filename = f"./stylistic_{ARGS.dataset}/betas_between.txt" if ARGS.between else f"./stylistic_{ARGS.dataset}/betas_orig.txt"

    with open(betas_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EQUATION: {ARGS.analysis}\nTARGET GENDER: {target_gender}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")
        betas = [0, 1, 2, 3] if eq == 1 else [0, 1] if eq == 2 else [0, 1, 2, 3, 4, 5, 6, 7]

        f.write("CATEGORIES:")
        for bb in betas:
            f.write(f"\tBETA_{bb}")

        f.write("\n----------------------------------\n")

        for w in z:
            f.write(f"{w}")
            for bb in betas:
                f.write(f"\t{b[w][bb]:.3f}")
            f.write(f"\n")



def main():
    au.prep(ARGS.dataset)

    os.makedirs(SAVEDIR, exist_ok=True)
    for target_gender in ["m", "f"]:
        if ARGS.analysis == 1 or ARGS.analysis == 3:
            filename = f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl"

            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    adj_pair_list = pkl.load(f)
            else:
                adj_pair_list = prep_apl(target_gender)

            calculate_alignment(adj_pair_list, ARGS.analysis, target_gender)

        elif ARGS.analysis == 2:
            filename = f"{SAVEDIR}/p_f_t_{target_gender}_prepped_apl.pkl"
            
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    female_prime_apl = pkl.load(f)

                with open(f"{SAVEDIR}/p_m_t_{target_gender}_prepped_apl.pkl", "rb") as f:
                    male_prime_apl = pkl.load(f)
                
            else:
                female_prime_apl, male_prime_apl = prep_apl_two(target_gender)

            calculate_alignment(female_prime_apl, ARGS.analysis, target_gender)
            calculate_alignment(male_prime_apl, ARGS.analysis, target_gender, prime="m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for stylistic alignment.')
    parser.add_argument('--dataset', type=str, default="AMI",
                        help="\"AMI\" or \"ICSI\", name of the dataset being analysed")
    parser.add_argument('--analysis', type=int, default=1,
						help="1, 2 or 3, the type of analysis to perform (equation number)")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate utterances or not, default True")

    ARGS = parser.parse_args()
    
    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SAVEDIR = f"./stylistic_{ARGS.dataset}/between" if ARGS.between else f"./stylistic_{ARGS.dataset}/orig"

    main()